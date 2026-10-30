import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from ibicus.debias import ISIMIP, QuantileDeltaMapping, ScaledDistributionMapping, LinearScaling
from scipy.stats import norm, laplace, gamma, beta
from ibicus.variables import *
from ibicus.utils import gen_PrecipitationHurdleModel, gen_PrecipitationGammaLeftCensoredModel
from ibicus.debias import ECDFM, QuantileMapping, DeltaChange, CDFt
from ibicus.debias import QuantileDeltaMapping, ScaledDistributionMapping, LinearScaling
from ibicus.evaluate import assumptions, correlation, marginal, multivariate, trend
from ibicus.evaluate.metrics import *
import os

# Configuration
base_dir = Path('/scratch/u/u301861/THESIS/HighRESMIP')
domains = ['HadGEM3-HM'] #, 'MPI-XR','HadGEM3-HM']
scenarios = {
    'rcp85': ('2015-01-01', '2050-12-31'),
    # Add other scenarios as needed
    # 'rcp45': ('2020-01-01', '2099-12-31'),
    # 'rcp26': ('1998-01-01', '2099-12-31')
}
variables = ['prsnratio'] #'hurs', 'sfcWind', 'rsds', 'rlds'] #, 'tasrange']

def setup_logging(domain):
    """Setup logging for the domain"""
    new_base_dir = Path("/scratch/u/u301861/THESIS/HighRESMIP")
    log_dir = new_base_dir / domain / f'logs.{domain}'
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log_file = log_dir / 'bias_correction_detailed.log'
    error_log_file = log_dir / 'bias_correction_errors.log'

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file),
            logging.StreamHandler()
        ]
    )

    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    class ErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.ERROR

    error_handler.addFilter(ErrorFilter())
    logging.getLogger().addHandler(error_handler)

def load_and_preprocess_data(file_path, variable_name, domain=None):
    """Load and preprocess data without spatial domain selection"""
    try:
        data = xr.open_dataset(file_path)[variable_name]
        data = data.sortby('lat')

        logging.info(f"Loaded {variable_name} from {file_path}")
        logging.info(f"Data shape: {data.shape}")

        return data

    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        raise

def check_dimensions(obs, hist, fut):
    """Check that spatial dimensions match between datasets"""
    spatial_dims = ['lat', 'lon']
    for dim in spatial_dims:
        if not (obs.sizes[dim] == hist.sizes[dim] == fut.sizes[dim]):
            logging.error(f"Spatial dimension mismatch for {dim}: obs {obs.sizes[dim]}, hist {hist.sizes[dim]}, fut {fut.sizes[dim]}")
            raise ValueError(f"Input datasets have inconsistent {dim} dimensions")

    # Check that future dataset has timesteps
    if fut.sizes['time'] < 1:
        logging.error(f"Future time dimension is empty: fut {fut.sizes['time']}")
        raise ValueError("Future dataset has no timesteps")

    logging.info(f"Dimensions checked: obs {obs.sizes}, hist {hist.sizes}, fut {fut.sizes}")

def preprocess_data(data, variable):
    """Preprocess data with variable-specific treatments"""
    # Fill None or missing values with the mean along the time dimension
    data = data.fillna(data.mean(dim="time"))

    # Interpolate remaining NaNs (if any) linearly along the time dimension
    data = data.interpolate_na(dim='time', method='linear', fill_value=0)

    # Ensure all NaNs are filled with the overall mean
    overall_mean = data.mean()
    data = data.fillna(overall_mean)

    if variable == 'pr':
        # Remove trace amounts of precipitation
        upper_bound = 0.0069444444
        data = data.where(data <= upper_bound, upper_bound)

    # Apply variable-specific bounds
    bounds = {
        'pr': (0, 0.0069444444),
        'prsnratio': (0, 1),
        'tasrange': (0, None),
        'tasskew': (0, 1),
        'rsds': (0, 500),
        'rlds': (40, 600),
        'hurs': (1, 100),
        'sfcWind': (0.1, 50)
    }

    if variable in bounds:
        lower, upper = bounds[variable]
        if lower is not None:
            data = data.clip(min=lower)
        if upper is not None:
            data = data.clip(max=upper)

    logging.info(f"Preprocessed {variable} data. Shape: {data.shape}, Min: {data.min().item():.2f}, Max: {data.max().item():.2f}")
    return data

def apply_bias_correction(obs, hist, fut, variable):
    """Apply bias correction using ISIMIP method"""
    obs_time = pd.to_datetime(obs.time.values)
    hist_time = pd.to_datetime(hist.time.values)
    fut_time = pd.to_datetime(fut.time.values)

    # Configure debiaser based on variable
    if variable == 'pr':
        debiaser = ISIMIP.from_variable(
            variable,
            lower_bound=0,
            upper_bound=0.0069444444,
            trend_preservation_method="multiplicative",
            running_window_mode=True,
            running_window_length=31
        )
    else:
        debiaser = ISIMIP.from_variable(
            variable,
            running_window_mode=True,
            running_window_length=31
        )

    try:
        num_processes = min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 128)))

        # Correct historical data
        corrected_hist = debiaser.apply(
            obs.values, hist.values, hist.values,
            time_obs=obs_time, time_cm_hist=hist_time, time_cm_future=hist_time,
            parallel=True, nr_processes=num_processes
        )

        # Correct future data
        corrected_fut = debiaser.apply(
            obs.values, hist.values, fut.values,
            time_obs=obs_time, time_cm_hist=hist_time, time_cm_future=fut_time,
            parallel=True, nr_processes=num_processes
        )

        return corrected_hist, corrected_fut

    except Exception as e:
        logging.error(f"Error in bias correction for {variable}: {str(e)}")
        raise

def get_cf_standard_name(var):
    cf_names = {
        'tas': 'air_temperature',
        'tasmin': 'air_temperature',
        'tasmax': 'air_temperature',
        'pr': 'precipitation_flux',
        'rsds': 'surface_downwelling_shortwave_flux_in_air',
        'rlds': 'surface_downwelling_longwave_flux_in_air',
        'sfcWind': 'wind_speed',
        'hurs': 'relative_humidity'
    }
    return cf_names.get(var, '')

def get_cf_long_name(var):
    long_names = {
        'tas': 'Near-Surface Air Temperature',
        'tasmin': 'Daily Minimum Near-Surface Air Temperature',
        'tasmax': 'Daily Maximum Near-Surface Air Temperature',
        'pr': 'Precipitation',
        'rsds': 'Surface downwelling shortwave radiation',
        'rlds': 'Surface downwelling longwave radiation',
        'sfcWind': 'Near-surface (usually 10 m) wind speed',
        'hurs': 'Near-surface (usually 2 m) relative humidity'
    }
    return long_names.get(var, '')

def get_cf_units(var):
    units = {
        'tas': 'K',
        'tasmin': 'K',
        'tasmax': 'K',
        'pr': 'kg m-2 s-1',
        'rsds': 'W m⁻²',
        'rlds': 'W m⁻²',
        'sfcWind': 'm s⁻¹',
        'hurs': '%'
    }
    return units.get(var, '')

def update_metadata(ds, var, scenario, domain, input_files):
    """Update dataset metadata"""
    ds.attrs.update({
        'title': f'Bias-Adjusted {var} data for {scenario} scenario in {domain} domain using ISIMIP3BASD',
        'product': 'bias-adjusted-output',
        'bc_method': 'ISIMIP3BASD method for bias adjustment',
        'bc_observation': 'W5E5 v2 observational climate input data for ISIMIP3a',
        'bc_period': 'Hist:1979-2014, Fut: 2015-2050',
        'creation_date': pd.Timestamp.now().isoformat(),
        'created_by': 'Yakubu Fuseini',
        'contact': 'fuseini.yakubu@studium.uni-hamburg.de',
        'institute': 'Universität Hamburg',
        'references': ('Lange, S. (2019): Trend-preserving bias adjustment and statistical '
                      'downscaling with ISIMIP3BASD (v1.0), Geosci. Model Dev., 12, 3055–3070'),
        'input_files': ', '.join(str(file) for file in input_files),
        'Conventions': 'CF-1.7',
        'GCM': 'HadGEM3-GC31-HM, r1i1p1f1',
        'driving_experiment': 'CMIP6, HighResMIP',
        'Domain': f'{domain}, 50 Km',
        'standard_name': get_cf_standard_name(var),
        'long_name': get_cf_long_name(var),
        'units': get_cf_units(var)
    })
    return ds

def save_corrected_data(corrected_data, original_data, variable, scenario, domain, output_dir, input_files):
    """Save bias-corrected data to NetCDF"""
    start_year = original_data.time.dt.year.min().item()
    end_year = original_data.time.dt.year.max().item()

    output_file_path = output_dir / f"{scenario}_{variable}_{domain}_50_{start_year}-{end_year}_debiasedN.nc"

    # Create dataset with correct coordinates
    corrected_ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), corrected_data)},
        coords={
            "time": original_data.time,
            "lat": original_data.lat,
            "lon": original_data.lon
        }
    )

    # Update metadata
    corrected_ds = update_metadata(corrected_ds, variable, scenario, domain, input_files)

    # Save to NetCDF
    corrected_ds.to_netcdf(output_file_path)
    logging.info(f"Saved debiased {variable} for {domain}, scenario: {scenario} ({start_year}-{end_year})")

def process_historical_data(obs_data, hist_data, variable, domain, output_dir, obs_file, hist_file):
    """Process historical data - separated for easy commenting out"""
    try:
        logging.info(f"Processing historical data for {variable} in {domain}")
        corrected_hist, _ = apply_bias_correction(obs_data, hist_data, hist_data, variable)
        save_corrected_data(corrected_hist, hist_data, variable, 'hist', domain, output_dir, [obs_file, hist_file])
    except Exception as e:
        logging.error(f"Error processing historical data for {variable} in {domain}: {str(e)}")

def main():
    """Main processing function"""
    base_dir = Path("/work/bu1501/u301861/thesis/datadir/HighRESMIP")

    for domain in domains:
        setup_logging(domain)

        # Create output directory
        output_dir = base_dir / domain / 'debiased_output'
        output_dir.mkdir(parents=True, exist_ok=True)

        for variable in variables:
            logging.info(f"Processing {variable} for {domain}")

            # Define file paths
            obs_file = base_dir / f"Obs_{variable}_44_1979-2014.nc"
            hist_file = base_dir / domain / f"hist_{variable}_{domain}_50_1979-2014.nc"

            try:
                # Load and preprocess observational and historical data
                obs_data = load_and_preprocess_data(obs_file, variable, domain)
                hist_data = load_and_preprocess_data(hist_file, variable, domain)

                obs_data = preprocess_data(obs_data, variable)
                hist_data = preprocess_data(hist_data, variable)

                # Check dimensions
                check_dimensions(obs_data, hist_data, hist_data)

                # Process historical data (comment out this line if you don't want historical correction)
#                process_historical_data(obs_data, hist_data, variable, domain, output_dir, obs_file, hist_file)

                # Process future scenarios
                for scenario, (start_date, end_date) in scenarios.items():
                    fut_file = base_dir / domain / f"{scenario}_{variable}_{domain}_50_2015-2050.nc"

                    if fut_file.exists():
                        logging.info(f"Processing {scenario} scenario for {variable} in {domain}")

                        # Load future data
                        fut_data = load_and_preprocess_data(fut_file, variable, domain)
                        fut_data = preprocess_data(fut_data, variable)

                        # Check dimensions
                        check_dimensions(obs_data, hist_data, fut_data)

                        # Apply bias correction
                        _, corrected_fut = apply_bias_correction(obs_data, hist_data, fut_data, variable)

                        # Save corrected future data
                        save_corrected_data(corrected_fut, fut_data, variable, scenario, domain, output_dir, [obs_file, hist_file, fut_file])

                    else:
                        logging.warning(f"Future scenario file not found: {fut_file}")

            except Exception as e:
                logging.error(f"Error processing {variable} for {domain}: {str(e)}")

if __name__ == "__main__":
    main()
