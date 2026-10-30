import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import xarray as xr
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import subprocess
import tempfile
import shutil

# Set better default parameters for publication-quality figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def create_tasmax_colormap():
    """Discrete colormap: blue only for cooling (<0), yellow-red for warming (>0)."""
    colors = [
        '#9ecae1',  # -1 to 0  (only blue bin)
        '#ffffb2',  #  0 to 1
        '#fed976',  #  1 to 2
        '#feb24c',  #  2 to 3
        '#fc4e2a',  #  3 to 4
        '#bd0026',  #  4 to 5
    ]
    cmap = ListedColormap(colors, name='tasmax_discrete')
    return cmap.with_extremes(over='#7f0000')  # arrow above 5 °C

#def create_shifted_colormap():
#    """Create a custom colormap that starts from light blue instead of dark blue."""
    # Get the RdYlBu_r colormap
#    original_cmap = cm.get_cmap('RdYlBu_r')

    # Sample colors from the original colormap, skipping the dark blue portion
    # We want to start from where light blue would be (around 0.15 of the colormap)
    # and go to the end (red)
#    colors = original_cmap(np.linspace(0.15, 1.0, 256))

    # Create new colormap
#    custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlBu', colors)

#    return custom_cmap

# Base path for all models
BASE_PATH = "/work/ch0636/u301861/phd/HighRESMIP"

# Mask and grid files
MASK_FILE = "/work/ch0636/u301861/phd/HighRESMIP/MPI-XR/hist_pr_MPI-XR_50_1979-2014_landmon.nc"
GRID_FILE = "/work/ch0636/u301861/phd/HighRESMIP/results/compoundXTREMe_3/OBS/6months/global_obgrid.txt"
EC_GRID_FILE = "/work/ch0636/u301861/phd/HighRESMIP/HadGEM3-HM/debiased_output/newfiles1/SIGNAL/ecgrid.txt"

# Define model information
MODEL_INFO = {
    'CNRM-CERFACS': {
        'folder': 'CNRM-CERFACS',
        'display_name': 'CNRM',
        'needs_setgrid': False,
        'grid_file': None,
    },
    'EC-Earth3P-HR': {
        'folder': 'EC-Earth-HR',
        'display_name': 'EC-Earth3P',
        'needs_setgrid': True,
        'grid_file': EC_GRID_FILE,
    },
    'HadGEM3-GC31-HM': {
        'folder': 'HadGEM3-HM',
        'display_name': 'HadGEM3',
        'needs_setgrid': True,
        'grid_file': None,  # Will try to infer or use default
    },
    'MPI-ESM1-2-XR': {
        'folder': 'MPI-XR',
        'display_name': 'MPI-XR',
        'needs_setgrid': False,
        'grid_file': None,
    }
}

# Define variable settings
VAR_SETTINGS = {
    'tasmax': {
        'display_name': 'tasmax',
        'units': r'$^\circ$C',
        'cmap': create_tasmax_colormap(),
#        'cmap': create_shifted_colormap(),  # Use custom colormap
        'range': (-1, 5),
        'levels': [-1, 0, 1, 2, 3, 4, 5],
    },
    'pr': {
        'display_name': 'pr',
        'units': r'mm d$^{-1}$',
        'cmap': 'PiYG',
        'range': (-1, 1),
        'levels': [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
    }
}

def run_cdo_command(command, verbose=True):
    """Run a CDO command and handle errors.

    Args:
        command: List of command arguments
        verbose: Print command and output

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"  Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if verbose and result.stdout:
            print(f"  Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error running CDO command: {e}")
        print(f"  stderr: {e.stderr}")
        return False

def create_land_mask(mask_file, grid_file, temp_dir):
    """Create and regrid land mask from mask file.

    Args:
        mask_file: Path to the mask netCDF file
        grid_file: Path to the CDO grid definition file
        temp_dir: Temporary directory for intermediate files

    Returns:
        Tuple of (lats, lons, mask) or (None, None, None) if failed
    """
    print("\nCreating land mask...")

    # Check if files exist
    if not os.path.exists(mask_file):
        print(f"Error: Mask file not found: {mask_file}")
        return None, None, None

    if not os.path.exists(grid_file):
        print(f"Error: Grid file not found: {grid_file}")
        return None, None, None

    # Create temporary file for regridded mask
    regridded_mask_file = os.path.join(temp_dir, "mask_regridded.nc")

    # Extract first time step and regrid
    print("  Extracting first time step and regridding mask...")
    cmd = ["cdo", "remapbil," + grid_file,
           "-seltimestep,1", mask_file, regridded_mask_file]

    if not run_cdo_command(cmd):
        return None, None, None

    # Load the regridded mask
    try:
        ds = xr.open_dataset(regridded_mask_file)

        # Find the data variable (exclude coordinate variables and bounds)
        exclude_vars = ['time', 'lat', 'lon', 'latitude', 'longitude',
                       'time_bnds', 'lat_bnds', 'lon_bnds', 'latitude_bnds',
                       'longitude_bnds', 'bnds']

        data_vars = [v for v in ds.data_vars if v not in exclude_vars
                     and 'bnd' not in v.lower()]

        if not data_vars:
            print("Error: No data variables found in mask file")
            print(f"Available variables: {list(ds.variables)}")
            ds.close()
            return None, None, None

        var_name = data_vars[0]
        print(f"  Using variable '{var_name}' from mask file")

        # Extract data
        data = ds[var_name].values

        # Extract lat and lon
        if 'lat' in ds.coords:
            lats = ds.lat.values
        elif 'latitude' in ds.coords:
            lats = ds.latitude.values
        else:
            print("Error: Latitude dimension not found")
            print(f"Available coordinates: {list(ds.coords)}")
            ds.close()
            return None, None, None

        if 'lon' in ds.coords:
            lons = ds.lon.values
        elif 'longitude' in ds.coords:
            lons = ds.longitude.values
        else:
            print("Error: Longitude dimension not found")
            print(f"Available coordinates: {list(ds.coords)}")
            ds.close()
            return None, None, None

        ds.close()

        # If data has multiple dimensions, take the first slice
        print(f"  Data shape before processing: {data.shape}")
        while data.ndim > 2:
            data = data[0]
        print(f"  Data shape after processing: {data.shape}")

        # Create mask: True for land (valid values including 0), False for ocean (NaN)
        mask = ~np.isnan(data)

        print(f"  Mask created: {np.sum(mask)} land points out of {mask.size} total points")

        return lats, lons, mask

    except Exception as e:
        print(f"Error loading mask file: {str(e)}")
        return None, None, None

def find_signal_file(model_key, variable):
    """Find the SIGNAL file for a specific model and variable.

    Args:
        model_key: Model key from MODEL_INFO
        variable: Variable name (tasmax or pr)

    Returns:
        Path to the signal file or None if not found
    """
    model_folder = MODEL_INFO[model_key]['folder']
    base_path = os.path.join(BASE_PATH, model_folder, 'debiased_output', 'newfiles1', 'SIGNAL')

    # Signal files pattern: e.g., tasmaxAdjust_EC-Earth-HR_signal.nc
    pattern = f"{variable}Adjust_{model_folder}_signal.nc"
    file_path = os.path.join(base_path, pattern)

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    return file_path

def regrid_and_load_signal(file_path, variable, grid_file, temp_dir, model_key):
    """Regrid signal file to common grid and load data.

    Args:
        file_path: Path to the signal netCDF file
        variable: Variable name to extract
        grid_file: Path to the CDO grid definition file
        temp_dir: Temporary directory for intermediate files
        model_key: Model key to check if setgrid is needed

    Returns:
        Data array or None if failed
    """
    if file_path is None or not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    # Create temporary file names
    basename = os.path.basename(file_path)
    temp_setgrid = os.path.join(temp_dir, f"setgrid_{variable}_{basename}")
    temp_output = os.path.join(temp_dir, f"signal_regridded_{variable}_{basename}")

    # Check if model needs setgrid
    model_info = MODEL_INFO[model_key]
    needs_setgrid = model_info.get('needs_setgrid', False)
    model_grid_file = model_info.get('grid_file', None)

    # First, try to set grid if needed
    input_for_remap = file_path
    if needs_setgrid:
        if model_grid_file and os.path.exists(model_grid_file):
            print(f"    Setting grid using {os.path.basename(model_grid_file)}")
            cmd_setgrid = ["cdo", "setgrid," + model_grid_file, file_path, temp_setgrid]
            if run_cdo_command(cmd_setgrid, verbose=False):
                input_for_remap = temp_setgrid
            else:
                print(f"    Warning: setgrid failed, trying direct remapping")
        else:
            # Try griddes to see the grid structure
            print(f"    Attempting to use file's native grid")

    # Regrid using CDO
    cmd = ["cdo", "remapbil," + grid_file, input_for_remap, temp_output]

    if not run_cdo_command(cmd, verbose=False):
        # If remapping failed and we didn't try setgrid, try with sellonlatbox to regularize
        if not needs_setgrid:
            print(f"    Trying alternative: sellonlatbox first")
            temp_regular = os.path.join(temp_dir, f"regular_{variable}_{basename}")
            cmd_regular = ["cdo", "sellonlatbox,-180,180,-90,90", file_path, temp_regular]
            if run_cdo_command(cmd_regular, verbose=False):
                cmd = ["cdo", "remapbil," + grid_file, temp_regular, temp_output]
                if not run_cdo_command(cmd, verbose=False):
                    return None
            else:
                return None
        else:
            return None

    # Load the regridded data
    try:
        ds = xr.open_dataset(temp_output)

        # Try to find the variable in the dataset
        var_name = None
        if variable in ds.variables:
            var_name = variable
        else:
            # Try to find the variable with common variations
            for v in ds.variables:
                if variable.lower() in v.lower():
                    var_name = v
                    break

        if var_name is None:
            print(f"Warning: Variable {variable} not found in {temp_output}")
            print(f"Available variables: {list(ds.variables)}")
            ds.close()
            return None

        # Extract data
        data = ds[var_name].values
        ds.close()

        # If data has multiple dimensions, take the first slice or mean
        if data.ndim > 2:
            data = data[0] if data.shape[0] == 1 else np.mean(data, axis=0)

        return data

    except Exception as e:
        print(f"Error loading regridded file {temp_output}: {str(e)}")
        return None

def calculate_ensemble_and_agreement(model_data_list, lats, lons):
    """Calculate ensemble mean and agreement mask.
    
    Args:
        model_data_list: List of 2D numpy arrays (model signals)
        lats: Latitude array
        lons: Longitude array
    
    Returns:
        Tuple of (ensemble_mean, agreement_mask)
        - ensemble_mean: Mean across all models
        - agreement_mask: Boolean array where at least 3 models agree on sign
    """
    if not model_data_list:
        return None, None
    
    # Stack all model data
    stacked_data = np.stack(model_data_list, axis=0)
    
    # Calculate ensemble mean
    ensemble_mean = np.nanmean(stacked_data, axis=0)
    
    # Calculate sign agreement
    # Count how many models have positive values at each grid point
    positive_count = np.sum(stacked_data > 0, axis=0)
    # Count how many models have negative values at each grid point
    negative_count = np.sum(stacked_data < 0, axis=0)
    
    # Agreement where at least 3 models agree on sign (positive OR negative)
    agreement_mask = (positive_count >= 3) | (negative_count >= 3)
    
    return ensemble_mean, agreement_mask

def plot_signal_comparison(output_dir="./"):
    """Create a figure showing land-masked SIGNAL data for tasmax and pr across all models plus ensemble."""

    print("="*70)
    print("Creating Land-Masked Signal Comparison Plot with Ensemble")
    print("="*70)

    # Create temporary directory for CDO operations
    temp_dir = os.path.join(os.getcwd(), "temp_cdo_files")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\nTemporary directory: {temp_dir}")

    try:
        # Create land mask
        lats, lons, land_mask = create_land_mask(MASK_FILE, GRID_FILE, temp_dir)

        if land_mask is None:
            print("Error: Could not create land mask. Aborting.")
            return

        variables = ['tasmax', 'pr']
        models = list(MODEL_INFO.keys())
        num_models = len(models)
        num_vars = len(variables)
        num_rows = num_models + 1  # 4 models + 1 ensemble row

        # Create figure with 2 columns (tasmax, pr) and 5 rows (4 models + ensemble)
        fig_width = 8
        fig_height = 1.5 * num_rows
        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = gridspec.GridSpec(num_rows, num_vars, figure=fig,
                               hspace=0.15,
                               wspace=-0.40,
                               left=0.08, right=0.95,
                               bottom=0.08, top=0.96)

        # Store contour mappables for colorbars (one per variable)
        contour_mappables = [None, None]
        
        # Store model data for ensemble calculation
        model_data_by_var = {var: [] for var in variables}

        # Create subplots for individual models
        for row, model in enumerate(models):
            # Add model name on the left
            y_position = 0.96 - ((row + 0.5) * (0.88 / num_rows))
            fig.text(0.15, y_position, MODEL_INFO[model]['display_name'],
                     va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)

            for col, variable in enumerate(variables):
                # Load and regrid signal data
                file_path = find_signal_file(model, variable)
                data = regrid_and_load_signal(file_path, variable, GRID_FILE, temp_dir, model)

                if data is None:
                    print(f"  Skipping {model} - {variable} (no data)")
                    continue

                # Apply land mask
                data_masked = np.where(land_mask, data, np.nan)
                
                # Store for ensemble calculation
                model_data_by_var[variable].append(data_masked)

                print(f"  Loaded and masked: {model} - {variable}")

                # Get variable settings
                settings = VAR_SETTINGS[variable]

                # Create subplot
                ax = fig.add_subplot(gs[row, col], projection=ccrs.Robinson(central_longitude=0))

                # Add cyclic point for continuous map
                data_cyclic, lons_cyclic = add_cyclic_point(data_masked, coord=lons)

                # Set up normalization
                if variable == 'tasmax':
                    norm = None
                    extend_type = 'max'
                else:
                    norm = mpl.colors.TwoSlopeNorm(vmin=settings['range'][0], vcenter=0, vmax=settings['range'][1])
                    extend_type = 'both'

#                if variable == 'tasmax':
                    # Non-diverging colormap for temperature
#                    norm = mpl.colors.Normalize(vmin=settings['range'][0],
#                                               vmax=settings['range'][1])
#                    extend_type = 'max'
#                else:  # pr
                    # Diverging colormap for precipitation
                    norm = mpl.colors.TwoSlopeNorm(vmin=settings['range'][0],
                                                  vcenter=0,
                                                  vmax=settings['range'][1])
                    extend_type = 'both'

                # Filled contour plot
                cs = ax.contourf(lons_cyclic, lats, data_cyclic,
                                transform=ccrs.PlateCarree(),
                                cmap=settings['cmap'],
                                norm=norm,
                                levels=settings['levels'],
                                extend=extend_type)

                # Add map features
                ax.coastlines(linewidth=0.5, color='gray')
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
                ax.gridlines(draw_labels=False, linewidth=0.3, color='gray',
                            alpha=0.3, linestyle=':')
                ax.set_global()

        # Create ensemble row (last row)
        row = num_models
        y_position = 0.96 - ((row + 0.5) * (0.88 / num_rows))
        fig.text(0.15, y_position, 'ENSEMBLE',
                 va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
        
        for col, variable in enumerate(variables):
            # Calculate ensemble mean and agreement
            ensemble_mean, agreement_mask = calculate_ensemble_and_agreement(
                model_data_by_var[variable], lats, lons
            )
            
            if ensemble_mean is None:
                print(f"  Skipping ENSEMBLE - {variable} (no data)")
                continue
            
            print(f"  Calculated ensemble for {variable}")
            print(f"    Agreement: {np.sum(agreement_mask)} out of {agreement_mask.size} grid points")
            
            # Get variable settings
            settings = VAR_SETTINGS[variable]
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col], projection=ccrs.Robinson(central_longitude=0))
            
            # Add cyclic point for continuous map
            ensemble_cyclic, lons_cyclic = add_cyclic_point(ensemble_mean, coord=lons)
            agreement_cyclic, _ = add_cyclic_point(agreement_mask, coord=lons)
            
            # Set up normalization
            if variable == 'tasmax':
                norm = mpl.colors.Normalize(vmin=settings['range'][0],
                                           vmax=settings['range'][1])
                extend_type = 'max'
            else:  # pr
                norm = mpl.colors.TwoSlopeNorm(vmin=settings['range'][0],
                                              vcenter=0,
                                              vmax=settings['range'][1])
                extend_type = 'both'
            
            # Filled contour plot for ensemble mean
            cs = ax.contourf(lons_cyclic, lats, ensemble_cyclic,
                            transform=ccrs.PlateCarree(),
                            cmap=settings['cmap'],
                            norm=norm,
                            levels=settings['levels'],
                            extend=extend_type)
            
            # Add stippling for agreement (at least 3 models agree on sign)
            lons_mesh, lats_mesh = np.meshgrid(lons, lats)
            # Subsample for stippling (every 3rd point for visibility)
            mask_stipple = agreement_mask[::3, ::3]
            ax.scatter(
                lons_mesh[::3, ::3][mask_stipple], 
                lats_mesh[::3, ::3][mask_stipple],
                s=0.2, c='black', alpha=0.25, marker='.',
                transform=ccrs.PlateCarree()
            )
            
            # Add map features
            ax.coastlines(linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.3, color='gray',
                        alpha=0.3, linestyle=':')
            ax.set_global()
            
            # Store contour mappables for colorbars
            contour_mappables[col] = cs

        # Add colorbars at the bottom
        # Tasmax colorbar
        if contour_mappables[0] is not None:
            cbar_ax1 = fig.add_axes([0.22, 0.02, 0.26, 0.02])
            cbar1 = plt.colorbar(contour_mappables[0], cax=cbar_ax1,
                                orientation='horizontal', extend='max')
            cbar1.ax.tick_params(labelsize=9)
            settings = VAR_SETTINGS['tasmax']
            cbar1.set_label(f'Δ {settings["display_name"]} ({settings["units"]})',
                           fontsize=11, fontweight='bold')

        # Precipitation colorbar
        if contour_mappables[1] is not None:
            cbar_ax2 = fig.add_axes([0.54, 0.02, 0.26, 0.02])
            cbar2 = plt.colorbar(contour_mappables[1], cax=cbar_ax2,
                                orientation='horizontal', extend='both')
            cbar2.ax.tick_params(labelsize=9)
            settings = VAR_SETTINGS['pr']
            cbar2.set_label(f'Δ {settings["display_name"]} ({settings["units"]})',
                           fontsize=11, fontweight='bold')

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        output_path_png = os.path.join(output_dir, "HighResMIP_signal_tasmax_pr_landonly_ensemble.png")
        output_path_pdf = os.path.join(output_dir, "HighResMIP_signal_tasmax_pr_landonly_ensemble.pdf")

        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"\nSaved: {output_path_png}")
        print(f"Saved: {output_path_pdf}")
        print("="*70)

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            print("Temporary files deleted.")

# Main script execution
if __name__ == "__main__":
    output_directory = "."
    plot_signal_comparison(output_dir=output_directory)
    print("\nVisualization complete!")
