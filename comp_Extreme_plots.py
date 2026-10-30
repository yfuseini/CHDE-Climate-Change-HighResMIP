import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CompoundExtremesPlotter:
    def __init__(self, base_path):
        """
        Initialize the plotter with base directory path
        """
        self.base_path = Path(base_path)
        self.models = ['CNRM-CERFACS', 'EC-Earth-HR', 'HadGEM3-HM', 'MPI-XR']
        self.obs_name = 'OBS'
        
        # Display labels mapping (for plots only)
        self.display_labels = {
            'CNRM-CERFACS': 'CNRM',
            'EC-Earth-HR': 'EC-Earth3P',
            'HadGEM3-HM': 'HadGEM3',
            'MPI-XR': 'MPI-XR',
            'OBS': 'Reference',
            'ENSEMBLE': 'ENSEMBLE'
        }

        # Severity levels mapping
        self.severity_levels = {
            '1.0': {'compound': 'TEMP1.0std_SPI1'},
            '1.5': {'compound': 'TEMP1.5std_SPI1.5'},
            '2.0': {'compound': 'TEMP2.0std_SPI2'}
        }

        # Variable name
        self.var_name = 'CompoundXTREMe'

        # Color scheme
        self.setup_colormaps()

    def setup_colormaps(self):
        """
        Setup discrete colormaps for compound extremes
        """
#        bounds_compound = [0, 1, 5, 8, 11, 15, 18, 21, 25, 30, 35, 40, 45, 50, 55, 60, 70]
        bounds_compound = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25]
        colors_compound = [
            '#ffffff',   # White (0-1)
            '#d1e5f0',   # Very pale blue
            '#3399ff',   # Blue
            '#a6d96a',   # Light green
            '#33cc33',   # Green
            '#ffcc00',   # Yellow
            '#ff9933',   # Orange
            '#ff3333',   # Red
            '#cc0000',   # Dark red
            '#ef6548',   # Red-orange
            '#f781bf',   # Pink-magenta
            '#2d004b',   # Very dark purple
            '#654ea3',   # Deep violet
            '#67001f',   # Very dark red
            '#08306b',   # Deep indigo
            '#fee8c8',   # Pale orange
        ]
        self.cmap_compound = ListedColormap(colors_compound)
        self.norm_compound = BoundaryNorm(bounds_compound, self.cmap_compound.N)
        self.bounds_compound = bounds_compound

    def load_obs_data(self, severity):
        """
        Load observational data for historical period
        """
        severity_key = self.severity_levels[severity]['compound']
        file_path = self.base_path / self.obs_name / '6months' / f"OBS_compound_{severity_key}_hist.nc"

        if file_path.exists():
            ds = xr.open_dataset(file_path)
            print(f"Loaded OBS: {file_path.name}")
            return ds, self.var_name
        else:
            print(f"Warning: OBS file not found: {file_path}")
            return None, None

    def load_model_data(self, model, severity):
        """
        Load model data (merged historical + future)
        """
        severity_key = self.severity_levels[severity]['compound']
        file_path = self.base_path / model / '6months' / f"{model}_compound_{severity_key}_corrected_merged.nc"

        if file_path.exists():
            ds = xr.open_dataset(file_path)
            print(f"Loaded {model}: {file_path.name}")
            return ds, self.var_name
        else:
            print(f"Warning: Model file not found: {file_path}")
            return None, None

    def filter_time_period(self, ds, start_year, end_year):
        """
        Filter dataset to specific time period
        """
        if 'time' in ds.coords:
            time_coord = pd.to_datetime(ds.time.values)
            mask = (time_coord.year >= start_year) & (time_coord.year <= end_year)
            return ds.isel(time=mask)
        return ds

    def calculate_ensemble(self, model_data_list):
        """
        Calculate ensemble mean from list of model datasets
        """
        stacked = xr.concat(model_data_list, dim='model', coords='minimal')
        ensemble_mean = stacked.mean(dim='model')
        return ensemble_mean

    def calculate_model_agreement(self, model_data_list, threshold=3):
        """
        Calculate where at least 'threshold' models agree (have events)
        Returns binary mask
        """
        stacked = xr.concat(model_data_list, dim='model', coords='minimal')
        agreement = (stacked > 0).sum(dim='model')
        return agreement >= threshold

    def to_percent(self, da):
        """Share of valid months (%) in which a compound event occurred."""
        n_valid = da.notnull().sum(dim='time')
        events = da.sum(dim='time')
        return events / n_valid.where(n_valid > 0) * 100.0


    def plot_compound_comparison(self, time_period='hist'):
        """
        Create comparison plot for compound extremes across all severity levels (1.0, 1.5, 2.0)
        Shows 3 columns (one for each severity) with rows for OBS + models + ENSEMBLE
        """
        if time_period == 'hist':
            start_year, end_year = 1985, 2014
            include_obs = True
            n_rows = 6
        else:  # fut
            start_year, end_year = 2021, 2050
            include_obs = False
            n_rows = 5

        severities = ['1.0', '1.5', '2.0']

        # Create figure
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=3,
            figsize=(18, n_rows * 2.5),
            subplot_kw={'projection': ccrs.Robinson()}
        )

        # Collect data for all rows and severities
        all_data = {}
        row_labels = []

        # Load OBS if historical
        if include_obs:
            row_labels.append(self.display_labels['OBS'])
            all_data['OBS'] = {}
            for severity in severities:
                ds, var_name = self.load_obs_data(severity)
                if ds is not None:
                    ds_filtered = self.filter_time_period(ds, start_year, end_year)
                    frequency = self.to_percent(ds_filtered[var_name])
 #                   frequency = ds_filtered[var_name].sum(dim='time')
                    all_data['OBS'][severity] = frequency

        # Load models
        model_data_by_severity = {severity: [] for severity in severities}
        for model in self.models:
            row_labels.append(self.display_labels[model])
            all_data[model] = {}
            for severity in severities:
                ds, var_name = self.load_model_data(model, severity)
                if ds is not None:
                    ds_filtered = self.filter_time_period(ds, start_year, end_year)
 #                   frequency = ds_filtered[var_name].sum(dim='time')
                    frequency = self.to_percent(ds_filtered[var_name])
                    all_data[model][severity] = frequency
                    model_data_by_severity[severity].append(frequency)

        # Calculate ensemble
        row_labels.append(self.display_labels['ENSEMBLE'])
        all_data['ENSEMBLE'] = {}
        agreement_masks = {}
        for severity in severities:
            if model_data_by_severity[severity]:
                ensemble_freq = self.calculate_ensemble(model_data_by_severity[severity])
                all_data['ENSEMBLE'][severity] = ensemble_freq
                agreement_masks[severity] = self.calculate_model_agreement(model_data_by_severity[severity])

        # Plot data
        plot_keys = ['OBS'] + self.models + ['ENSEMBLE'] if include_obs else self.models + ['ENSEMBLE']

        for row, key in enumerate(plot_keys):
            for col, severity in enumerate(severities):
                ax = axes[row, col]

                if severity not in all_data[key]:
                    ax.axis('off')
                    continue

                plot_data = all_data[key][severity]

                # Use compound colormap
                cmap, norm, bounds = self.cmap_compound, self.norm_compound, self.bounds_compound

                # Plot frequency
                im = ax.pcolormesh(
                    plot_data.lon, plot_data.lat, plot_data.values,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, norm=norm, shading='auto'
                )

                # Add stippling for ensemble agreement
                if key == 'ENSEMBLE' and severity in agreement_masks:
                    agreement = agreement_masks[severity]
                    lons, lats = np.meshgrid(agreement.lon, agreement.lat)
                    mask = agreement.values[::3, ::3]
                    ax.scatter(
                        lons[::3, ::3][mask], lats[::3, ::3][mask],
                        s=0.2, c='black', alpha=0.25, marker='.',
                        transform=ccrs.PlateCarree()
                    )

                # Add map features
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_global()

                # Add gridlines (without labels)
                ax.gridlines(linewidth=0.3, alpha=0.5)
                # Add gridlines
#                gl = ax.gridlines(draw_labels=True, dms=True,
#                                x_inline=False, y_inline=False,
#                                linewidth=0.3, alpha=0.5)
#                gl.top_labels = False
#                gl.right_labels = False
#                gl.left_labels = (col == 0)
#                gl.bottom_labels = (row == n_rows - 1)
                
#                if col == 0:
#                    gl.ylabel_style = {'rotation': 0}
##                    gl.xlabels_top = False
#                    gl.xlabels_bottom = (row == n_rows - 1)
#                    gl.ylabels_left = True
#                    gl.ylabels_right = False  # Explicitly disable right side y-labels

                # Add column titles (first row only)
                if row == 0:
                    severity_labels = {
                        '1.0': 'Low (1.0σ)',
                        '1.5': 'Moderate (1.5σ)',
                        '2.0': 'High (2.0σ)'
                    }
                    ax.set_title(severity_labels[severity], fontsize=14, fontweight='bold', pad=10)

           #     if row == 0:
           #         ax.set_title(f'Severity {severity}σ', fontsize=16, fontweight='bold', pad=10)

                # Add row labels (first column only) - using display labels
                if col == 0:
                    ax.text(-0.12, 0.72, row_labels[row],
                           va='center', ha='right', rotation=90,
                           rotation_mode='anchor', transform=ax.transAxes,
                           fontsize=16, fontweight='bold')

        # Add colorbar
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.012])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm_compound, cmap=self.cmap_compound),
            cax=cbar_ax, orientation='horizontal',
            boundaries=self.bounds_compound,
            ticks=[0, 1, 2, 4, 6, 8, 12, 16, 20, 25]   # thinned so labels don't overlap
        )
        cb.set_label('Compound extreme frequency (% of months)', fontsize=16, fontweight='bold')
#        cb = fig.colorbar(
#            plt.cm.ScalarMappable(norm=self.norm_compound, cmap=self.cmap_compound),
#            cax=cbar_ax, orientation='horizontal',
#            boundaries=self.bounds_compound, ticks=self.bounds_compound
#        )
#        cb.set_label('Compound Extreme Frequency (months)', fontsize=16, fontweight='bold')
        cb.ax.tick_params(labelsize=14)

        # Adjust layout
        plt.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08,
                          wspace=-0.35, hspace=0.05)

        # Save figure
        period_str = f"hist_{start_year}-{end_year}" if time_period == 'hist' else f"fut_{start_year}-{end_year}"
        filename_base = f"compound_comparison_all_severities_{period_str}"
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"Saved: {filename_base}.png and .pdf")
        plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    base_path = "/work/ch0636/u301861/phd/HighRESMIP/results/compoundXTREMe_3"

    print("="*70)
    print("COMPOUND EXTREMES COMPARISON ANALYSIS")
    print("="*70)
    print(f"Base path: {base_path}")
    print(f"Time periods: Historical (1985-2014) | Future (2021-2050)")
    print(f"Models: CNRM, EC-Earth3P, HadGEM3, MPI-XR + ENSEMBLE")
    print(f"Observations: Historical only")
    print(f"Severity levels: 1.0σ, 1.5σ, 2.0σ")
    print("="*70)

    plotter = CompoundExtremesPlotter(base_path)
    
    print(f"\n{'='*60}")
    print(f"Creating Compound Extremes Comparison Plots")
    print(f"{'='*60}\n")

    print("Historical period (1985-2014)...")
    plotter.plot_compound_comparison(time_period='hist')

    print("\nFuture period (2021-2050)...")
    plotter.plot_compound_comparison(time_period='fut')

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
