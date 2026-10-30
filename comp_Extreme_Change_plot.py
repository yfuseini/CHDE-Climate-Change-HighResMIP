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

# ============================================
# CONFIGURATION - ADJUST THESE AS NEEDED
# ============================================
BASE_PATH = "/work/ch0636/u301861/phd/HighRESMIP/results/compoundXTREMe_3"

# Time periods (same 30-year range for comparison)
HIST_START, HIST_END = 1985, 2014
FUT_START, FUT_END = 2021, 2050

# Models and severities
MODELS = ['CNRM-CERFACS', 'EC-Earth-HR', 'HadGEM3-HM', 'MPI-XR']
SEVERITIES = ['1.0', '1.5', '2.0']

# Display name mapping (for plots only)
DISPLAY_NAMES = {
    'CNRM-CERFACS': 'CNRM',
    'EC-Earth-HR': 'EC-Earth3P',
    'HadGEM3-HM': 'HadGEM3',
    'MPI-XR': 'MPI-XR',
    'OBS': 'Observations',
    'ENSEMBLE': 'ENSEMBLE'
}

# Colormap bounds for difference plot (easily adjustable)
# Based on data range: min=-35, max=+269, suggested=±270
# OPTION 1: Symmetric bounds (uncomment to use)
#COLORMAP_BOUNDS = [-30, -25, -20, -15, -10, -1, 0, 1, 10, 20, 35, 50, 75, 100, 150, 180, 200, 230, 260]
#COLORMAP_BOUNDS = [-30, -20, -10, -1, 0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120] #, 160] #, 180, 200, 230, 260]
COLORMAP_BOUNDS = [-8, -6, -3, -0.5, 0, 5, 8, 11, 14, 17, 20, 22, 25, 28, 31, 33]
# OPTION 2: Asymmetric bounds - better color resolution (uncomment to use)
#COLORMAP_BOUNDS = [-40, -35, -30, -25, -20, -15, -10, -5, -2, 0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 270]

# Output settings
OUTPUT_DPI = 300
SAVE_PNG = True
SAVE_PDF = True
# ============================================


class CompoundDifferencePlotter:
    def __init__(self, base_path):
        """Initialize the difference plotter"""
        self.base_path = Path(base_path)
        self.models = MODELS
        self.severities = SEVERITIES
        self.display_names = DISPLAY_NAMES

        # Severity levels mapping
        self.severity_levels = {
            '1.0': 'TEMP1.0std_SPI1',
            '1.5': 'TEMP1.5std_SPI1.5',
            '2.0': 'TEMP2.0std_SPI2'
        }

        self.var_name = 'CompoundXTREMe'

        # Setup colormap
        self.setup_colormap()

    def setup_colormap(self):
        """Setup diverging colormap for difference plot"""
        colors_diff = [
            '#08519c',  # Dark blue (large decrease)
            '#2171b5',  # Blue
 #           '#4292c6',  # Medium blue
            '#6baed6',  # Light blue
 #           '#9ecae1',  # Very light blue
 #           '#c6dbef',  # Pale blue
#            '#deebf7',  # Very pale blue
#            '#f7fbff',  # Almost white blue
            '#ffffff',  # White (zero change)
            '#ffffff',  # White (zero change)
 #           '#f7fbff',  # Almost white blue
 #           '#fffaf5',  # very pale, nearly-white
 #           '#fff5f0',  # Almost white (near zero)
 #           '#fee5d9',  # Very pale red
 #           '#fdd0c1',  # Pale red (added)
            '#fcbba1',  # Pale red
            '#fc9272',  # Light red
            '#fb6a4a',  # Medium-light red
            '#f84e3d',  # Medium red (added)
            '#ef3b2c',  # Medium red
            '#d52e2e',  # Deep red (added)
            '#cb181d',  # Red
            '#b01818',  # Dark red (added)
            '#a50f15',  # Dark red
            '#67000d',  # Very dark red
        ]

        self.cmap_diff = ListedColormap(colors_diff)
        self.norm_diff = BoundaryNorm(COLORMAP_BOUNDS, self.cmap_diff.N)
        self.bounds_diff = COLORMAP_BOUNDS

    def load_model_data(self, model, severity):
        """Load model compound extreme data"""
        severity_key = self.severity_levels[severity]
        file_path = self.base_path / model / '6months' / f"{model}_compound_{severity_key}_corrected_merged.nc"

        if file_path.exists():
            ds = xr.open_dataset(file_path)
            return ds, self.var_name
        else:
            print(f"Warning: File not found: {file_path}")
            return None, None

    def filter_time_period(self, ds, start_year, end_year):
        """Filter dataset to specific time period"""
        if 'time' in ds.coords:
            time_coord = pd.to_datetime(ds.time.values)
            mask = (time_coord.year >= start_year) & (time_coord.year <= end_year)
            return ds.isel(time=mask)
        return ds

    def calculate_ensemble(self, model_data_list):
        """Calculate ensemble mean from list of model datasets"""
        stacked = xr.concat(model_data_list, dim='model', coords='minimal')
        ensemble_mean = stacked.mean(dim='model')
        return ensemble_mean
    def to_percent(self, da):
        """Share of valid months (%) in which a compound event occurred."""
        n_valid = da.notnull().sum(dim='time')
        events = da.sum(dim='time')
        return events / n_valid.where(n_valid > 0) * 100.0

    def compute_difference_ranges(self):
        """
        Compute and analyze min/max ranges for difference plots
        Returns suggested symmetric bounds
        """
        print("\n" + "="*80)
        print("STEP 1: COMPUTING DIFFERENCE RANGES (Future - Historical)")
        print("="*80)
        print(f"Historical period: {HIST_START}-{HIST_END} ({HIST_END-HIST_START+1} years)")
        print(f"Future period: {FUT_START}-{FUT_END} ({FUT_END-FUT_START+1} years)")
        print("-"*80)

        all_diffs = []
        severity_stats = {}

        for severity in self.severities:
            print(f"\nSeverity {severity}σ:")
            severity_diffs = []

            for model in self.models:
                # Load historical
                ds_hist, var_name = self.load_model_data(model, severity)
                if ds_hist is None:
                    continue

                ds_hist_filtered = self.filter_time_period(ds_hist, HIST_START, HIST_END)
#                freq_hist = ds_hist_filtered[var_name].sum(dim='time')
                freq_hist = self.to_percent(ds_hist_filtered[var_name])

                # Load future
                ds_fut, var_name = self.load_model_data(model, severity)
                if ds_fut is None:
                    continue

                ds_fut_filtered = self.filter_time_period(ds_fut, FUT_START, FUT_END)
#                freq_fut = ds_fut_filtered[var_name].sum(dim='time')
                freq_fut = self.to_percent(ds_fut_filtered[var_name])

                # Calculate difference
                diff = freq_fut - freq_hist
                severity_diffs.append(diff)
                all_diffs.append(diff)

                diff_min = float(diff.min().values)
                diff_max = float(diff.max().values)
                print(f"  {model:15s}: min={diff_min:7.2f}, max={diff_max:7.2f}")

            # Calculate ensemble difference for this severity
            if severity_diffs:
                ensemble_diff = self.calculate_ensemble(severity_diffs)
                ens_min = float(ensemble_diff.min().values)
                ens_max = float(ensemble_diff.max().values)
                print(f"  {'ENSEMBLE':15s}: min={ens_min:7.2f}, max={ens_max:7.2f}")

                severity_stats[severity] = {
                    'min': min([float(d.min().values) for d in severity_diffs]),
                    'max': max([float(d.max().values) for d in severity_diffs]),
                    'ens_min': ens_min,
                    'ens_max': ens_max
                }

        # Overall statistics
        if all_diffs:
            stacked_diffs = xr.concat(all_diffs, dim='model', coords='minimal')
            overall_min = float(stacked_diffs.min().values)
            overall_max = float(stacked_diffs.max().values)
            suggested_bound = max(abs(overall_min), abs(overall_max))

            print("\n" + "-"*80)
            print("OVERALL STATISTICS:")
            print(f"  Minimum difference (most negative): {overall_min:.2f}")
            print(f"  Maximum difference (most positive): {overall_max:.2f}")
            print(f"  Suggested symmetric bound: ±{suggested_bound:.0f}")

            # Provide recommendations
            print("\n" + "-"*80)
            print("COLORMAP RECOMMENDATIONS:")

            # Round to nice numbers
#            nice_bound = np.ceil(suggested_bound / 10) * 10
            nice_bound = np.ceil(suggested_bound / 5) * 5
            print(f"  Recommended symmetric range: ±{nice_bound:.0f}")

            # Check if current bounds are appropriate
            current_max = max(abs(COLORMAP_BOUNDS[0]), abs(COLORMAP_BOUNDS[-1]))
            if suggested_bound > current_max * 0.95:
                print(f"  ⚠️  WARNING: Current bounds (±{current_max}) may be too small!")
                print(f"     Consider updating COLORMAP_BOUNDS to span ±{nice_bound:.0f}")
            elif suggested_bound < current_max * 0.5:
                print(f"  ℹ️  INFO: Current bounds (±{current_max}) are larger than needed")
                print(f"     You could use tighter bounds (±{nice_bound:.0f}) for better contrast")
            else:
                print(f"  ✓ Current bounds (±{current_max}) are appropriate")

            print("="*80 + "\n")

            return overall_min, overall_max, nice_bound, severity_stats

        return None, None, None, {}

    def plot_difference(self, severity_stats=None):
        """
        Create difference plot (fut - hist) for compound extremes
        Shows 3 columns (severities) × 5 rows (4 models + ENSEMBLE)
        """
        print("\n" + "="*80)
        print("STEP 2: CREATING DIFFERENCE PLOT")
        print("="*80)

        n_rows = 5  # 4 models + ENSEMBLE

        # Create figure
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=3,
            figsize=(18, n_rows * 2.5),
            subplot_kw={'projection': ccrs.Robinson()}
        )

        # Collect difference data
        all_diff_data = {}
        row_labels = []

        # Load models and calculate differences
        model_diff_by_severity = {severity: [] for severity in self.severities}

        for model in self.models:
            # Use display name for row label
            row_labels.append(self.display_names.get(model, model))
            all_diff_data[model] = {}

            for severity in self.severities:
                # Historical
                ds_hist, var_name = self.load_model_data(model, severity)
                if ds_hist is None:
                    continue

                ds_hist_filtered = self.filter_time_period(ds_hist, HIST_START, HIST_END)
#                freq_hist = ds_hist_filtered[var_name].sum(dim='time')
                freq_hist = self.to_percent(ds_hist_filtered[var_name])

                # Future
                ds_fut, var_name = self.load_model_data(model, severity)
                if ds_fut is None:
                    continue

                ds_fut_filtered = self.filter_time_period(ds_fut, FUT_START, FUT_END)
#                freq_fut = ds_fut_filtered[var_name].sum(dim='time')
                freq_fut = self.to_percent(ds_fut_filtered[var_name])
                # Calculate difference
                diff = freq_fut - freq_hist
                all_diff_data[model][severity] = diff
                model_diff_by_severity[severity].append(diff)

        # Calculate ensemble differences
        row_labels.append(self.display_names['ENSEMBLE'])
        all_diff_data['ENSEMBLE'] = {}
        agreement_masks = {}

        for severity in self.severities:
            if model_diff_by_severity[severity]:
                ensemble_diff = self.calculate_ensemble(model_diff_by_severity[severity])
                all_diff_data['ENSEMBLE'][severity] = ensemble_diff

                # Agreement: where at least 3 models show positive change
                agreement_masks[severity] = sum(
                    (d > 0).astype(int) for d in model_diff_by_severity[severity]
                ) >= 3

        # Plot data
        plot_keys = self.models + ['ENSEMBLE']

        for row, key in enumerate(plot_keys):
            for col, severity in enumerate(self.severities):
                ax = axes[row, col]

                if severity not in all_diff_data[key]:
                    ax.axis('off')
                    continue

                plot_data = all_diff_data[key][severity]

                # Plot difference
                im = ax.pcolormesh(
                    plot_data.lon, plot_data.lat, plot_data.values,
                    transform=ccrs.PlateCarree(),
                    cmap=self.cmap_diff, norm=self.norm_diff, shading='auto'
                )

                # Add stippling for ensemble agreement (positive change)
                if key == 'ENSEMBLE' and severity in agreement_masks:
                    agreement = agreement_masks[severity]
                    lons, lats = np.meshgrid(plot_data.lon, plot_data.lat)
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

                # Add gridlines
                # Add gridlines (without labels)
                ax.gridlines(linewidth=0.3, alpha=0.5)
#                gl = ax.gridlines(draw_labels=True, dms=True,
#                                x_inline=False, y_inline=False,
#                                linewidth=0.3, alpha=0.5)
#                gl.top_labels = False
#                gl.right_labels = False
#                gl.left_labels = (col == 0)
#                gl.bottom_labels = (row == n_rows - 1)

                # Add column titles (first row only)
                if row == 0:
                    severity_labels = {
                        '1.0': 'Low (1.0σ)',
                        '1.5': 'Moderate (1.5σ)',
                        '2.0': 'High (2.0σ)'
                    }
                    ax.set_title(severity_labels[severity], fontsize=16, fontweight='bold', pad=10)

#                if row == 0:
#                    ax.set_title(f'Severity {severity}σ', fontsize=14, fontweight='bold', pad=10)

                # Add row labels (first column only) - using display names
                if col == 0:
                    ax.text(-0.12, 0.75, row_labels[row],
                           va='center', ha='right', rotation=90,
                           rotation_mode='anchor', transform=ax.transAxes,
                           fontsize=16, fontweight='bold')

        # Add colorbar
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.012])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm_diff, cmap=self.cmap_diff),
            cax=cbar_ax, orientation='horizontal',
            boundaries=self.bounds_diff, ticks=self.bounds_diff, extend='max'
        )
        cb.set_label('Δ compound extreme frequency (% of Months)',
             fontsize=16, fontweight='bold')
#        cb.set_label(
#            f'Δ CHDE (months)',
#            fontsize=16, fontweight='bold'
#        )
        cb.ax.tick_params(labelsize=14)

        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08,
                          wspace=-0.35, hspace=0.05)

        # Save figure
        filename_base = f"compound_difference_{HIST_START}-{HIST_END}_vs_{FUT_START}-{FUT_END}"

        if SAVE_PNG:
            plt.savefig(f"{filename_base}.png", dpi=OUTPUT_DPI, bbox_inches='tight')
            print(f"✓ Saved: {filename_base}.png")

        if SAVE_PDF:
            plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
            print(f"✓ Saved: {filename_base}.pdf")

        plt.close()

        print("="*80 + "\n")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPOUND EXTREMES DIFFERENCE ANALYSIS")
    print("Future Period vs Historical Period")
    print("="*80)
    print(f"Base path: {BASE_PATH}")
    print(f"Historical: {HIST_START}-{HIST_END} ({HIST_END-HIST_START+1} years)")
    print(f"Future: {FUT_START}-{FUT_END} ({FUT_END-FUT_START+1} years)")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Severity levels: {', '.join([s+'σ' for s in SEVERITIES])}")
    print(f"Current colormap bounds: ±{max(abs(COLORMAP_BOUNDS[0]), abs(COLORMAP_BOUNDS[-1]))}")
    print("="*80)

    # Initialize plotter
    plotter = CompoundDifferencePlotter(BASE_PATH)

    # Step 1: Compute ranges and get recommendations
    overall_min, overall_max, suggested_bound, severity_stats = plotter.compute_difference_ranges()

    # Step 2: Create the plot
    if overall_min is not None:
        plotter.plot_difference(severity_stats)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nIf the colormap bounds need adjustment:")
        print("  1. Update COLORMAP_BOUNDS at the top of this script")
        print("  2. Rerun the script to regenerate the plot")
        print("="*80 + "\n")
    else:
        print("\n⚠️  ERROR: Could not compute differences. Check data paths and files.")
