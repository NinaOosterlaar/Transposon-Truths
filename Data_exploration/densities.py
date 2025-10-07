import json
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os, sys
from tqdm import tqdm 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from SGD_API.yeast_architecture import Centromeres, Nucleosomes
from reader import read_wig
# import statsmodels.api as sm
# from statsmodels.gam.api import GLMGam, BSplines

chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
    "ChrM": 85779,          # mitochondrial genome (approx for S288C)
    "2micron": 6318         # 2-micron plasmid
}

def compute_distances(input_folder, output_folder):
    """For each signal in the SATAY wig file, compute its distance from the nearest nucleosome and centromere.

    Args:
        input_folder (str): Path to the folder containing SATAY wig files (including subfolders).
        output_folder (str): Path to the folder where the output CSV files will be saved.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    nucleosome_obj = Nucleosomes()
    centromere_obj = Centromeres()
    
    for root, dirs, files in os.walk(input_folder):
        for wig_file in files:
            if not wig_file.endswith(".wig"): continue
            
            print(f"Processing wig file: {wig_file}")
            
            relative_path = os.path.relpath(root, input_folder)
            
            # Create an output folder structure that mirrors the input structure
            if relative_path == ".": wig_output_folder = os.path.join(output_folder, wig_file.replace(".wig", ""))
            else: wig_output_folder = os.path.join(output_folder, relative_path, wig_file.replace(".wig", ""))

            os.makedirs(wig_output_folder, exist_ok=True)

            # Read the wig file
            wig_file_path = os.path.join(root, wig_file)
            print(f"Processing wig file: {wig_file_path}")
            wig_data = read_wig(wig_file_path)

            for i, chrom in tqdm(enumerate(wig_data), total=len(wig_data)):
                df = wig_data[chrom]
                if df.empty:
                    print(f"No data for {chrom} in {wig_file}. Skipping.")
                    continue
                # Initialize lists to store distances
                distances = []

                for _, row in df.iterrows():
                    position = int(row['Position'])
                    value = row['Value']

                    # Compute distance to nearest nucleosome
                    nucleosome_distance = nucleosome_obj.compute_distance(chrom, position)
                    centromere_distance = centromere_obj.compute_distance(chrom, position)
                    distances.append({
                        'Position': position,
                        'Value': value,
                        'Nucleosome_Distance': nucleosome_distance,
                        'Centromere_Distance': centromere_distance
                    })
                    
                # Convert to DataFrame
                distances_df = pd.DataFrame(distances)

                # Save each chromosome to its own file
                output_file = os.path.join(wig_output_folder, f"{chrom}_distances.csv")
                distances_df.to_csv(output_file, index=False)


def process_single_dataset_centromere(strain_name, dataset_path, dataset_name, output_folder, bin=100, max_distance_global=None, min_distance_global=None, boolean=False):
    """Process a single dataset and save results immediately to save memory.
    
    Args:
        strain_name (str): Name of the strain
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        output_folder (str): Base output folder
        bin (int): Size of the sliding window for density calculation
        max_distance_global (int): Maximum distance to consider
        boolean (bool): If True, compute presence/absence density instead of counts
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    print(f"Processing strain: {strain_name}")
    
    # Load only one dataset's data
    dataset_data = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        file_path = os.path.join(dataset_path, csv_file)
        dataset_data[chrom] = pd.read_csv(file_path)
    
    # Process each chromosome in this dataset
    for chrom in dataset_data:
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
            
        df = dataset_data[chrom]

        if max_distance_global is not None:
            max_distance = max_distance_global
        else:
            max_distance = df['Centromere_Distance'].max()
        if min_distance_global is not None:
            min_distance = min_distance_global
        else:
            min_distance = df['Centromere_Distance'].min()
            
        # Create bins aligned around centromere (position 0) to ensure consistent bin centers across datasets
        # Find the range needed to cover all data
        data_range = max(abs(min_distance), abs(max_distance))
        
        # Calculate how many bins we need on each side of 0
        n_bins_each_side = int(np.ceil(data_range / bin)) + 1  # +1 for safety margin
        
        # Create bin centers that are multiples of the bin size: ..., -2*bin, -bin, 0, bin, 2*bin, ...
        bin_centers = np.arange(-n_bins_each_side * bin, (n_bins_each_side + 1) * bin, bin)
        
        # Create bin edges by shifting centers by half a bin size
        # For centers at ..., -bin, 0, bin, ..., edges will be at ..., -1.5*bin, -0.5*bin, 0.5*bin, 1.5*bin, ...
        bin_edges = bin_centers - bin/2
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin)  # Add final edge
        
        df['Distance_Bin'] = pd.cut(df['Centromere_Distance'], bins=bin_edges, right=False, include_lowest=True)

        if boolean:
            # Convert counts to presence/absence
            df['Value'] = df['Value'].apply(lambda x: 1 if x > 0 else 0)
        density = df.groupby('Distance_Bin')['Value'].sum().reset_index()
        density['Bin_Center'] = density['Distance_Bin'].apply(lambda x: x.left + bin / 2)
        
        # Debug: print some bin centers to verify alignment
        if chrom != "ChrM":  # Only print for one chromosome to avoid spam
            print(f"[debug] {strain_name}/{dataset_name} bin centers around 0: {sorted(density['Bin_Center'].values)[:10]}")
        density['Density_per_bp'] = density['Value'] / (bin)
        
        # Filter out outlier values > 1000
        density = density[density['Density_per_bp'] <= 1000]
        
        if density.empty:
            print(f"No valid density data for {chrom} in {strain_name}/{dataset_name} after filtering outliers. Skipping.")
            continue
        
        # Save to CSV immediately
        output_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_bin:{bin}_centromere_density.csv")
        density.to_csv(output_file, index=False)
    
    # Clear dataset from memory
    del dataset_data
    
    # Create plot for this dataset
    create_dataset_plot(strain_name, dataset_name, dataset_output_folder, boolean, bin)


def create_individual_chromosome_plots(strain_name, dataset_name, dataset_output_folder, boolean, bin=100):
    """Create individual clean bar plots for each chromosome."""
    
    # Get all chromosomes (I-XVI) and ensure consistent ordering
    all_chromosomes = [f"Chr{roman}" for roman in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", 
                                                  "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]]
    
    plots_created = 0
    
    for chrom in all_chromosomes:
        density_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_bin:{bin}_centromere_density.csv")
        
        if not os.path.exists(density_file):
            print(f"File not found: {density_file}")
            continue
        
        density = pd.read_csv(density_file)
        
        if density.empty:
            print(f"No valid data for {chrom} after reading CSV")
            continue
        
        # Create individual plot for this chromosome
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extract x and y values from the density data
        x_values = density['Bin_Center']
        y_values = density['Density_per_bp']
        
        # Create bar plot with proper width
        bar_width = bin * 0.8  # Make bars slightly smaller than bin size for cleaner look
        
        bars = ax.bar(x_values, y_values, width=bar_width, 
                     color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
        
        if boolean: 
            ax.set_title(f'Centromere Distance Insertion Rate - {chrom}\n{strain_name}/', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
            ax.set_ylabel('Insertion Rate', fontsize=12)
        else:
            ax.set_title(f'Centromere Distance Density - {chrom}\n{strain_name}/', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
            ax.set_ylabel('Density per bp', fontsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Format x-axis (now handling negative to positive range)
        min_distance = x_values.min()
        max_distance = x_values.max()
        distance_range = max_distance - min_distance
        
        if distance_range > 10000:
            # Show fewer ticks for large ranges
            tick_interval = int(distance_range / 8)  # About 8 ticks
            tick_interval = (tick_interval // 1000) * 1000  # Round to nearest 1000
            if tick_interval == 0:
                tick_interval = 1000
            # Create ticks that span from min to max distance
            tick_start = int(min_distance // tick_interval) * tick_interval
            tick_end = int(max_distance // tick_interval + 1) * tick_interval
            ax.set_xticks(np.arange(tick_start, tick_end + tick_interval, tick_interval))
            ax.tick_params(axis='x', rotation=45)
        
        # Add vertical line at x=0 (centromere position)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Centromere')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Add some statistics to the plot
        mean_density = y_values.mean()
        max_density = y_values.max()
        ax.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_density:.3f}')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Tight layout for clean appearance
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"{chrom}_centromere_density_Boolean_{boolean}_Bin{bin}.png"
        plt_file = os.path.join(dataset_output_folder, plot_filename)
        
        try:
            plt.savefig(plt_file, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Individual plot saved: {plot_filename}")
            plots_created += 1
        except Exception as e:
            print(f"✗ Error saving plot for {chrom}: {e}")
        
        plt.close()  # Important: close figure to free memory


def create_dataset_plot(strain_name, dataset_name, dataset_output_folder, boolean, bin=100):
    """Create plots for a single dataset - now creates individual chromosome plots."""
    create_individual_chromosome_plots(strain_name, dataset_name, dataset_output_folder, boolean, bin)


def density_from_centromere(input_folder, output_folder, bin=1000, max_distance_global=None, min_distance_global=None, boolean=False):
    """Memory-efficient version that processes one dataset at a time.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
        bin (int): Size of the sliding window for density calculation.
        max_distance_global (int): Maximum distance to consider globally.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_centromere(strain_name, dataset_path, dataset_name, output_folder, 
                                bin, max_distance_global, min_distance_global, boolean)
        print(f"✓ Completed: {strain_name}/{dataset_name}")


def density_from_nucleosome(input_folder, output_folder, boolean=False):
    """Compute density from nucleosome distances for all datasets in the input folder.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
        bin (int): Size of the sliding window for density calculation.
        max_distance_global (int): Maximum distance to consider globally.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    nucleosomes = Nucleosomes()
    nucleosomes_normalization = {}
    for chrom in chromosome_length.keys():
        print(f"Computing normalization for {chrom}")
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        normalized_counts = nucleosomes.compute_exposure(chrom)
        nucleosomes_normalization[chrom] = normalized_counts

    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    print(f"Found {len(datasets_to_process)} datasets to process")
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_nucleosome(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization, boolean)
        print(f"✓ Completed: {strain_name}/{dataset_name}")

    
def process_single_dataset_nucleosome(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization, boolean=False):
    """Process a single dataset for nucleosome distances and save results immediately to save memory.
    
    Args:
        strain_name (str): Name of the strain
        dataset_name (str): Name of the dataset
        output_folder (str): Path to the folder where the output CSV files will be saved.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    counts = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        if chrom == "ChrM": continue
        print(f"Processing {chrom} data from {csv_file}")
        file_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(file_path)
        counts[chrom] = {}

        for index, item in df.iterrows():
            # print(item)
            if boolean and item['Value'] > 0: 
                value = 1 
            else: 
                value = item['Value']
                # Search if a value is larger than 1000 and set it to 0
                if value > 1000:
                    value = 0
            nucleosome_distance = item['Nucleosome_Distance']
            if nucleosome_distance in counts[chrom]:
                counts[chrom][nucleosome_distance] += value
            else:
                counts[chrom][nucleosome_distance] = value
        
        
        for distance in counts[chrom]:
            distance = int(distance)
            if distance in nucleosomes_normalization[chrom]:
                counts[chrom][distance] /= nucleosomes_normalization[chrom][distance]
            else:
                del counts[chrom][distance]
        for distance in nucleosomes_normalization[chrom]:
            if distance not in counts[chrom]:
                counts[chrom][distance] = 0

        # Save the processed counts to a CSV file
        output_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:_{boolean}_nucleosome_density.csv")
        with open(output_file, "w") as f:
            f.write("distance,density\n")  # Add header
            for dist, count in counts[chrom].items():
                f.write(f"{dist},{count}\n")
        create_nucleosome_plot(strain_name, dataset_name, dataset_output_folder, chrom, counts[chrom], boolean)

    # Clear counts from memory
    del counts


def create_nucleosome_plot(strain_name, dataset_name, dataset_output_folder, chrom, counts, boolean):
    """Create a plot for nucleosome distance density for a single chromosome.
    It should show each density value as a dot, and show a fitted polynomial line.
    """

    distances = np.array(list(counts.keys()))
    densities = np.array(list(counts.values()))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Scatter plot of the raw data points
    ax.scatter(distances, densities, color='steelblue', alpha=0.6, edgecolor='darkblue', s=20, label='Data Points')
    

    coeffs = np.polyfit(distances, densities, deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(distances.min(), distances.max(), 500)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, color='orange', linewidth=2, label='Fitted Polynomial (deg=3)')
    
    # Add the polynomial equation to the plot (simplified for readability)
    equation_text = f"y = {coeffs[0]:.2e}x³ + {coeffs[1]:.2e}x² + {coeffs[2]:.2e}x + {coeffs[3]:.2e}"
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

    if boolean:
        # Customize the plot
        ax.set_title(f'Nucleosome Distance Insertion Rate- {chrom}\n{strain_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance from Nearest Nucleosome (bp)', fontsize=12)
        ax.set_ylabel('Insertion Rate per bp', fontsize=12)
    else:
        # Customize the plot
        ax.set_title(f'Nucleosome Distance Density - {chrom}\n{strain_name}/{dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance from Nearest Nucleosome (bp)', fontsize=12)
        ax.set_ylabel('Density per bp', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Format x-axis
    max_distance = distances.max()
    if max_distance > 10000:
        # Show fewer ticks for large distances
        tick_interval = int(max_distance / 8)  # About 8 ticks
        tick_interval = (tick_interval // 1000) * 1000  # Round to nearest 1000
        if tick_interval == 0:
            tick_interval = 1000
        ax.set_xticks(np.arange(0, max_distance + tick_interval, tick_interval))
        ax.tick_params(axis='x', rotation=45)

    # Show legend
    ax.legend(loc='upper right', fontsize=12)

    # Save the plot
    output_file = os.path.join(dataset_output_folder, f"{chrom}_nucleosome_density.png")
    
    try:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    except Exception as e:
        print(f"✗ Error saving nucleosome plot: {e}")
    
    plt.close(fig)

# ----------------- loader (unchanged) -----------------
def _load_nuc_density_tables(base_folder: str, boolean: bool) -> pd.DataFrame:
    rows = []
    suffix = f"_Boolean:_{boolean}_nucleosome_density.csv"

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith(suffix):
                continue
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            parts = os.path.normpath(root).split(os.sep)
            strain = parts[-2] if len(parts) >= 2 else "unknown_strain"
            dataset = parts[-1] if len(parts) >= 1 else "unknown_dataset"
            chrom = file.split("_")[0]

            df = pd.read_csv(path).rename(columns={"Distance":"distance","Density":"density"})
            if {"distance","density"} - set(df.columns):
                print(f"[skip] {path} missing distance/density")
                continue

            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df["density"]  = pd.to_numeric(df["density"], errors="coerce")
            df = df.dropna(subset=["distance","density"])
            df["chrom"]   = chrom
            df["strain"]  = strain
            df["dataset"] = dataset
            df["path"]    = path
            rows.append(df[["chrom","strain","dataset","distance","density","path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom","strain","dataset","distance","density","path"])
    return pd.concat(rows, ignore_index=True)

# ----------------- combiner (+ plotting) -----------------
def _combine_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool):
    """
    Writes one combined CSV (and PNG if plot=True) per group.
    CSV columns: distance, mean_density, sd_density, se_density, n_datasets
    """
    if df.empty:
        print("[combine] no data found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"[debug] Nucleosome: Starting combination with {len(df)} total rows")
    print(f"[debug] Nucleosome: Group by keys: {group_by}")
    print(f"[debug] Nucleosome: Unique datasets: {df['dataset'].nunique()}")
    if 'chrom' in df.columns:
        print(f"[debug] Nucleosome: Unique chromosomes: {sorted(df['chrom'].unique())}")

    keys = group_by + ["distance"]
    combined = (df.groupby(keys, as_index=False)
                  .agg(mean_density=("density","mean"),
                       sd_density  =("density","std"),
                       n_datasets  =("density","size")))
    combined["se_density"] = combined["sd_density"] / combined["n_datasets"].clip(lower=1).pow(0.5)
    combined = combined.sort_values(keys)
    # # 1) average within (dataset, distance, + group keys)
    # within_keys = group_by + ["dataset", "distance"]
    # per_ds = df.groupby(within_keys, as_index=False)["density"].mean()
    
    # print(f"[debug] Nucleosome: After step 1: {len(per_ds)} rows from {per_ds['dataset'].nunique()} unique datasets")

    # # 2) average across datasets for each group+distance
    # across_keys = group_by + ["distance"]
    # combined = (per_ds.groupby(across_keys, as_index=False)
    #                   .agg(mean_density=("density","mean"),
    #                        sd_density  =("density","std"),
    #                        n_datasets  =("density","size")))
    # combined["se_density"] = combined["sd_density"] / combined["n_datasets"].clip(lower=1).pow(0.5)
    # combined = combined.sort_values(across_keys)
    
    print(f"[debug] Nucleosome: After step 2: {len(combined)} rows, max n_datasets = {combined['n_datasets'].max()}")

    # 3) write and plot per group
    for keys, sub in combined.groupby(group_by if group_by else [lambda _: True]):
        # build filename label
        if not group_by:
            label = "ALL"
        else:
            if not isinstance(keys, tuple): keys = (keys,)
            label = "_".join(f"{col}-{val}" for col, val in zip(group_by, keys))

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)
        print(f"[write] {out_csv} (max N={sub['n_datasets'].max()})")

        if plot:
            fig, ax = plt.subplots(figsize=(7,4))
            # main line
            ax.plot(sub["distance"], sub["mean_density"], label="Mean density")
            # ribbon: ±1 SE
            lo = sub["mean_density"] - sub["se_density"].fillna(0)
            hi = sub["mean_density"] + sub["se_density"].fillna(0)
            ax.fill_between(sub["distance"], lo, hi, alpha=0.2, label="±1 SE")

            ax.set_xlabel("Distance from nucleosome (bp)")
            ax.set_ylabel("Density")
            ax.set_title(f"Combined nucleosome density — {label}")
            ax.legend(loc="best")
            ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
            ax.minorticks_on()  # enable minor ticks on both axes

            fig.tight_layout()

            out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


def combine_nucleosome_data(data="All", boolean=False, plot=False):
    """
    Combine nucleosome density curves (distance,density) across folders.

    data:
      - "All":         one global curve
      - "Chromosomes": one curve per chromosome
      - "Strains":     one curve per strain
      - "Datasets":    one curve per dataset
    plot:
      - if True, saves a PNG next to each CSV
    """
    base_folder = "Data_exploration/results/densities/nucleosome"
    out_base = os.path.join(base_folder, f"combined_{data}_Boolean_{boolean}")
    os.makedirs(out_base, exist_ok=True)

    df = _load_nuc_density_tables(base_folder, boolean=boolean)
    if df.empty:
        print("[combine] no matching files found.")
        return

    tag = f"Boolean_{boolean}_nucleosome_density"

    if data == "All":
        _combine_curves(df, group_by=[], out_dir=out_base, tag=tag, plot=plot)
    elif data == "Chromosomes":
        _combine_curves(df, group_by=["chrom"], out_dir=out_base, tag=tag, plot=plot)
    elif data == "Strains":
        _combine_curves(df, group_by=["strain"], out_dir=out_base, tag=tag, plot=plot)
    elif data == "Datasets":
        _combine_curves(df, group_by=["dataset"], out_dir=out_base, tag=tag, plot=plot)
    else:
        raise ValueError("data must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


# ---------- 1) Loader: read per-dataset centromere CSVs ----------
def _load_cen_density_tables(base_folder: str, boolean: bool = None, bin_size: int = None) -> pd.DataFrame:
    """
    Scans base_folder for centromere density files and returns one long DF with:
      ['chrom','strain','dataset','bin_size','boolean','Bin_Center','Density_per_bp','path']
    Filters by boolean and bin_size if specified.
    """
    rows = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith("_centromere_density.csv"):
                continue
            
            # Extract boolean and bin size from filename: ChrI_Boolean:True_bin:1000_centromere_density.csv
            file_parts = file.split("_")
            file_boolean = None
            file_bin_size = None
            
            for part in file_parts:
                if part.startswith("Boolean:"):
                    file_boolean = part.split(":")[1] == "True"
                elif part.startswith("bin:"):
                    file_bin_size = int(part.split(":")[1])
            
            # Skip files that don't match the filters
            if boolean is not None and file_boolean != boolean:
                continue
            if bin_size is not None and file_bin_size != bin_size:
                continue
                
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            root_parts = os.path.normpath(root).split(os.sep)
            strain  = root_parts[-2] if len(root_parts) >= 2 else "unknown_strain"
            dataset = root_parts[-1] if len(root_parts) >= 1 else "unknown_dataset"
            chrom   = file_parts[0]  # filename starts with 'chrX_...'

            df = pd.read_csv(path)
            # normalize column names
            df = df.rename(columns={
                "Distance": "Bin_Center",
                "distance": "Bin_Center",
                "Density": "Density_per_bp",
                "density": "Density_per_bp",
            })
            required = {"Bin_Center", "Density_per_bp"}
            if not required.issubset(df.columns):
                print(f"[skip] {path} missing {required - set(df.columns)}")
                continue

            df["Bin_Center"]    = pd.to_numeric(df["Bin_Center"], errors="coerce")
            df["Density_per_bp"] = pd.to_numeric(df["Density_per_bp"], errors="coerce")
            df = df.dropna(subset=["Bin_Center", "Density_per_bp"])

            df["chrom"]   = chrom
            df["strain"]  = strain
            df["dataset"] = dataset
            df["bin_size"] = file_bin_size
            df["boolean"] = file_boolean
            df["path"]    = path
            rows.append(df[["chrom","strain","dataset","bin_size","boolean","Bin_Center","Density_per_bp","path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom","strain","dataset","bin_size","boolean","Bin_Center","Density_per_bp","path"])
    return pd.concat(rows, ignore_index=True)


# ---------- 2) Combiner: unweighted mean ± SE, optional plotting ----------
def _combine_cen_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool):
    """
    Unweighted means across datasets per Bin_Center (no exposure weighting).
    Writes one CSV (and PNG if plot=True) per group.
    CSV columns: Bin_Center, mean_density, sd_density, se_density, n_datasets
    """
    if df.empty:
        print("[centromere] no data found.")
        return
    os.makedirs(out_dir, exist_ok=True)

    if 'chrom' in df.columns:
        print(f"[debug] Unique chromosomes: {sorted(df['chrom'].unique())}")
    print(df.head())
    print(f"The columns are: {df.columns.tolist()}")
        

    # 1) average within (Bin_Center, + group keys), and average it across all the datasets that have that Bin_Center
    # Also save the number of datasets, sd_densituy, se_density
    within_keys = group_by + ["Bin_Center"]
    combined = (df.groupby(within_keys, as_index=False).agg(
                mean_density=("Density_per_bp","mean"),
                sd_density  =("Density_per_bp","std"),
                n_datasets  =("Density_per_bp","size"),
                se_density  =("Density_per_bp","sem"),
            ))
   
    print(f"The columns are: {combined.columns.tolist()}")

    # within_keys = group_by + ["Bin_Center"]
    # per_ds = (df.groupby(within_keys, as_index=False)["Density_per_bp"]
    #             .mean()
    #             .rename(columns={"Density_per_bp": "density"}))
    # print(f"The columns are: {per_ds.columns.tolist()}")
    # # print(f"[debug] After step 1: {len(per_ds)} rows from {per_ds['dataset'].nunique()} unique datasets")

    # # 2) average across datasets for each group + Bin_Center
    # across_keys = group_by + ["Bin_Center"]
    # combined = (per_ds.groupby(across_keys, as_index=False)
    #                   .agg(mean_density=("density","mean"),
    #                        sd_density  =("density","std"),
    #                        n_datasets  =("density","size")))
    # combined["se_density"] = combined["sd_density"] / combined["n_datasets"].clip(lower=1).pow(0.5)
    # combined = combined.sort_values(across_keys)
    
    print(f"[debug] After step 2: {len(combined)} rows, max n_datasets = {combined['n_datasets'].max()}")

    # 3) write + plot per concrete group
    group_iter = [((), combined)] if not group_by else combined.groupby(group_by, dropna=False)

    for keys, sub in group_iter:
        label = "ALL" if not group_by else "_".join(
            f"{col}-{val}" for col, val in zip(group_by, keys if isinstance(keys, tuple) else (keys,))
        )

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}  (max N={sub['n_datasets'].max()})")

        if plot:
            sub_sorted = sub.sort_values("Bin_Center").copy()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(sub_sorted["Bin_Center"], sub_sorted["mean_density"], label="Mean density")
            lo = sub_sorted["mean_density"] - sub_sorted["se_density"].fillna(0)
            hi = sub_sorted["mean_density"] + sub_sorted["se_density"].fillna(0)
            ax.fill_between(sub_sorted["Bin_Center"], lo, hi, alpha=0.2, label="±1 SE")
            ax.axvline(0, linestyle="--", linewidth=1, color="red", alpha=0.7, label="Centromere")
            ax.set_xlabel("Signed distance from centromere (bp)")
            ax.set_ylabel("Density (per bp)")
            ax.set_title(f"Centromere meta-curve — {label}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


# ---------- 3) Public API ----------
def combine_centromere_data(mode="All", boolean=None, bin_size=None, plot=True):
    """
    Combine signed centromere-distance curves (Bin_Center, Density_per_bp) with
    unweighted means across datasets (no exposure weighting), and plot if requested.

    mode:
      - "All":         one global curve
      - "Chromosomes": one curve per chromosome
      - "Strains":     one curve per strain
      - "Datasets":    one curve per dataset
    boolean: Filter by boolean value (True/False) - if None, includes all
    bin_size: Filter by bin size (e.g., 100, 1000) - if None, includes all
    """
    base_folder = "Data_exploration/results/densities/centromere"
    
    # Create descriptive folder name based on filters
    folder_parts = [f"combined_{mode}"]
    if boolean is not None:
        folder_parts.append(f"Boolean_{boolean}")
    if bin_size is not None:
        folder_parts.append(f"bin_{bin_size}")
    
    out_dir = os.path.join(base_folder, "_".join(folder_parts))
    os.makedirs(out_dir, exist_ok=True)

    df = _load_cen_density_tables(base_folder, boolean=boolean, bin_size=bin_size)
    if df.empty:
        print(f"[centromere] no matching files found for boolean={boolean}, bin_size={bin_size}.")
        return

    # Create descriptive tag for output files
    tag_parts = ["centromere_density"]
    if boolean is not None:
        tag_parts.append(f"Boolean_{boolean}")
    if bin_size is not None:
        tag_parts.append(f"bin_{bin_size}")
    tag = "_".join(tag_parts)

    if mode == "All":
        _combine_cen_curves(df, group_by=[], out_dir=out_dir, tag=tag, plot=plot)
    elif mode == "Chromosomes":
        _combine_cen_curves(df, group_by=["chrom"], out_dir=out_dir, tag=tag, plot=plot)
    elif mode == "Strains":
        _combine_cen_curves(df, group_by=["strain"], out_dir=out_dir, tag=tag, plot=plot)
    elif mode == "Datasets":
        _combine_cen_curves(df, group_by=["dataset"], out_dir=out_dir, tag=tag, plot=plot)
    else:
        raise ValueError("mode must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


if __name__ == "__main__":
    # Example usage:
    # Generate centromere densities with specific bin size:
    # density_from_centromere("Data_exploration/results/distances", "Data_exploration/results/densities/centromere", bin=1000, boolean=True)
    
    # # Generate nucleosome densities:
    density_from_nucleosome("Data_exploration/results/distances", "Data_exploration/results/densities/nucleosome", boolean=True)
    
    # Combine nucleosome data:
    combine_nucleosome_data(data="All", boolean=True, plot=True)
    combine_nucleosome_data(data="Chromosomes", boolean=True, plot=True)
    
    # Combine centromere data with specific filters:
    # bin_size = 10000
    # density_from_centromere("Data_exploration/results/distances", "Data_exploration/results/densities/centromere", bin=bin_size, boolean=True)
    # combine_centromere_data(mode="All", boolean=True, bin_size=bin_size, plot=True)
    # combine_centromere_data(mode="Chromosomes", boolean=True, bin_size=bin_size, plot=True)
    