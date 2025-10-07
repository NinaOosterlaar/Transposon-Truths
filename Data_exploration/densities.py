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
        print(f"root: {root}, files: {files}, dirs: {dirs}")
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
        print(chrom)
            
        df = dataset_data[chrom]
        print(df.head())
        
        print(f"max distance: {max_distance_global}, min distance: {min_distance_global}")

        if max_distance_global is not None:
            max_distance = max_distance_global
        else:
            max_distance = df['Centromere_Distance'].max()
        if min_distance_global is not None:
            min_distance = min_distance_global
        else:
            min_distance = df['Centromere_Distance'].min()
            
        print(f"max_distance: {max_distance}, min_distance: {min_distance}")


        bins = np.arange(min_distance - bin, max_distance + bin, bin)
        df['Distance_Bin'] = pd.cut(df['Centromere_Distance'], bins=bins, right=False)

        if boolean:
            # Convert counts to presence/absence
            df['Value'] = df['Value'].apply(lambda x: 1 if x > 0 else 0)
        density = df.groupby('Distance_Bin')['Value'].sum().reset_index()
        density['Bin_Center'] = density['Distance_Bin'].apply(lambda x: x.left + bin / 2)
        density['Density_per_bp'] = density['Value'] / (bin)
        
        # Filter out outlier values > 1000
        density = density[density['Density_per_bp'] <= 1000]
        
        if density.empty:
            print(f"No valid density data for {chrom} in {strain_name}/{dataset_name} after filtering outliers. Skipping.")
            continue
        
        # Save to CSV immediately
        output_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_centromere_density.csv")
        density.to_csv(output_file, index=False)
    
    # Clear dataset from memory
    del dataset_data
    
    # Create plot for this dataset
    create_dataset_plot(strain_name, dataset_name, dataset_output_folder, boolean, bin)


def create_individual_chromosome_plots(strain_name, dataset_name, dataset_output_folder, boolean, bin=100):
    """Create individual clean bar plots for each chromosome."""
    print(f"Creating individual chromosome plots for {strain_name}/{dataset_name}")
    
    # Get all chromosomes (I-XVI) and ensure consistent ordering
    all_chromosomes = [f"Chr{roman}" for roman in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", 
                                                  "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]]
    
    plots_created = 0
    
    for chrom in all_chromosomes:
        density_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_centromere_density.csv")
        
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
        plot_filename = f"{chrom}_centromere_density_Boolean_{boolean}.png"
        plt_file = os.path.join(dataset_output_folder, plot_filename)
        
        try:
            plt.savefig(plt_file, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Individual plot saved: {plot_filename}")
            plots_created += 1
        except Exception as e:
            print(f"✗ Error saving plot for {chrom}: {e}")
        
        plt.close()  # Important: close figure to free memory
    
    print(f"✓ Created {plots_created} individual chromosome plots for {strain_name}/{dataset_name}")


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
    
    print(f"Found {len(datasets_to_process)} datasets to process")
    
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
        normalized_counts = nucleosomes.compute_normalization(chrom)
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
            if distance in nucleosomes_normalization[chrom]:
                counts[chrom][distance] /= nucleosomes_normalization[chrom][distance]
            else:
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


def combine_nucleosome_data(data = "All", boolean=False):
    """Combine nucleosome density data across all strains and datasets for a given chromosome.
    
    Args:
        data (str): Specifies which data to combine. 
            "All" combines all strains and datasets.
            "Chromosomes" combines all the same chromosome across strains and datasets.
            "Strains" combines all datasets within the same strain.
            "Datasets" combines all strains for the same dataset.
        boolean (bool): If True, combine presence/absence density data instead of counts.
    """
    base_folder = "Data_exploration/results/densities/nucleosome"
    combined_output_folder = os.path.join(base_folder, f"combined_{data}_Boolean_{boolean}")
    os.makedirs(combined_output_folder, exist_ok=True)

    # Combine data logic goes here
    if data == "All":
        # Combine all strains and datasets
        pass
    elif data == "Chromosomes":
        # Combine all the same chromosome across strains and datasets
        combined_dataset = {}
        for chrom in chromosome_length.keys():
            for root, dir, files in os.walk(base_folder):
                for file in files:
                    if file.endswith(f"{chrom}_Boolean:{boolean}_nucleosome_density.csv"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            data = pd.read_csv(f)
                            # Combine the data as needed
                            if chrom not in combined_dataset:
                                combined_dataset[chrom] = data
                            else:
                                
                        # Save combined data
                        combined_file = os.path.join(combined_output_folder, f"{chrom}_combined_Boolean_{boolean}_nucleosome_density.csv")
                        # data.to_csv(combined_file, index=False)  # Uncomment and implement actual combining logic
    elif data == "Strains":
        # Combine all datasets within the same strain
        pass
    elif data == "Datasets":
        # Combine all strains for the same dataset
        pass


if __name__ == "__main__":
    # Example usage:
    density_from_nucleosome("Data_exploration/results/distances", "Data_exploration/results/densities/nucleosome", boolean=True)
    # combine_nucleosome_data(data="Chromosomes", boolean=True)