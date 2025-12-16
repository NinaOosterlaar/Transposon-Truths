import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns  
from tqdm import tqdm
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from Utils.reader import read_wig, label_from_filename


yeast_chrom_lengths = {
    "chrI":     230218,  "chrII":    813184,  "chrIII":   316620, "chrIV":   1531933,
    "chrV":     576874,  "chrVI":    270161,  "chrVII":  1090940, "chrVIII":  562643,
    "chrIX":    439888,  "chrX":     745751,  "chrXI":    666816, "chrXII":  1078177,
    "chrXIII":  924431,  "chrXIV":   784333,  "chrXV":   1091291, "chrXVI":   948066,
    "chrM":      85779,
}
 
 
def show_counts_part_chromosome(file, chrom, start, end):
    """
    Displays counts for a specific chromosome region.
    """
    dict = read_wig(file)
    
    if chrom not in dict:
        print(f"Chromosome {chrom} not found in data.")
        return

    df = dict[chrom]
    region_df = df[(df['Position'] >= start) & (df['Position'] <= end)]

    # Plot bar plot with counts
    plt.figure(figsize=(12, 6))
    plt.bar(region_df['Position'], region_df['Value'], width=1, color='blue')
    plt.title(f'Counts for {chrom}:{start}-{end}')
    plt.xlabel('Position')
    plt.ylabel('Counts')
    plt.xlim(start, end)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
def plot_presence_transposon(folder, chrom, start, end):
    """Plot a heatmap that shows presence or absence of transposon insertions in a given chromosomal region across multiple samples.
    """
    all_counts = []
    for file in os.listdir(folder):
        if not file.endswith('.wig'):
            continue
        file_path = os.path.join(folder, file)
        dict = read_wig(file_path)
        
        if chrom not in dict:
            print(f"Chromosome {chrom} not found in {file}. Skipping.")
            continue
        df = dict[chrom]
        region_df = df[(df['Position'] >= start) & (df['Position'] <= end)]
        # Create a binary presence/absence array for the region
        presence = np.zeros(end - start + 1, dtype=int)
        for _, row in region_df.iterrows():
            pos = int(row['Position']) - start
            if 0 <= pos < len(presence):
                presence[pos] = 1  # Mark presence
        label = label_from_filename(file)
        all_counts.append((label, presence))
        
    if not all_counts:
        print("No valid data found for the specified chromosome and region.")
        return
    all_counts.sort(key=lambda x: x[0])  # Sort by label
    labels = [x[0] for x in all_counts]
    data_matrix = np.array([x[1] for x in all_counts])
    plt.figure(figsize=(12, max(6, len(labels) * 0.3)))
    sns.heatmap(data_matrix, cmap='Greys', cbar=False, yticklabels=labels)
    plt.title(f'Transposon Insertion Presence in {chrom}:{start}-{end}')
    plt.xlabel('Position in Region')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()
    
def plot_basic_statistics(stats, output_folder):
    # --- Total sum per file ---
    file_labels = [s[0] for s in stats]
    total_sums  = [s[1] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.bar(file_labels, total_sums, color='deeppink')
    plt.title('Total Sum of Counts per WIG File')
    plt.xlabel('Sample')
    plt.ylabel('Total Counts')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'total_counts_per_file.png'))
    else:
        plt.show()

    # --- Mean counts per bp ---
    mean_counts = [s[2] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.bar(file_labels, mean_counts, color='hotpink')
    plt.title('Mean Counts per WIG File')
    plt.xlabel('Sample')
    plt.ylabel('Mean Counts')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'mean_counts.png'))
    else:
        plt.show()

    # --- Occupied vs unoccupied sites ---
    occupied = [s[3] for s in stats]
    unoccupied = [s[4] for s in stats]

    plt.figure(figsize=(12, 6))
    plt.bar(file_labels, occupied, label="Occupied sites", color="lightskyblue")
    plt.bar(file_labels, unoccupied, bottom=occupied, label="Unoccupied sites", color="lightpink")
    plt.title('Occupied vs Unoccupied Sites per Sample')
    plt.xlabel('Sample')
    plt.ylabel('Number of Sites')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'occupied_vs_unoccupied_sites.png'))
    else:
        plt.show()
        
        
# def create_histogram(folder_name, output_file):
#     """Create and save a histogram of counts from all WIG files in the specified folder."""
#     all_values = {}

#     for root, dirs, files in os.walk(folder_name):
#         print(f"root: {root}, files: {files}, dirs: {dirs}")
#         number_of_datasets = len(files)
#         for wig_file in files:
#             if not wig_file.endswith(".wig"): continue
            
#             # Read the wig file
#             file_path = os.path.join(root, wig_file)
#             wig_dict = read_wig(file_path) 
#             assert len(wig_dict) == 17, f"Expected 17 chromosomes, found {len(wig_dict)} in {wig_file}"

#             for chrom, df in wig_dict.items():
#                 for count in df['Value']:
#                     if count in all_values:
#                         all_values[count] += 1
#                     else:
#                         all_values[count] = 1
#         # normalize by number of datasets
#         for key in all_values:
#             all_values[key] /= number_of_datasets

    # plt.figure(figsize=(10, 6))
    # plt.hist(all_values.keys(), bins=50, weights=all_values.values(), color='purple', alpha=0.7)
    # plt.title('Histogram of Counts from All WIG Files')
    # plt.xlabel('Counts')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y')
    # plt.tight_layout() 
    # plt.savefig(output_file)
    # plt.close()

# create_histogram("Data/wiggle_format", "Data_exploration/results/histogram_counts.png")
    
def save_basic_info(folder_name, output_file, plot = False, output_folder_figures = None):
    """Saves basic statistics to a text file."""
    stats = []  # (label, total_sum, mean_count, occupied_sites, unoccupied_sites)

    genome_size = sum(yeast_chrom_lengths.values())
    
    for root, dirs, files in os.walk(folder_name):
        print(f"root: {root}, files: {files}, dirs: {dirs}")
        for wig_file in files:
            if not wig_file.endswith(".wig"): continue
            
            # Read the wig file
            file_path = os.path.join(root, wig_file)
            # file_path = os.path.join(folder_name, wig_file)
            wig_dict = read_wig(file_path) 
            assert len(wig_dict) == 17, f"Expected 17 chromosomes, found {len(wig_dict)} in {wig_file}"

            total_sum = 0
            occupied_sites = 0
            non_zero_sum = 0
            non_zero_count = 0
            all_values = []

            for chrom, df in wig_dict.items():
                # Filter out positions with values > 1 million
                high_values = df[df['Value'] > 1000000]
                if not high_values.empty:
                    for _, row in high_values.iterrows():
                        print(f"Filtered out high value in {wig_file}, {chrom}, Position {row['Position']}: {row['Value']}")
                    df = df[df['Value'] <= 1000000]
                
                total_sum += df['Value'].sum()
                if not df.empty:
                    occupied_sites += (df['Value'] > 0).sum()
                    # Calculate sum and count of non-zero values
                    non_zero_values = df[df['Value'] > 0]['Value']
                    non_zero_sum += non_zero_values.sum()
                    non_zero_count += len(non_zero_values)
                    # Collect all values for standard deviation calculation
                    all_values.extend(df['Value'].tolist())

            mean_count = total_sum / genome_size
            mean_non_zero = non_zero_sum / non_zero_count if non_zero_count > 0 else 0
            std_dev = np.std(all_values) if len(all_values) > 0 else 0
            unoccupied_sites = genome_size - occupied_sites
            label = label_from_filename(wig_file)
            density = occupied_sites / (occupied_sites + unoccupied_sites) 

            stats.append((label, total_sum, mean_count, occupied_sites, unoccupied_sites, density, mean_non_zero, std_dev))

    with open(output_file, 'w') as f:
        f.write("Sample\tTotal_Sum\tMean_Coverage_per_bp\tOccupied_Sites\tUnoccupied_Sites\tDensity\tMean_Non_Zero_Count\tStd_Dev\n")
        for s in stats:
            f.write(f"{s[0]}\t{s[1]}\t{s[2]:.6f}\t{s[3]}\t{s[4]}\t{s[5]:.6f}\t{s[6]:.6f}\t{s[7]:.6f}\n")
            
    if plot:
        plot_basic_statistics(stats, output_folder_figures)


def read_csv_combined_replicates(folder_path):
    """
    Reads CSV files from combined_replicates format and returns a dict of DataFrames, one per chromosome.
    Combined replicates have columns: Position, Value, Nucleosome_Distance, Centromere_Distance
    """
    chrom_data = {}
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and file.startswith('Chr'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # Extract chromosome name from filename (e.g., ChrI_distances.csv -> ChrI)
            chrom = file.split('_')[0]
            
            # Keep only Position and Value columns to match the wig format
            if 'Position' in df.columns and 'Value' in df.columns:
                chrom_data[chrom] = df[['Position', 'Value']]
    
    return chrom_data


def save_basic_info_csv(folder_name, output_file, plot=False, output_folder_figures=None):
    """
    Saves basic statistics to a text file for CSV format (combined_replicates).
    This handles CSV files with zeros included, unlike wiggle files which omit zeros.
    """
    stats = []  # (label, total_sum, mean_count, occupied_sites, unoccupied_sites)

    genome_size = sum(yeast_chrom_lengths.values())
    
    for root, dirs, files in os.walk(folder_name):
        # Check if this folder contains CSV files (not subdirectories)
        csv_files = [f for f in files if f.endswith('.csv') and f.startswith('Chr')]
        if not csv_files:
            continue
            
        print(f"Processing: {root}")
        
        # Read all CSV files for this sample
        sample_dict = read_csv_combined_replicates(root)
        
        if len(sample_dict) == 0:
            continue
            
        # Get label from the folder name
        label = label_from_filename(os.path.basename(root))

        total_sum = 0
        occupied_sites = 0
        non_zero_sum = 0
        non_zero_count = 0
        all_values = []

        for chrom, df in sample_dict.items():
            # Skip mitochondrial chromosome if desired
            if chrom == 'ChrM':
                continue
                
            # Filter out positions with values > 1 million
            high_values = df[df['Value'] > 1000000]
            if not high_values.empty:
                for _, row in high_values.iterrows():
                    print(f"Filtered out high value in {label}, {chrom}, Position {row['Position']}: {row['Value']}")
                df = df[df['Value'] <= 1000000]
            
            total_sum += df['Value'].sum()
            occupied_sites += (df['Value'] > 0).sum()
            
            # Calculate sum and count of non-zero values
            non_zero_values = df[df['Value'] > 0]['Value']
            non_zero_sum += non_zero_values.sum()
            non_zero_count += len(non_zero_values)
            
            # Collect all values for standard deviation calculation
            all_values.extend(df['Value'].tolist())

        mean_count = total_sum / genome_size
        mean_non_zero = non_zero_sum / non_zero_count if non_zero_count > 0 else 0
        std_dev = np.std(all_values) if len(all_values) > 0 else 0
        unoccupied_sites = genome_size - occupied_sites
        density = occupied_sites / genome_size if genome_size > 0 else 0

        stats.append((label, total_sum, mean_count, occupied_sites, unoccupied_sites, density, mean_non_zero, std_dev))

    # Sort stats by label for consistent output
    stats.sort(key=lambda x: x[0])

    with open(output_file, 'w') as f:
        f.write("Sample\tTotal_Sum\tMean_Coverage_per_bp\tOccupied_Sites\tUnoccupied_Sites\tDensity\tMean_Non_Zero_Count\tStd_Dev\n")
        for s in stats:
            f.write(f"{s[0]}\t{s[1]}\t{s[2]:.6f}\t{s[3]}\t{s[4]}\t{s[5]:.6f}\t{s[6]:.6f}\t{s[7]:.6f}\n")
            
    if plot:
        plot_basic_statistics(stats, output_folder_figures)
            

# Example usage:
# For wiggle format files (without zeros):
# save_basic_info("Data/old/wiggle_format", "Data_exploration/results/basic_statistics_wiggle.txt", plot=False, output_folder_figures="Data_exploration/figures")

# For combined_replicates CSV files (with zeros):
save_basic_info_csv("Data/combined_strains", "Data_exploration/results/basic_statistics_combined_strains.txt", plot=False, output_folder_figures="Data_exploration/figures")
