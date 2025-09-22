import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns  # not used, but keeping since you had it

chromosome_mapper = {
    "NC_001133": "chrI",
    "NC_001134": "chrII",
    "NC_001135": "chrIII",
    "NC_001136": "chrIV",
    "NC_001137": "chrV",
    "NC_001138": "chrVI",
    "NC_001139": "chrVII",
    "NC_001140": "chrVIII",
    "NC_001141": "chrIX",
    "NC_001142": "chrX",
    "NC_001143": "chrXI",
    "NC_001144": "chrXII",
    "NC_001145": "chrXIII",
    "NC_001146": "chrXIV",
    "NC_001147": "chrXV",
    "NC_001148": "chrXVI",
    "NC_001224": "chrM",
}

yeast_chrom_lengths = {
    "chrI":     230218,  "chrII":    813184,  "chrIII":   316620, "chrIV":   1531933,
    "chrV":     576874,  "chrVI":    270161,  "chrVII":  1090940, "chrVIII":  562643,
    "chrIX":    439888,  "chrX":     745751,  "chrXI":    666816, "chrXII":  1078177,
    "chrXIII":  924431,  "chrXIV":   784333,  "chrXV":   1091291, "chrXVI":   948066,
    "chrM":      85779,
}

def read_wig(file_path):
    """
    Reads a WIG file and returns a dict of DataFrames, one per chromosome.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    chrom_data = {}
    current_chrom = None
    current_data = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('track'):
                continue  # skip metadata lines

            if line.startswith(('VariableStep', 'variableStep', 'fixedStep')):
                # Save previous chromosome before moving on
                if current_chrom and current_data:
                    chrom_data[current_chrom] = pd.DataFrame(
                        current_data, columns=['Position', 'Value']
                    )
                    current_data = []

                # Detect chromosome ID in line
                for ncbi_id, mapped_chrom in chromosome_mapper.items():
                    if ncbi_id in line or mapped_chrom in line:
                        current_chrom = mapped_chrom
                        break
                continue  # skip definition line

            # Parse data lines
            parts = line.strip().split()
            if len(parts) == 2 and current_chrom:
                position, value = parts
                current_data.append((int(position), float(value)))

    # Save last chromosome
    if current_chrom and current_data:
        chrom_data[current_chrom] = pd.DataFrame(
            current_data, columns=['Position', 'Value']
        )

    return chrom_data


def label_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]

    # cut before merged/FDDP tokens
    cut_positions = [p for p in (
        stem.find('_merged'), stem.find('_FDDP'),
        stem.find('-merged'), stem.find('-FDDP')
    ) if p != -1]
    head = stem[:min(cut_positions)] if cut_positions else stem

    # FD{strain}_{rep}
    m = re.search(r'(?i)\bFD(\d+)[-_](\d+)\b', head)
    if m:
        return f"FD{m.group(1)}_{m.group(2)}"

    # yLIC{strain}_{rep}
    m = re.search(r'(?i)\byLIC(\d+)[-_](\d+)\b', head)
    if m:
        return f"yLIC{m.group(1)}_{m.group(2)}"

    # dnrp{strain}-{repnum} ... {a|b}
    m_strain = re.search(r'(?i)\bdnrp(\d+)\b', head)
    if m_strain:
        strain = m_strain.group(1)
        # numeric replicate before _merged
        m_repnum = re.search(r'[._-](\d+)$', head)
        repnum = m_repnum.group(1) if m_repnum else None
        # a/b letter anywhere in full stem
        m_letter = re.search(r'(?i)(?:^|[-_])(a|b)(?:$|[-_])', stem)
        letter = m_letter.group(1).lower() if m_letter else None

        if repnum and letter:
            return f"dnrp{strain}-{repnum}-{letter}"  # UNIQUE
        if letter:
            return f"dnrp{strain}-{letter}"
        if repnum:
            return f"dnrp{strain}_{repnum}"
        return f"dnrp{strain}"

    # fallback: keep head
    return head

def plot_basic_statistics(folder):
    wig_files = [f for f in os.listdir(folder) if f.endswith('.wig')]
    wig_files.sort()

    stats = []  # (label, total_sum, mean_count, occupied_sites, unoccupied_sites)

    genome_size = sum(yeast_chrom_lengths.values())

    for wig_file in wig_files:
        file_path = os.path.join(folder, wig_file)
        wig_dict = read_wig(file_path)  # {chrom: df}

        total_sum = 0
        occupied_sites = 0

        for chrom, df in wig_dict.items():
            total_sum += df['Value'].sum()
            if not df.empty:
                occupied_sites += (df['Value'] > 0).sum()

        mean_count = total_sum / genome_size
        unoccupied_sites = genome_size - occupied_sites
        label = label_from_filename(wig_file)

        stats.append((label, total_sum, mean_count, occupied_sites, unoccupied_sites))

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
    plt.show()

    # --- Mean coverage per bp ---
    mean_counts = [s[2] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.bar(file_labels, mean_counts, color='hotpink')
    plt.title('Mean Coverage per Base Pair per WIG File')
    plt.xlabel('Sample')
    plt.ylabel('Mean Coverage per bp')
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
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
    plt.show()
 
 
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
    

        

if __name__ == "__main__":
# Example of showing counts for a specific region
    # show_counts_part_chromosome("Data/E-MTAB-14476/FD7_1_FDDP210435821-2a_HTWL7DSX2_L4_trimmed_forward_notrimmed.sorted.bam.wig", "chrI", 10000, 20000)  # Change as needed
    plot_presence_transposon("Data/E-MTAB-14476", "chrI", 10000, 20000)  # Change as needed