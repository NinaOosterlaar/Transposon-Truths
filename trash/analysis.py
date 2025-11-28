import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

chrom_name_table = {
    "chrI":     "Chromosome_I",  "chrII":    "Chromosome_II",  "chrIII":   "Chromosome_III", "chrIV":   "Chromosome_IV",
    "chrV":     "Chromosome_V",  "chrVI":    "Chromosome_VI",  "chrVII":  "Chromosome_VII", "chrVIII":  "Chromosome_VIII",
    "chrIX":    "Chromosome_IX",  "chrX":     "Chromosome_X",  "chrXI":    "Chromosome_XI", "chrXII":  "Chromosome_XII",
    "chrXIII":  "Chromosome_XIII",  "chrXIV":   "Chromosome_XIV",  "chrXV":   "Chromosome_XV", "chrXVI":   "Chromosome_XVI",
    "chrM":      "Chromosome_M",
}

def analyze_gene_length(gene_file):
    """Analyze gene lengths from a given gene annotation file."""
    with open(gene_file, 'r') as f:
        genes = json.load(f)
    gene_lengths = []
    min_gene = None
    max_gene = None
    min_length = float('inf')
    max_length = float('-inf')
    for gene in genes:
        gene_length = genes[gene]['location']['end'] - genes[gene]['location']['start']
        gene_lengths.append(gene_length)
        if gene_length < min_length:
            min_length = gene_length
            min_gene = gene
        if gene_length > max_length:
            max_length = gene_length
            max_gene = gene
    gene_summary = {
        'total_genes': len(genes),
        'average_length': np.mean(gene_lengths),
        'standard_deviation': np.std(gene_lengths),
        'median_length': np.median(gene_lengths),
        'max_length': np.max(gene_lengths),
        'min_length': np.min(gene_lengths)
    }
    print(f"Shortest gene: {min_gene} with length {min_length}")
    print(f"Longest gene: {max_gene} with length {max_length}")
    print(sorted(gene_lengths))
    return gene_summary, gene_lengths


def histogram_gene_lengths(gene_lengths, bin_size=100):
    """Generate a histogram of gene lengths."""
    plt.figure(figsize=(10, 6))
    plt.hist(gene_lengths, bins=range(0, max(gene_lengths) + bin_size, bin_size), color='blue', alpha=0.7)
    plt.title('Histogram of Gene Lengths')
    plt.xlabel('Gene Length (bp)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def analyze_distance_between_genes(gene_file):
    """Analyze distances between consecutive genes."""
    with open(gene_file, 'r') as f:
        genes = json.load(f)
    all_distances = [] 
    total_overlaps = 0
    for chrom in chrom_name_table.keys():
        chrom_genes = [gene for gene in genes if genes[gene]['location']['chromosome'] == chrom_name_table[chrom]]
        chrom_genes.sort(key=lambda g: genes[g]['location']['start'])
        distances = []
        for i in range(1, len(chrom_genes)):
            prev_end = genes[chrom_genes[i-1]]['location']['end']
            curr_start = genes[chrom_genes[i]]['location']['start']
            distance = curr_start - prev_end
            distances.append(distance)
        overlap_count = sum(1 for d in distances if d < 0)
        print(f"Chromosome {chrom} has {overlap_count} overlapping genes.")
        if distances:
            print(f"Chromosome {chrom}:")
            print(f"  Average distance between genes: {np.mean(distances)}")
            print(f"  Median distance between genes: {np.median(distances)}")
            print(f"  Max distance between genes: {np.max(distances)}")
            print(f"  Min distance between genes: {np.min(distances)}")
        total_overlaps += overlap_count
        all_distances.extend(distances)
    print(f"Total overlapping genes across all chromosomes: {total_overlaps}")
    return all_distances

def plot_distance_histogram(distances, bin_size=100):
    """Plot histogram of distances between genes."""
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=range(min(distances), max(distances) + bin_size, bin_size), color='green', alpha=0.7)
    plt.title('Histogram of Distances Between Genes')
    plt.xlabel('Distance (bp)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gene_file = 'SGD_API/architecture_info/yeast_genes_with_info.json'
    # summary, gene_lengths = analyze_gene_length(gene_file)
    # histogram_gene_lengths(gene_lengths, bin_size=50)
    # print("Gene Length Analysis Summary:")
    # for key, value in summary.items():
    #     print(f"{key}: {value}")  
    distances = analyze_distance_between_genes(gene_file)
    plot_distance_histogram(distances, bin_size=50)
    # print("Distance Between Genes Analysis Summary:")
    # for key, value in summary.items():
    #     print(f"{key}: {value}")