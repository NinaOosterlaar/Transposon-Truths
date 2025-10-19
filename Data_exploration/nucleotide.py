import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from SGD_API.genome import Genome
from reader import read_wig, label_from_filename


def document_sequences(input_file, output_dir="Data_exploration/results/sequences", bin = 5):
    """ Document nucleotide sequences surrounding a transposon position with a given bin size.
    
    Args:
        genome: Genome object to retrieve sequences from
        input_file: Path to input file with transposon positions
        output_dir: Directory to save the sequences 
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label = label_from_filename(input_file)
    genome = Genome()
    chrom_df = read_wig(input_file)
    sequences = {}
    for chrom in chrom_df:
        sequences[chrom] = []
        df = chrom_df[chrom]
        chrom_sequence = genome.get_sequence(chrom)
        for index, row in df.iterrows():
            position = int(row['Position'])
            start = max(0, position - bin)
            end = position + bin
            sequence = chrom_sequence[start:end]
            sequences[chrom].append({
                'position': position,
                'sequence': sequence
            })
    # Save sequences to output files
    file_path = os.path.join(output_dir, f"{label}_sequences.json")
    with open(file_path, 'w') as f:
        json.dump(sequences, f, indent=4)

def process_all_data(input_folder, bin=5, sequences=False, kmers = False):
    """ Process all WIG files in the input folder to document sequences.
    
    Args:
        input_folder: Folder containing WIG files
        bin: Number of nucleotides to include on each side of the position§
    """
    if sequences and kmers:
        raise ValueError("Please choose either sequences or kmers processing, not both.")
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if sequences:
                if file.endswith(".wig"):
                    input_file = os.path.join(root, file)
                    output_dir = os.path.join("Data_exploration/results/sequences", os.path.relpath(root, input_folder))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    document_sequences(input_file, output_dir=output_dir, bin=bin)
            if kmers:
                if file.endswith("_sequences.json"):
                    input_file = os.path.join(root, file)
                    print(f"Processing k-mers for {input_file}")
                    with open(input_file, 'r') as f:
                        sequences = json.load(f)
                    output_dir = os.path.join("Data_exploration/results/kmer_counts", os.path.relpath(root, input_folder))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_file = os.path.join(output_dir, file.replace("_sequences.json", "_kmer_counts.json"))
                    calculate_kmer_occurrences(sequences, output_file, k_sizes=[1,2,3,4,5])
                
def calculate_kmer_occurrences(sequences, output_file, k_sizes=[1, 2, 3, 4, 5]):
    """ Calculate k-mer occurrences in the given sequences.
    
    Args:
        sequences: Dict of sequences per chromosome
        k_sizes: List of k-mer sizes to consider
    """
    kmer_counts = {}
    for chrom, seq_list in sequences.items():
        for item in seq_list:
            sequence = item['sequence']
            for k in k_sizes:
                kmer_counts[k] = {} if k not in kmer_counts else kmer_counts[k]
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i:i + k]
                    if kmer not in kmer_counts[k]:
                        kmer_counts[k][kmer] = 0
                    kmer_counts[k][kmer] += 1
    print(f"Saving k-mer counts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(kmer_counts, f, indent=4)
        

      

if __name__ == "__main__":
    # input_file = "Data/wiggle_format/strain_FD/FD7_1_FDDP210435821-2a_HTWL7DSX2_L4_trimmed_forward_notrimmed.sorted.bam.wig"
    # document_sequences(input_file)
    process_all_data(input_folder="Data/wiggle_format", bin=5, sequences=True)
    # input_file = "Data_exploration/results/sequences/strain_FD/FD7_1_sequences.json"
    # with open(input_file, 'r') as f:
    #     sequences = json.load(f)
    # output_file = "Data_exploration/results/sequences/strain_FD/FD7_1_kmer_counts.json"
    # calculate_kmer_occurrences(sequences, output_file, k_sizes=[1,2,3,4,5])
    process_all_data("Data_exploration/results/sequences", kmers=True)
