"""
Test script to verify the combine_replicates function works correctly.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from Utils.reader import read_csv_file_with_distances

# Import the necessary components from preprocessing_new
from AE.preprocessing_new import combine_replicates, replicate_names, replicate_to_strain

if __name__ == "__main__":
    print("=" * 80)
    print("Testing combine_replicates function")
    print("=" * 80)
    
    # Read the data
    input_folder = "Data/distances_with_zeros"
    print(f"\nReading data from: {input_folder}")
    transposon_data = read_csv_file_with_distances(input_folder)
    
    print(f"\nOriginal datasets loaded: {len(transposon_data)}")
    print("Dataset names:")
    for i, dataset in enumerate(sorted(transposon_data.keys()), 1):
        print(f"  {i:2d}. {dataset}")
    
    # Show which datasets will be combined
    print(f"\n{'=' * 80}")
    print("Replicate groups to be combined:")
    print("=" * 80)
    for replicate_name in replicate_names:
        matching = [d for d in transposon_data if replicate_name in d]
        if matching:
            print(f"\n{replicate_name}: {len(matching)} datasets")
            for dataset in sorted(matching):
                print(f"  - {dataset}")
    
    # Combine replicates and save
    print(f"\n{'=' * 80}")
    print("Combining replicates...")
    print("=" * 80)
    combined_data = combine_replicates(transposon_data, method="average", save=True)
    
    print(f"\n{'=' * 80}")
    print("After combining:")
    print("=" * 80)
    print(f"Total datasets: {len(combined_data)}")
    print("\nDataset names:")
    for i, dataset in enumerate(sorted(combined_data.keys()), 1):
        print(f"  {i:2d}. {dataset}")
    
    # Verify the output structure
    output_folder = "Data/combined_replicates/"
    print(f"\n{'=' * 80}")
    print(f"Verifying output in: {output_folder}")
    print("=" * 80)
    
    if os.path.exists(output_folder):
        strains = sorted(os.listdir(output_folder))
        strains = [s for s in strains if not s.startswith('.')]  # Skip hidden files
        print(f"\nStrain folders: {len(strains)}")
        for strain in strains:
            strain_path = os.path.join(output_folder, strain)
            if os.path.isdir(strain_path):
                datasets = sorted(os.listdir(strain_path))
                datasets = [d for d in datasets if not d.startswith('.')]
                print(f"\n  {strain}:")
                for dataset in datasets:
                    dataset_path = os.path.join(strain_path, dataset)
                    if os.path.isdir(dataset_path):
                        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
                        print(f"    - {dataset} ({len(csv_files)} CSV files)")
    
    print(f"\n{'=' * 80}")
    print("Test completed successfully!")
    print("=" * 80)
