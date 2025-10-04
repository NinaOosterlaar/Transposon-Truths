import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os, sys

from tqdm import tqdm 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from SGD_API.yeast_architecture import Centromeres, Nucleosomes
from reader import read_wig

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

compute_distances("Data/wiggle_format", "Data_exploration/results/distances")
