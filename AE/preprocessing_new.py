from copy import deepcopy
import json
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from bin import bin_data, sliding_window
from Utils.reader import read_csv_file_with_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

replicate_names = ["FD7", "FD9", "drnp1-1", "drnp1-2"]

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
}

def combine_replicates(data, method = "average", save = True):
    """Combine replicate datasets by averaging or summing their data.
    Assumes replicate datasets have names containing replicate identifiers.
    Every dataset point in the dataset is combined using the specified method.
    
    Args:
        data (Dictionary): Dictionary containing region dictionaries.
        method (str): Method to combine replicates, either "average" or "sum".
    Returns:
        new_data (Dictionary): Dictionary with combined replicate datasets.
    """
    new_data = {}
    for replicate_name in replicate_names:
        combined_regions = {}
        for chrom in chromosome_length.keys():
            combined_regions[chrom] = []
            for i in range(chromosome_length[chrom]):
                combined_count = 0.0
                count_datasets = 0
                for dataset in data:
                    if replicate_name in dataset:
                        combined_count += data[dataset][chrom]['Value'][i]
                        count_datasets += 1
                position = data[dataset][chrom]['Position'][i] if count_datasets > 0 else i + 1
                Nucleosome_Distance = data[dataset][chrom]['Nucleosome_Distance'][i] if count_datasets > 0 else np.nan
                Centromere_Distance = data[dataset][chrom]['Centromere_Distance'][i] if count_datasets > 0 else np.nan
            if count_datasets > 0:
                if method == "average":
                    combined_count /= count_datasets
                combined_regions[chrom].append([position, combined_count, Nucleosome_Distance, Centromere_Distance])
        new_data[replicate_name] = combined_regions
    if save:
        output_folder = "Data/combined_replicates/"
        os.makedirs(output_folder, exist_ok=True)
        for replicate_name in new_data:
            output_file = os.path.join(output_folder, f"{replicate_name}_combined.csv")
            with open(output_file, 'w') as f:
                f.write("Position,Value,Nucleosome_Distance,Centromere_Distance\n")
                for chrom in new_data[replicate_name]:
                    for pos in range(len(new_data[replicate_name][chrom])):
                        f.write(f"{pos+1},{new_data[replicate_name][chrom][pos]}\n")
    return new_data

def standardize_data(train_data, test_data, val_data, features):
    """Standardize data to have mean 0 and standard deviation 1.
    Only standardizes non-padded values (actual data).
    
    Args:
        train_data, test_data, val_data: Dictionaries containing region dictionaries.
        features: List of features to standardize.
        
    Returns:
        standardized_data and scalers: Training, validation, and testing data with standardized features.
    """
    scalers = {}
    for feature in features:
        if feature == 'Chrom':
            continue  # skip standardization for categorical feature
        scaler = StandardScaler()
        # Collect all values for this feature across all datasets and chromosomes (ONLY non-padded values)
        all_values = []
        for dataset in train_data:
            for chrom in train_data[dataset]:
                for region_dict in train_data[dataset][chrom]:
                    data_sample = region_dict['data']
                    actual_length = region_dict['actual_length']
                    feature_index = features.index(feature) + 1  # +1 because column 0 is counts
                    # Only collect non-padded values
                    all_values.extend(data_sample[:actual_length, feature_index])
        all_values = np.array(all_values).reshape(-1, 1)
        scaler.fit(all_values)
        scalers[feature] = scaler
        # Apply the scaler to training data (only non-padded values)
        for dataset in train_data:
            for chrom in train_data[dataset]:
                for region_dict in train_data[dataset][chrom]:
                    data_sample = region_dict['data']
                    actual_length = region_dict['actual_length']
                    feature_index = features.index(feature) + 1
                    # Only transform non-padded values
                    if actual_length > 0:
                        data_sample[:actual_length, feature_index] = scaler.transform(
                            data_sample[:actual_length, feature_index].reshape(-1, 1)
                        ).flatten()
        # Apply the scaler to validation data (only non-padded values)
        for dataset in val_data:
            for chrom in val_data[dataset]:
                for region_dict in val_data[dataset][chrom]:
                    data_sample = region_dict['data']
                    actual_length = region_dict['actual_length']
                    feature_index = features.index(feature) + 1
                    if actual_length > 0:
                        data_sample[:actual_length, feature_index] = scaler.transform(
                            data_sample[:actual_length, feature_index].reshape(-1, 1)
                        ).flatten()
        # Apply the scaler to testing data (only non-padded values)
        for dataset in test_data:
            for chrom in test_data[dataset]:
                for region_dict in test_data[dataset][chrom]:
                    data_sample = region_dict['data']
                    actual_length = region_dict['actual_length']
                    feature_index = features.index(feature) + 1
                    if actual_length > 0:
                        data_sample[:actual_length, feature_index] = scaler.transform(
                            data_sample[:actual_length, feature_index].reshape(-1, 1)
                        ).flatten()
    return train_data, val_data, test_data, scalers



def preprocess_counts(data):
    """Preprocess transposon insertion counts.
    1. Add pseudocounts
    2. Log-transform counts
    3. Per-dataset normalization
    Only processes non-padded values.
    
    Args:
        data (Dictionary): Dictionary containing region dictionaries.
        
    Returns:
        preprocessed_data (Dictionary): Preprocessed counts data.
        stats (Dictionary): Statistics about the preprocessing.
    """
    norm_data = deepcopy(data)  # so we don't mutate the original
    stats = {}

    for dataset in norm_data:
        # 1. compute total insertions in this dataset (ONLY from non-padded values)
        total_insertions = 0.0
        for chrom in norm_data[dataset]:
            for region_dict in norm_data[dataset][chrom]:
                data_sample = region_dict['data']
                actual_length = region_dict['actual_length']
                # column 0 is raw counts, only sum non-padded values
                total_insertions += np.nansum(data_sample[:actual_length, 0])

        if total_insertions == 0:
            total_insertions = 1.0  # avoid div-by-zero

        # CPM scale factor = counts * (1e6 / total_insertions)
        cpm_scale_factor = 1e6 / total_insertions
        stats[dataset] = {
            "total_insertions": float(total_insertions),
            "cpm_scale_factor": float(cpm_scale_factor),
        }

        # 2. apply CPM + log1p(CPM) to every region for this dataset (only non-padded values)
        for chrom in norm_data[dataset]:
            for region_dict in norm_data[dataset][chrom]:
                data_sample = region_dict['data']
                actual_length = region_dict['actual_length']
                # Only process non-padded values
                raw_counts = data_sample[:actual_length, 0].astype(float)

                # CPM normalization
                cpm = raw_counts * cpm_scale_factor  # counts per million

                # log1p transform
                log_cpm = np.log1p(cpm)  # natural log(1 + CPM)

                # write back into column 0 (only non-padded values)
                data_sample[:actual_length, 0] = log_cpm

    return norm_data, stats

if __name__ == "__main__":
    input_folder = "Data/test"
    transposon_data = read_csv_file_with_distances(input_folder)
    combine_replicates(transposon_data)