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

replicate_names = ["FD7", "FD9", "dnrp1-1", "dnrp1-2"]

# Mapping of replicate names to their strain folders
replicate_to_strain = {
    "FD7": "strain_FD",
    "FD9": "strain_FD",
    "FD11": "strain_FD",
    "FD12": "strain_FD",
    "dnrp1-1": "strain_dnrp",
    "dnrp1-2": "strain_dnrp",
}

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

def get_strain_folder(dataset_name):
    """Determine the strain folder for a dataset."""
    # Check if it's a combined replicate
    if dataset_name in replicate_to_strain:
        return replicate_to_strain[dataset_name]
    
    # Check if it matches a replicate pattern (non-combined)
    for replicate_name in replicate_names:
        if replicate_name in dataset_name:
            if replicate_name in replicate_to_strain:
                return replicate_to_strain[replicate_name]
    
    # Try to infer from dataset name
    if dataset_name.startswith("FD"):
        return "strain_FD"
    elif dataset_name.startswith("dnrp"):
        return "strain_dnrp"
    elif dataset_name.startswith("yEK19"):
        return "strain_yEK19"
    elif dataset_name.startswith("yEK23"):
        return "strain_yEK23"
    elif dataset_name.startswith("yTW001"):
        return "strain_yTW001"
    elif dataset_name.startswith("yWT03"):
        return "strain_yWT03a"
    elif dataset_name.startswith("yWT04"):
        return "strain_yWT04a"
    elif dataset_name.startswith("yLIC") or dataset_name.startswith("ylic"):
        return "strain_ylic137"
    else:
        return "strain_unknown"

def combine_replicates(data, method = "average", save = True, output_folder = "Data/combined_replicates/"):
    """Combine replicate datasets by averaging or summing their data.
    Assumes replicate datasets have names containing replicate identifiers.
    Every dataset point in the dataset is combined using the specified method.
    
    Saves all datasets in the same folder structure as the original data:
    - Combined replicates (e.g., FD7, FD9) go into their strain folders (strain_FD)
    - Non-replicate datasets are copied as-is into their strain folders
    
    Args:
        data (Dictionary): Dictionary containing chromosome DataFrames for each dataset.
        method (str): Method to combine replicates, either "average" or "sum".
        save (bool): Whether to save combined data to CSV files.
        output_folder (str): Base output folder path.
    Returns:
        new_data (Dictionary): Dictionary with combined replicate datasets.
    """
    new_data = {}
    datasets_to_remove = []
    
    # Step 1: Combine replicates
    for replicate_name in replicate_names:
        # Find all datasets that match this replicate name
        matching_datasets = [dataset for dataset in data if replicate_name in dataset]
        
        if not matching_datasets:
            print(f"No datasets found for replicate: {replicate_name}")
            continue
        
        print(f"Combining {len(matching_datasets)} datasets for replicate: {replicate_name}")
        combined_regions = {}
        
        for chrom in chromosome_length.keys():
            # Initialize a dictionary to accumulate values by position
            position_data = {}
            
            # Accumulate data from all matching datasets
            for dataset in matching_datasets:
                if chrom not in data[dataset]:
                    continue
                
                df = data[dataset][chrom]
                
                for _, row in df.iterrows():
                    pos = int(row['Position'])
                    value = row['Value']
                    nuc_dist = row['Nucleosome_Distance']
                    cent_dist = row['Centromere_Distance']
                    
                    if pos not in position_data:
                        position_data[pos] = {
                            'values': [],
                            'nucleosome_distance': nuc_dist,
                            'centromere_distance': cent_dist
                        }
                    position_data[pos]['values'].append(value)
            
            # Compute combined values for this chromosome
            combined_data = []
            for pos in sorted(position_data.keys()):
                values = position_data[pos]['values']
                
                if method == "average":
                    # Only consider non-zero values for averaging
                    non_zero_values = [v for v in values if v != 0]
                    
                    if len(non_zero_values) == 0:
                        # All values are zero
                        combined_value = 0
                    elif len(non_zero_values) == 1:
                        # Only one non-zero value, use it directly
                        combined_value = non_zero_values[0]
                    else:
                        # Two or more non-zero values, take the average
                        combined_value = np.mean(non_zero_values)
                elif method == "sum":
                    combined_value = np.sum(values)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                combined_data.append({
                    'Position': pos,
                    'Value': combined_value,
                    'Nucleosome_Distance': position_data[pos]['nucleosome_distance'],
                    'Centromere_Distance': position_data[pos]['centromere_distance']
                })
            
            # Convert to DataFrame
            if combined_data:
                combined_regions[chrom] = pd.DataFrame(combined_data)
            else:
                combined_regions[chrom] = pd.DataFrame(columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
        
        new_data[replicate_name] = combined_regions
        
        # Mark original replicate datasets for removal
        for dataset in matching_datasets:
            if dataset != replicate_name:
                datasets_to_remove.append(dataset)
    
    # Step 2: Remove original replicate datasets from data
    for dataset in datasets_to_remove:
        del data[dataset]
    
    # Step 3: Add combined data to the data dictionary
    data.update(new_data)
    
    # Step 4: Save all datasets if requested
    if save:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save all datasets
        for dataset in data:
            strain_folder = get_strain_folder(dataset)
            dataset_folder = os.path.join(output_folder, strain_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)
            
            for chrom in data[dataset]:
                output_path = os.path.join(dataset_folder, f"{chrom}_distances.csv")
                df = data[dataset][chrom]
                df.to_csv(output_path, index=False)
            
            print(f"Saved data for {dataset} to {dataset_folder}")
    
    return data

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

def split_data(data, train_val_test_split, split_on):
    """Split data into training, validation, and testing sets.
    
    Args:
        data (DataFrame): DataFrame containing the data to be split.
        train_val_test_split (list): Proportions for training, validation, and testing sets.
        split_on (str): Feature to split data on ('Chrom', 'Dataset', 'Random').
        
    Returns:
        train_data (DataFrame): Training data.
        val_data (DataFrame): Validation data.
        test_data (DataFrame): Testing data.
    """
    train_data = {}
    val_data = {}
    test_data = {}
    train_size = train_val_test_split[0]
    val_size = train_val_test_split[1]
    test_size = train_val_test_split[2]
    if split_on == 'Dataset':
        all_datasets = list(data.keys())
        train_datasets, temp_datasets = train_test_split(all_datasets, train_size=train_size, random_state=42)
        if test_size + val_size > 0:
            val_datasets, test_datasets = train_test_split(temp_datasets, test_size=test_size/(test_size + val_size), random_state=42)
        else:
            val_datasets = []
            test_datasets = []
        for dataset in data:
            if dataset in train_datasets:
                train_data[dataset] = data[dataset]
            elif dataset in val_datasets:
                val_data[dataset] = data[dataset]
            elif dataset in test_datasets:
                test_data[dataset] = data[dataset]
    return train_data, val_data, test_data

def preprocess(input_folder, 
               features = ['Pos', 'Chrom', 'Nucl', 'Centr'], 
               train_val_test_split = [0.7, 0.15, 0.15], 
               split_on = 'Dataset', 
               bin_size = 10, 
               moving_average = True,
               data_point_length = 2000,
               step_size = 200
               ):
    """Preprocessing the data before using at as an input for the Autoencoder

    Args:
        input_folder (Str): The folder with the raw csv files
        features (list, optional): The features to be used. Defaults to ['Pos', 'Chrom', 'Nucl', 'Centr'].
        train_val_test_split (list, optional): The proportions for training, validation, and testing sets. Defaults to [0.7, 0.15, 0.15].
        split_on (str, optional): The feature to split data on ('Chrom', 'Dataset', 'Random'). Defaults to 'Dataset'.
        bin_size (int, optional): The bin size for binning the data of moving average to overcome sparsity. Defaults to 10.
        moving_average (bool, optional): Whether to apply a moving average to the data or use separate bins. Defaults to True.
        data_point_length (int, optional): The length of each data point. Defaults to 2000.
        step_size (int, optional): The step size for sliding window for the data points. Defaults to 200.
    """
    transposon_data = read_csv_file_with_distances(input_folder)
    train, val, test = split_data(transposon_data, train_val_test_split, split_on)
    train = process_data(train, features, bin_size, moving_average, step_size, data_point_length)
    val = process_data(val, features, bin_size, moving_average, step_size, data_point_length)
    test = process_data(test, features, bin_size, moving_average, step_size, data_point_length)
    train, val, test, scalers = standardize_data(train, test, val, features)
    return train, val, test, scalers

def process_data(transposon_data, features, bin_size, moving_average, step_size, data_point_length):
    for dataset in transposon_data:
        for chrom in transposon_data[dataset]:
            df = transposon_data[dataset][chrom]
            if moving_average:
                binned_values = sliding_window(df, bin_size, step_size=1, func=np.mean)
            else:
                binned_values = bin_data(df.values, bin_size)
            transposon_data[dataset][chrom] = pd.DataFrame(binned_values, columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
    # Remove features that are not in the features list
    for dataset in transposon_data:
        for chrom in transposon_data[dataset]:
            df = transposon_data[dataset][chrom]
            cols_to_keep = ['Value']
            if 'Pos' in features:
                cols_to_keep.append('Position')
            if 'Chrom' in features:
                cols_to_keep.append('Chromosome')
            if 'Nucl' in features:
                cols_to_keep.append('Nucleosome_Distance')
            if 'Centr' in features:
                cols_to_keep.append('Centromere_Distance')
            transposon_data[dataset][chrom] = df[cols_to_keep]
    # Apply sliding window to create data points from the data and store them in one big array
    data_points = []
    for dataset in transposon_data:
        for chrom in transposon_data[dataset]:
            df = transposon_data[dataset][chrom]
            windows = sliding_window(df, data_point_length, step_size)
            
            

            

if __name__ == "__main__":
    input_folder = "Data/distances_with_zeros"
    transposon_data = read_csv_file_with_distances(input_folder)
    combine_replicates(transposon_data)