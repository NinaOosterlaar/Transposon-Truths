import json
import numpy as np
import pandas as pd
import os, sys
import gc
import argparse
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

def standardize_data(train_data, val_data, test_data, features):
    """Standardize features to have mean 0 and standard deviation 1.
    Fits scalers on training data and applies to all splits.
    Does NOT standardize 'Value' (counts - already log-normalized) or 'Chrom' (categorical).
    
    Args:
        train_data, val_data, test_data: Dictionaries containing {dataset: {chromosome: DataFrame}}.
        features: List of features to use (e.g., ['Pos', 'Chrom', 'Nucl', 'Centr']).
        
    Returns:
        train_data, val_data, test_data: Data with standardized features (same objects, modified in-place).
        scalers: Dictionary of StandardScaler objects for each feature.
    """
    scalers = {}
    
    # Features to standardize (exclude 'Value' and 'Chrom')
    features_to_standardize = []
    feature_to_column = {
        'Pos': 'Position',
        'Nucl': 'Nucleosome_Distance',
        'Centr': 'Centromere_Distance'
    }
    
    for feature in features:
        if feature in feature_to_column:
            features_to_standardize.append(feature)
    
    # For each feature, fit scaler on training data
    for feature in features_to_standardize:
        column_name = feature_to_column[feature]
        scaler = StandardScaler()
        
        # Collect all values for this feature from training data
        all_values = []
        for dataset in train_data:
            for chrom in train_data[dataset]:
                df = train_data[dataset][chrom]
                if column_name in df.columns:
                    all_values.extend(df[column_name].values)
        
        if len(all_values) > 0:
            all_values = np.array(all_values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(all_values)
            scalers[feature] = scaler
            
            # Clean up
            del all_values
            gc.collect()
            
            # Apply scaler to training data (in-place)
            for dataset in train_data:
                for chrom in train_data[dataset]:
                    df = train_data[dataset][chrom]
                    if column_name in df.columns:
                        train_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
            
            # Apply scaler to validation data (in-place)
            for dataset in val_data:
                for chrom in val_data[dataset]:
                    df = val_data[dataset][chrom]
                    if column_name in df.columns:
                        val_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
            
            # Apply scaler to test data (in-place)
            for dataset in test_data:
                for chrom in test_data[dataset]:
                    df = test_data[dataset][chrom]
                    if column_name in df.columns:
                        test_data[dataset][chrom][column_name] = scaler.transform(
                            df[column_name].values.reshape(-1, 1)
                        ).flatten().astype(np.float32)
    
    return train_data, val_data, test_data, scalers

def preprocess_counts(data, zinb_mode=False):
    """Preprocess transposon insertion counts.
    1. CPM (Counts Per Million) normalization per dataset
    2. Log-transform counts: log1p(CPM)
    3. For ZINB mode: save raw counts in 'Value_Raw' and add size factor column
    
    Args:
        data (Dictionary): Dictionary containing {dataset: {chromosome: DataFrame}} structure.
        zinb_mode (bool): If True, save raw counts and add size factor column. Default=False.
        
    Returns:
        data (Dictionary): Preprocessed counts data (same object, modified in-place).
        stats (Dictionary): Statistics about the preprocessing (total insertions and CPM scale factor per dataset).
    """
    stats = {}

    for dataset in data:
        # 1. Compute total insertions in this dataset across all chromosomes
        total_insertions = 0.0
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            total_insertions += df['Value'].sum()

        if total_insertions == 0:
            total_insertions = 1.0  # avoid div-by-zero

        # CPM scale factor = counts * (1e6 / total_insertions)
        cpm_scale_factor = 1e6 / total_insertions
        # Size factor for ZINB = total_insertions / 1e6 (library size normalization)
        size_factor = total_insertions / 1e6
        
        stats[dataset] = {
            "total_insertions": float(total_insertions),
            "cpm_scale_factor": float(cpm_scale_factor),
            "size_factor": float(size_factor),
        }

        # 2. Apply CPM + log1p(CPM) to every chromosome in this dataset (in-place)
        for chrom in data[dataset]:
            df = data[dataset][chrom]
            
            if zinb_mode:
                # Save raw counts before normalization
                data[dataset][chrom]['Value_Raw'] = df['Value'].copy()
                # Add size factor as constant column (same value for all rows in this dataset)
                data[dataset][chrom]['Size_Factor'] = size_factor
            
            # CPM normalization
            cpm = df['Value'] * cpm_scale_factor  # counts per million

            # log1p transform
            log_cpm = np.log1p(cpm).astype(np.float32)  # Use float32 to save memory

            # Write back into the DataFrame (in-place)
            data[dataset][chrom]['Value'] = log_cpm
            
            # Clean up temporary variables
            del cpm, log_cpm

    return data, stats

def _split_items(items, train_size, val_size, test_size):
    """Helper function to split a list of items into train/val/test."""
    if not items or (val_size == 0 and test_size == 0):
        return items, [], []
    if len(items) == 1 or train_size >= 1.0:
        return items, [], []
    if train_size <= 0.0:
        return [], items, []
    
    # Split train from (val + test)
    train_items, temp_items = train_test_split(items, train_size=train_size, random_state=42)
    
    # Split val from test
    if not temp_items or test_size == 0:
        return train_items, temp_items, []
    if val_size == 0:
        return train_items, [], temp_items
    if len(temp_items) == 1:
        return train_items, temp_items, []
    
    val_items, test_items = train_test_split(temp_items, test_size=test_size/(test_size + val_size), random_state=42)
    return train_items, val_items, test_items

def split_data(data, train_val_test_split, split_on, chunk_size=50000):
    """Split data into training, validation, and testing sets.
    
    Args:
        data (DataFrame): DataFrame containing the data to be split.
        train_val_test_split (list): Proportions for training, validation, and testing sets.
        split_on (str): Feature to split data on ('Chrom', 'Dataset', 'Random').
        chunk_size (int): Size of chunks in base pairs for random splitting. Default: 50000.
        
    Returns:
        train_data (DataFrame): Training data.
        val_data (DataFrame): Validation data.
        test_data (DataFrame): Testing data.
    """
    train_size, val_size, test_size = train_val_test_split
    
    if split_on == 'Dataset':
        train_datasets, val_datasets, test_datasets = _split_items(list(data.keys()), train_size, val_size, test_size)
        
        train_data = {d: data[d] for d in train_datasets}
        val_data = {d: data[d] for d in val_datasets}
        test_data = {d: data[d] for d in test_datasets}
        
        print(f"Train datasets: {train_datasets}")
        print(f"Validation datasets: {val_datasets}")
        print(f"Test datasets: {test_datasets}")
    
    elif split_on == 'Chrom':
        # Get all unique chromosomes
        all_chroms = list(set(chrom for dataset in data.values() for chrom in dataset.keys()))
        train_chroms, val_chroms, test_chroms = _split_items(all_chroms, train_size, val_size, test_size)
        
        # Assign chromosomes to splits for each dataset
        train_data = {d: {c: data[d][c] for c in data[d] if c in train_chroms} for d in data}
        val_data = {d: {c: data[d][c] for c in data[d] if c in val_chroms} for d in data}
        test_data = {d: {c: data[d][c] for c in data[d] if c in test_chroms} for d in data}
        
        # Remove empty dataset dictionaries
        train_data = {d: v for d, v in train_data.items() if v}
        val_data = {d: v for d, v in val_data.items() if v}
        test_data = {d: v for d, v in test_data.items() if v}
    
    elif split_on == 'Random':
        # Create chunks from all chromosomes across all datasets
        all_chunks = []
        
        for dataset in data:
            for chrom in data[dataset]:
                df = data[dataset][chrom]
                if df.empty:
                    continue
                
                # Get min and max positions for this chromosome
                min_pos = df['Position'].min()
                max_pos = df['Position'].max()
                
                # Create chunks
                current_pos = min_pos
                while current_pos <= max_pos:
                    chunk_end = min(current_pos + chunk_size, max_pos + 1)
                    all_chunks.append({
                        'dataset': dataset,
                        'chrom': chrom,
                        'start': current_pos,
                        'end': chunk_end
                    })
                    current_pos = chunk_end
        
        # Split chunks into train/val/test
        train_chunks, val_chunks, test_chunks = _split_items(all_chunks, train_size, val_size, test_size)
        
        # Initialize data structures
        train_data = {d: {} for d in data.keys()}
        val_data = {d: {} for d in data.keys()}
        test_data = {d: {} for d in data.keys()}
        
        # Assign data points to splits based on chunk assignments
        def assign_chunks_to_split(chunks, split_data):
            for chunk in chunks:
                dataset = chunk['dataset']
                chrom = chunk['chrom']
                start = chunk['start']
                end = chunk['end']
                
                # Filter DataFrame for this chunk
                df = data[dataset][chrom]
                mask = (df['Position'] >= start) & (df['Position'] < end)
                chunk_df = df[mask].copy()
                
                if not chunk_df.empty:
                    # Add to existing chromosome data or create new
                    if chrom in split_data[dataset]:
                        split_data[dataset][chrom] = pd.concat([split_data[dataset][chrom], chunk_df], ignore_index=True)
                    else:
                        split_data[dataset][chrom] = chunk_df
        
        assign_chunks_to_split(train_chunks, train_data)
        assign_chunks_to_split(val_chunks, val_data)
        assign_chunks_to_split(test_chunks, test_data)
        
        # Remove empty datasets/chromosomes and sort by position
        for split_data in [train_data, val_data, test_data]:
            datasets_to_remove = []
            for dataset in split_data:
                chroms_to_remove = []
                for chrom in split_data[dataset]:
                    if split_data[dataset][chrom].empty:
                        chroms_to_remove.append(chrom)
                    else:
                        # Sort by position
                        split_data[dataset][chrom] = split_data[dataset][chrom].sort_values('Position').reset_index(drop=True)
                
                for chrom in chroms_to_remove:
                    del split_data[dataset][chrom]
                
                if not split_data[dataset]:
                    datasets_to_remove.append(dataset)
            
            for dataset in datasets_to_remove:
                del split_data[dataset]
    
    else:
        train_data, val_data, test_data = {}, {}, {}
    
    return train_data, val_data, test_data

def _find_consecutive_segments(df, gap_tolerance=1):
    """Find consecutive segments in a DataFrame based on Position column.
    
    Args:
        df (DataFrame): DataFrame with a 'Position' column.
        gap_tolerance (int): Maximum gap between consecutive positions. Default is 1.
        
    Returns:
        List of DataFrames, each containing a consecutive segment.
    """
    if df.empty:
        return []
    
    # Sort by position
    df = df.sort_values('Position').reset_index(drop=True)
    
    segments = []
    current_segment = [0]  # Start with first row
    
    for i in range(1, len(df)):
        prev_pos = df.iloc[i-1]['Position']
        curr_pos = df.iloc[i]['Position']
        
        # Check if positions are consecutive (within tolerance)
        if curr_pos - prev_pos <= gap_tolerance:
            current_segment.append(i)
        else:
            # Gap detected, save current segment and start new one
            segments.append(df.iloc[current_segment].copy())
            current_segment = [i]
    
    # Add the last segment
    if current_segment:
        segments.append(df.iloc[current_segment].copy())
    
    return segments

def _add_chromosome_encoding(df, chrom):
    """Add chromosome encoding to DataFrame.
    
    Args:
        df (DataFrame): DataFrame to add chromosome column to.
        chrom (str): Chromosome name (e.g., 'ChrI', 'ChrII', ..., 'ChrXVI').
        
    Returns:
        DataFrame with chromosome encoding added as categorical integers (1-16).
    """
    # Map chromosome names to integers (1-16)
    chrom_to_int = {
        'ChrI': 1, 'ChrII': 2, 'ChrIII': 3, 'ChrIV': 4,
        'ChrV': 5, 'ChrVI': 6, 'ChrVII': 7, 'ChrVIII': 8,
        'ChrIX': 9, 'ChrX': 10, 'ChrXI': 11, 'ChrXII': 12,
        'ChrXIII': 13, 'ChrXIV': 14, 'ChrXV': 15, 'ChrXVI': 16
    }
    
    chrom_int = chrom_to_int.get(chrom, 0)
    df['Chromosome'] = chrom_int
    
    return df

def process_data(transposon_data, features, bin_size, moving_average, step_size, data_point_length, split_on='Dataset', zinb_mode=False):
    """Process data: bin/window and convert to 3D array for autoencoder input.
    
    Args:
        zinb_mode (bool): If True, include Value_Raw and Size_Factor columns in output. Default=False.
    
    Returns:
        np.ndarray: 3D array of shape (num_samples, window_length, num_features)
    """
    # Check if chromosome encoding is needed
    use_chrom = 'Chrom' in features
    
    # Only check for non-consecutive segments when using Random split
    if split_on == 'Random':
        # Split each chromosome into consecutive segments to handle gaps from random splitting
        segmented_data = {}
        for dataset in transposon_data:
            segmented_data[dataset] = {}
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom].copy()
                segments = _find_consecutive_segments(df, gap_tolerance=1)
                segmented_data[dataset][chrom] = segments
        
        # Apply binning/windowing to each consecutive segment
        processed_segments = {}
        for dataset in segmented_data:
            processed_segments[dataset] = {}
            for chrom in segmented_data[dataset]:
                processed_segments[dataset][chrom] = []
                for segment_df in segmented_data[dataset][chrom]:
                    if segment_df.empty:
                        continue
                    
                    # Apply binning or moving average
                    if moving_average:
                        binned_values = sliding_window(segment_df.values, bin_size, step_size=1, moving_average=True)
                    else:
                        binned_values = bin_data(segment_df.values, bin_size)
                    
                    # Convert back to DataFrame
                    if len(binned_values) > 0:
                        # Use the original dataframe's columns to handle both normal and ZINB modes
                        binned_df = pd.DataFrame(binned_values, columns=segment_df.columns.tolist())
                        
                        # Add chromosome encoding if needed
                        if use_chrom:
                            binned_df = _add_chromosome_encoding(binned_df, chrom)
                        
                        processed_segments[dataset][chrom].append(binned_df)
        
        # Select and order columns based on features
        cols_to_keep = ['Value']
        if 'Pos' in features:
            cols_to_keep.append('Position')
        if 'Nucl' in features:
            cols_to_keep.append('Nucleosome_Distance')
        if 'Centr' in features:
            cols_to_keep.append('Centromere_Distance')
        if use_chrom:
            # Add chromosome column (categorical encoding)
            cols_to_keep.append('Chromosome')
        
        # Filter columns for each segment
        for dataset in processed_segments:
            for chrom in processed_segments[dataset]:
                for i, segment_df in enumerate(processed_segments[dataset][chrom]):
                    existing_cols = [col for col in cols_to_keep if col in segment_df.columns]
                    processed_segments[dataset][chrom][i] = segment_df[existing_cols]
        
        # Apply sliding window to create data points from consecutive segments only
        data_points = []
        for dataset in processed_segments:
            for chrom in processed_segments[dataset]:
                for segment_df in processed_segments[dataset][chrom]:
                    if segment_df.empty or len(segment_df) < data_point_length:
                        continue
                    
                    data_array = segment_df.values.astype(np.float32)
                    windows = sliding_window(data_array, data_point_length, step_size)
                    data_points.extend(windows)
                    
                    # Clean up
                    del segment_df, data_array, windows
                
                # Clean up processed segments as we go
                processed_segments[dataset][chrom] = None
            
            # Clean up dataset
            processed_segments[dataset] = None
        
        # Clean up
        del processed_segments
        gc.collect()
    
    else:
        # For Dataset and Chrom splits, data is already consecutive - use simpler processing
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom].copy()
                if moving_average:
                    binned_values = sliding_window(df.values, bin_size, step_size=1, moving_average=True)
                else:
                    binned_values = bin_data(df.values, bin_size)
                
                # Use the original dataframe's columns to handle both normal and ZINB modes
                binned_df = pd.DataFrame(binned_values, columns=df.columns.tolist())
                
                # Add chromosome encoding if needed
                if use_chrom:
                    binned_df = _add_chromosome_encoding(binned_df, chrom)
                
                transposon_data[dataset][chrom] = binned_df
                
                # Clean up
                del df, binned_values
        
        # Select and order columns based on features
        cols_to_keep = ['Value']
        if 'Pos' in features:
            cols_to_keep.append('Position')
        if 'Nucl' in features:
            cols_to_keep.append('Nucleosome_Distance')
        if 'Centr' in features:
            cols_to_keep.append('Centromere_Distance')
        if use_chrom:
            # Add chromosome column (categorical encoding)
            cols_to_keep.append('Chromosome')
        
        # For ZINB mode, add raw counts and size factor columns at the end
        if zinb_mode:
            cols_to_keep.append('Value_Raw')
            cols_to_keep.append('Size_Factor')
        
        # Filter columns
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom]
                existing_cols = [col for col in cols_to_keep if col in df.columns]
                transposon_data[dataset][chrom] = df[existing_cols]
        
        # Apply sliding window to create data points
        data_points = []
        for dataset in transposon_data:
            for chrom in transposon_data[dataset]:
                df = transposon_data[dataset][chrom]
                if df.empty or len(df) < data_point_length:
                    continue
                data_array = df.values.astype(np.float32)
                windows = sliding_window(data_array, data_point_length, step_size, moving_average=False)
                data_points.extend(windows)
                
                # Clean up
                del df, data_array, windows
            
            # Clean up dataset data as we process
            transposon_data[dataset] = None
        
        # Clean up
        del transposon_data
        gc.collect()
    
    # Convert list of arrays to 3D numpy array: (num_samples, window_length, num_features)
    if len(data_points) > 0:
        data_points = np.array(data_points, dtype=np.float32)
    else:
        data_points = np.array([], dtype=np.float32).reshape(0, data_point_length, len(cols_to_keep))
    
    return data_points
            
def preprocess(input_folder, 
               features = ['Pos', 'Chrom', 'Nucl', 'Centr'], 
               train_val_test_split = [0.7, 0.15, 0.15], 
               split_on = 'Dataset',
               chunk_size = 50000,
               normalize_counts = True,
               zinb_mode = False,
               bin_size = 10, 
               moving_average = True,
               data_point_length = 2000,
               step_size = 500
               ):
    """Preprocessing the data before using at as an input for the Autoencoder

    Args:
        input_folder (Str): The folder with the raw csv files
        features (list, optional): The features to be used. Defaults to ['Pos', 'Chrom', 'Nucl', 'Centr'].
            - 'Value': Transposon insertion counts (always included)
            - 'Pos': Position along chromosome
            - 'Chrom': Chromosome (categorical encoding: ChrI=1, ChrII=2, ..., ChrXVI=16)
            - 'Nucl': Distance to nearest nucleosome
            - 'Centr': Distance to centromere
        train_val_test_split (list, optional): The proportions for training, validation, and testing sets. Defaults to [0.7, 0.15, 0.15].
        split_on (str, optional): The feature to split data on ('Chrom', 'Dataset', 'Random'). Defaults to 'Dataset'.
        chunk_size (int, optional): Size of chunks in base pairs for random splitting. Defaults to 50000.
        normalize_counts (bool, optional): Whether to apply CPM normalization and log transform to counts. Defaults to True.
        zinb_mode (bool, optional): If True, save raw counts in 'Value_Raw' column for ZINB models. Defaults to False.
        bin_size (int, optional): The bin size for binning the data of moving average to overcome sparsity. Defaults to 10.
        moving_average (bool, optional): Whether to apply a moving average to the data or use separate bins. Defaults to True.
        data_point_length (int, optional): The length of each data point. Defaults to 2000.
        step_size (int, optional): The step size for sliding window for the data points. Defaults to 200.
    
    Returns:
        train (np.ndarray): Training data, shape (n_train_samples, window_length, n_features)
        val (np.ndarray): Validation data, shape (n_val_samples, window_length, n_features)
        test (np.ndarray): Test data, shape (n_test_samples, window_length, n_features)
        scalers (dict): Dictionary of StandardScaler objects for each feature.
        count_stats (dict, optional): Statistics from count normalization if normalize_counts=True.
    """
    transposon_data = read_csv_file_with_distances(input_folder)
    
    # Optionally normalize counts before splitting
    count_stats = None
    if normalize_counts:
        transposon_data, count_stats = preprocess_counts(transposon_data, zinb_mode=zinb_mode)
    
    # Split data
    train, val, test = split_data(transposon_data, train_val_test_split, split_on, chunk_size)
    
    # Standardize features (fit on train, transform on val/test)
    train, val, test, scalers = standardize_data(train, val, test, features)
    
    # Bin/window and convert to 3D arrays
    train = process_data(train, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    gc.collect()  # Clean up memory after processing train
    
    val = process_data(val, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    gc.collect()  # Clean up memory after processing val
    
    test = process_data(test, features, bin_size, moving_average, step_size, data_point_length, split_on, zinb_mode=zinb_mode)
    gc.collect()  # Clean up memory after processing test

    return train, val, test, scalers, count_stats

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess transposon insertion data for autoencoder training')
    
    # Input/Output
    parser.add_argument('--input_folder', type=str, default='Data/combined_replicates/',
                        help='Folder containing the raw CSV files (default: Data/combined_replicates/)')
    parser.add_argument('--output_dir', type=str, default='Data/processed_data/',
                        help='Directory to save processed data (default: Data/processed_data/)')
    
    # Features
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['Pos', 'Chrom', 'Nucl', 'Centr'],
                        choices=['Pos', 'Chrom', 'Nucl', 'Centr'],
                        help='Features to use (default: Pos Chrom Nucl Centr)')
    
    # Data splitting
    parser.add_argument('--train_val_test_split', type=float, nargs=3, 
                        default=[0.7, 0.15, 0.15],
                        help='Train, validation, and test split proportions (default: 0.7 0.15 0.15)')
    parser.add_argument('--split_on', type=str, default='Dataset',
                        choices=['Chrom', 'Dataset', 'Random'],
                        help='Feature to split data on (default: Dataset)')
    parser.add_argument('--chunk_size', type=int, default=50000,
                        help='Size of chunks in base pairs for random splitting (default: 50000)')
    
    # Normalization
    parser.add_argument('--normalize_counts', action='store_true', default=True,
                        help='Apply CPM normalization and log transform to counts (default: True)')
    parser.add_argument('--no_normalize_counts', action='store_false', dest='normalize_counts',
                        help='Disable count normalization')
    parser.add_argument('--zinb_mode', action='store_true', default=False,
                        help='ZINB mode: save raw counts in Value_Raw column for ZINB models (default: False)')
    
    # Binning/Windowing
    parser.add_argument('--bin_size', type=int, default=10,
                        help='Bin size for binning the data (default: 10)')
    parser.add_argument('--moving_average', action='store_true', default=True,
                        help='Apply moving average to the data (default: True)')
    parser.add_argument('--no_moving_average', action='store_false', dest='moving_average',
                        help='Use separate bins instead of moving average')
    parser.add_argument('--data_point_length', type=int, default=2000,
                        help='Length of each data point (default: 2000)')
    parser.add_argument('--step_size', type=int, default=500,
                        help='Step size for sliding window (default: 500)')
    
    return parser.parse_args()

            

if __name__ == "__main__":
    args = parse_args()
    
    # Run preprocessing with parsed arguments
    train, val, test, scalers, count_stats = preprocess(
        input_folder=args.input_folder,
        features=args.features,
        train_val_test_split=args.train_val_test_split,
        split_on=args.split_on,
        chunk_size=args.chunk_size,
        normalize_counts=args.normalize_counts,
        zinb_mode=args.zinb_mode,
        bin_size=args.bin_size,
        moving_average=args.moving_average,
        data_point_length=args.data_point_length,
        step_size=args.step_size
    )
    
    # Print some info
    print(f"\nProcessing complete!")
    print(f"Train data shape: {train.shape}")
    print(f"Validation data shape: {val.shape}")
    print(f"Test data shape: {test.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    zinb_suffix = "_ZINB" if args.zinb_mode else ""
    output_name = args.output_dir + f"Features{args.features}_SplitOn{args.split_on}_BinSize{args.bin_size}_DataPointLen{args.data_point_length}_StepSize{args.step_size}_Normalize{args.normalize_counts}_MovingAvg{args.moving_average}{zinb_suffix}_"
    
    # Save the train, validation, and test data as .npy files
    train_file = output_name + f"train_data.npy"
    val_file = output_name + f"val_data.npy"
    test_file = output_name + f"test_data.npy"
    
    np.save(train_file, train)
    np.save(val_file, val)
    np.save(test_file, test)
    
    print(f"\nData saved to:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  Test: {test_file}")
