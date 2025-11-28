from copy import deepcopy
import json
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from bin import bin_data
from Utils.reader import read_csv_file_with_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
chromosome_translator = {
    "Chromosome_I": "ChrI",
    "Chromosome_II": "ChrII",
    "Chromosome_III": "ChrIII",
    "Chromosome_IV": "ChrIV",
    "Chromosome_V": "ChrV",
    "Chromosome_VI": "ChrVI",
    "Chromosome_VII": "ChrVII",
    "Chromosome_VIII": "ChrVIII",
    "Chromosome_IX": "ChrIX",
    "Chromosome_X": "ChrX",
    "Chromosome_XI": "ChrXI",
    "Chromosome_XII": "ChrXII",
    "Chromosome_XIII": "ChrXIII",
    "Chromosome_XIV": "ChrXIV",
    "Chromosome_XV": "ChrXV",
    "Chromosome_XVI": "ChrXVI",
}

def split_genes(genes_list, distance_around_genes):
    """Split genes into data points and add distance around genes.
    Removes genes that are completely encompassed by other genes.
    
    Args:
        genes_list (list): List of genes with their positions.
        distance_around_genes (int): Distance around genes to consider.
        
    Returns:
        split_genes_list (Dictionary): Dictionary with chromosomes as keys and lists of gene regions as values.
    """
    split_genes_list = {}
    
    # First pass: collect all gene regions with distance around them
    gene_regions = {}  # chromosome -> list of (start, end, gene_name)
    
    for gene in genes_list:
        chrom = genes_list[gene]['location']['chromosome']
        if chrom not in chromosome_translator:
            continue  # skip if chromosome not recognized
        chrom = chromosome_translator.get(chrom, chrom)
        
        if chrom not in gene_regions:
            gene_regions[chrom] = []
        
        start = max(0, genes_list[gene]['location']['start'] - distance_around_genes)
        end = min(chromosome_length[chrom], genes_list[gene]['location']['end'] + distance_around_genes)
        gene_regions[chrom].append((start, end))
    
    # Second pass: remove encompassed genes (smaller regions completely inside larger ones)
    for chrom in gene_regions:
        regions = gene_regions[chrom]
        filtered_regions = []
        
        for i, (start_i, end_i) in enumerate(regions):
            is_encompassed = False
            
            # Check if this region is completely inside any other region
            for j, (start_j, end_j) in enumerate(regions):
                if i != j:
                    # Check if region i is completely inside region j
                    if start_j <= start_i and end_i <= end_j:
                        # Region i is encompassed by region j
                        is_encompassed = True
                        break
            
            if not is_encompassed:
                filtered_regions.append([start_i, end_i])
        
        split_genes_list[chrom] = filtered_regions
    
    return split_genes_list

def fill_gaps(regions_list, minimum_region_size):
    """Fill gaps between regions while ensuring minimal region size.
    If gaps are smaller than minimum_region_size, they are merged with adjacent regions.
    If gaps are larger, they are kept as separate regions.
    
    Args:
        regions_list (list): List of regions with their positions.
        minimum_region_size (int): Minimum size of regions to become its own data point.
        
    Returns:
        filled_regions_list (list): List of regions with gaps filled.
    """
    for chrom in regions_list:
        regions = sorted(regions_list[chrom], key=lambda x: x[0])
        filled_regions = []
        
        if len(regions) == 0:
            continue
        
        # Handle gap before first region
        first_region = regions[0]
        if first_region[0] >= minimum_region_size:
            # Gap is large enough, add it as a separate region
            filled_regions.append([0, first_region[0]])
            filled_regions.append([first_region[0], first_region[1]])
        else:
            # Gap is too small, extend first region to start at 0
            filled_regions.append([0, first_region[1]])
        
        # Process gaps between consecutive regions
        for i in range(1, len(regions)):
            prev_region = filled_regions[-1]  # Get the last added region (modifiable by reference)
            current_region = regions[i]
            
            gap_start = prev_region[1]
            gap_end = current_region[0]
            gap_size = gap_end - gap_start
            
            # Skip if regions are overlapping or touching (no gap or negative gap)
            if gap_size <= 0:
                # Just add the current region as-is (overlap will be handled by resolve_overlaps)
                filled_regions.append([current_region[0], current_region[1]])
                continue
            
            if gap_size >= minimum_region_size:
                # Gap is large enough to be its own region
                filled_regions.append([gap_start, gap_end])
                filled_regions.append([current_region[0], current_region[1]])
            else:
                # Gap is too small, split it between adjacent regions
                middle = (gap_start + gap_end) // 2
                prev_region[1] = middle  # Extend previous region (modifies in place!)
                filled_regions.append([middle, current_region[1]])
        
        # Handle gap after last region
        last_region = filled_regions[-1]
        gap_after = chromosome_length[chrom] - last_region[1]
        if gap_after >= minimum_region_size:
            filled_regions.append([last_region[1], chromosome_length[chrom]])
        else:
            # Extend last region to chromosome end (modifies in place!)
            last_region[1] = chromosome_length[chrom]
        
        # Update regions_list with filled regions (convert to tuples for consistency)
        regions_list[chrom] = [tuple(region) for region in filled_regions]
    
    return regions_list

def resolve_overlaps(regions_list):
    """Resolve overlapping regions by adjusting their boundaries.
    
    Args:
        regions_list (list): List of regions with their positions.
    
    Returns:
        resolved_regions_list (dict): List of regions with overlaps resolved.
    """
    resolved_regions = {}
    for chrom in regions_list:
        regions = regions_list[chrom]
        resolved_regions[chrom] = []
        
        if len(regions) == 0:
            continue
        
        for i in range(len(regions) - 1):
            current_region = list(regions[i])  # Convert to list for mutability
            next_region = list(regions[i + 1])
            
            if current_region[1] > next_region[0]:  # Overlap detected
                overlap_start = next_region[0]
                overlap_end = current_region[1]

                mid_point = (overlap_start + overlap_end) // 2
                current_region[1] = mid_point
                next_region[0] = mid_point
                
                # Validate BOTH regions before updating the list
                if next_region[0] >= next_region[1]:
                    # Next region would be invalid, don't update it
                    print(f"Warning: Overlap resolution would create invalid next region [{next_region[0]}, {next_region[1]}] on {chrom}. Keeping original next region.")
                else:
                    # Update the next_region in the regions list for subsequent iterations
                    regions[i + 1] = next_region
                    
            # Validate region before adding (skip if invalid)
            if current_region[0] < current_region[1]:
                resolved_regions[chrom].append(current_region)
            else:
                print(f"Warning: Skipping invalid region [{current_region[0]}, {current_region[1]}] on {chrom} after overlap resolution")
        
        # Add the last region with validation
        last_region = list(regions[-1])
        if last_region[0] < last_region[1]:
            resolved_regions[chrom].append(tuple(last_region))
        else:
            print(f"Warning: Skipping invalid last region [{last_region[0]}, {last_region[1]}] on {chrom}")
    
    return resolved_regions

def divide_long_sequences(regions_list, maximum_region_size):
    """Divide long sequences into smaller regions based on maximum_region_size.
    
    Args:
        regions_list (list): List of regions with their positions.
        maximum_region_size (int): Maximum size of regions.
        
    Returns:
        divided_regions_list (list): List of regions divided into smaller segments.
    """
    final_regions = {}
    for chrom in regions_list:
        regions = regions_list[chrom]
        final_regions[chrom] = []
        for region in regions:
            start, end = region
            region_size = end - start
            if region_size > maximum_region_size:
                num_subregions = (region_size + maximum_region_size - 1) // maximum_region_size
                subregion_size = region_size // num_subregions
                for i in range(num_subregions):
                    sub_start = start + i * subregion_size
                    if i == num_subregions - 1:
                        sub_end = end
                    else:
                        sub_end = sub_start + subregion_size
                    final_regions[chrom].append((sub_start, sub_end))
            else:
                final_regions[chrom].append(region)
    return final_regions

def add_transposon_and_features(transposon_data, regions_list, features, maximum_region_size=3000):
    """Add transposon insertion data and selected features to each region.

    Args:
        transposon_data (Dictionary): Dictionary of DataFrames containing transposon insertion data and additional information.
        regions_list (list): List of regions with their positions.
        features (list): List of features to include in the data, can include the position, chromosome, distance from nucleosome and distance from centromere.
        maximum_region_size (int): Maximum size for padding arrays.

    Returns:
        enriched_regions_list (Dictionary): Dictionary with datasets as key and region dictionaries as values.
            Each region is a dict: {'data': array, 'actual_length': int, 'start': int, 'end': int}
    """
    regions = {}
    for dataset in transposon_data:
        data = transposon_data[dataset]
        regions[dataset] = {}
        for chrom in regions_list:
            # Skip if chromosome not in data
            if chrom not in data:
                print(f"Warning: Chromosome {chrom} not found in dataset {dataset}. Skipping.")
                continue
            
            data_chrom = data[chrom]
            regions[dataset][chrom] = []
            for region in regions_list[chrom]:
                data_sample = np.zeros((maximum_region_size, len(features) + 1))  # +1 for transposon insertion counts
                start, end = region
                
                # Validate region
                if start >= end:
                    print(f"Warning: Invalid region {start}-{end} on {chrom} in dataset {dataset} (start >= end). Skipping.")
                    continue
                
                region_data = data_chrom[(data_chrom['Position'] >= start) & (data_chrom['Position'] < end)]
                actual_length = len(region_data)  # Track the actual data length
                
                # Skip regions with no data points (this shouldn't happen if your data includes all positions)
                if actual_length == 0:
                    print(f"Warning: Region {start}-{end} on {chrom} in dataset {dataset} has no data points. "
                          f"This may indicate missing data in your dataset. Skipping.")
                    continue
                
                # Fill the data array
                data_sample[:actual_length, 0] = region_data['Value'].values  # Transposon insertion counts
                for feature in features:
                    if feature == 'Pos':
                        data_sample[:actual_length, features.index('Pos') + 1] = region_data['Position'].values 
                    elif feature == 'Nucl':
                        nucl_index = region_data.columns.get_loc('Nucleosome_Distance')
                        data_sample[:actual_length, features.index('Nucl') + 1] = region_data.iloc[:, nucl_index].values
                    elif feature == 'Centr':
                        centr_index = region_data.columns.get_loc('Centromere_Distance')
                        data_sample[:actual_length, features.index('Centr') + 1] = region_data.iloc[:, centr_index].values
                    elif feature == 'Chrom':
                        chrom_value = list(chromosome_length.keys()).index(chrom) + 1  # +1 to avoid zero
                        data_sample[:actual_length, features.index('Chrom') + 1] = chrom_value
                
                # Store as dictionary with metadata
                region_dict = {
                    'data': data_sample,
                    'actual_length': actual_length,
                    'start': start,
                    'end': end
                }
                regions[dataset][chrom].append(region_dict)           
    return regions
            
        

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
        split_on (str): Feature to split data on ('Chrom', 'Dataset').
        
    Returns:
        train_data (DataFrame): Training data.
        val_data (DataFrame): Validation data.
        test_data (DataFrame): Testing data.
    """
    train_data = {}
    val_data = {}
    test_data = {}
    if split_on == 'Chrom':
        all_chroms = list(chromosome_length.keys())
        train_size = train_val_test_split[0]
        val_size = train_val_test_split[1]
        test_size = train_val_test_split[2]
        train_chroms, temp_chroms = train_test_split(all_chroms, train_size=train_size, random_state=42)
        if test_size + val_size > 0:
            val_chroms, test_chroms = train_test_split(temp_chroms, test_size=test_size/(test_size + val_size), random_state=42)
        else:
            val_chroms = []
            test_chroms = []
        for dataset in data:
            train_data[dataset] = {}
            val_data[dataset] = {}
            test_data[dataset] = {}
            for chrom in data[dataset]:
                if chrom in train_chroms:
                    train_data[dataset][chrom] = data[dataset][chrom]
                elif chrom in val_chroms:
                    val_data[dataset][chrom] = data[dataset][chrom]
                elif chrom in test_chroms:
                    test_data[dataset][chrom] = data[dataset][chrom]
    elif split_on == 'Dataset':
        all_datasets = list(data.keys())
        train_size = train_val_test_split[0]
        val_size = train_val_test_split[1]
        test_size = train_val_test_split[2]
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

def get_mask_from_data(region_dict):
    """Create a boolean mask indicating which positions contain real data vs padding.
    
    Args:
        region_dict (dict): Region dictionary with 'data' and 'actual_length'.
        
    Returns:
        mask (np.array): Boolean array where True = real data, False = padding.
    """
    data_sample = region_dict['data']
    actual_length = region_dict['actual_length']
    mask = np.zeros(len(data_sample), dtype=bool)
    mask[:actual_length] = True
    return mask

def extract_arrays_and_masks(data):
    """Extract data arrays and masks from the dictionary format.
    Useful for preparing data for neural network training.
    
    Args:
        data (Dict): Data dictionary with structure {dataset: {chrom: [region_dicts, ...]}}.
        
    Returns:
        arrays (list): List of all data arrays.
        masks (list): List of corresponding boolean masks.
        metadata (list): List of metadata dicts with 'dataset', 'chrom', 'start', 'end', 'actual_length'.
    """
    arrays = []
    masks = []
    metadata = []
    for dataset in data:
        for chrom in data[dataset]:
            for region_dict in data[dataset][chrom]:
                arrays.append(region_dict['data'])
                masks.append(get_mask_from_data(region_dict))
                metadata.append({
                    'dataset': dataset,
                    'chrom': chrom,
                    'start': region_dict['start'],
                    'end': region_dict['end'],
                    'actual_length': region_dict['actual_length']
                })
    return arrays, masks, metadata

def preprocess_data(input_folder, 
                    gene_file,
                    features = ['Pos', 'Chrom', 'Nucl', 'Centr'], 
                    train_val_test_split = [0.7, 0.15, 0.15], 
                    split_on = 'Chrom',
                    bin = 5, bin_method = 'average', 
                    distance_around_genes = 160, 
                    minimum_region_size = 200,
                    maximum_region_size = 3000,
                    scaling = True):
    """Preprocess data for autoencoder training and testing.
    
    Data is returned as dictionaries where each region is a dict with:
        - 'data': padded numpy array
        - 'actual_length': number of real data points
        - 'start': genomic start position
        - 'end': genomic end position
    
    Regions with zero data points are automatically filtered out.
    
    Args:
        input_folder (str): Path to the folder containing input data files.
        gene_file (str): Path to the gene annotation file.
        features (list): List of features to include in the data.
        train_val_test_split (list): Proportions for training, validation, and testing sets.
        split_on (str): Feature to split data on ('Chrom', 'Dataset').
        bin (int): Size of bins for data aggregation.
        bin_method (str): Method for binning ('average', 'sum', 'max', 'min', 'median').
        distance_around_genes (int): Distance around genes to consider.
        minimum_region_size (int): Minimum size of regions to become its own data point.
        maximum_region_size (int): Maximum size of regions.
        scaling (bool): Whether to standardize features.
        
    Returns:
        train_data (Dict): Training data with structure {dataset: {chrom: [region_dicts, ...]}}.
        val_data (Dict): Validation data with same structure.
        test_data (Dict): Testing data with same structure.
        scalers (Dict): Dictionary of StandardScaler objects used for each feature.
    """
    # Read genes json file
    with open(gene_file, 'r') as f:
        genes_data = json.load(f)
    # Read transposon insertion data
    transposon_data = read_csv_file_with_distances(input_folder)
    # Split genes into regions
    regions_list = split_genes(genes_data, distance_around_genes)
    # Fill gaps between regions
    regions_list = fill_gaps(regions_list, minimum_region_size)
    # Resolve overlapping regions
    regions_list = resolve_overlaps(regions_list)
    # Divide long sequences into smaller regions
    regions_list = divide_long_sequences(regions_list, maximum_region_size)
    # Add transposon insertion data and features
    regions_list = add_transposon_and_features(transposon_data, regions_list, features, maximum_region_size)
    # Bin data if bin size > 1
    if bin > 1:
        regions_list = bin_data(regions_list, bin, bin_method, maximum_region_size)
    # Preprocess counts
    regions_list, stats = preprocess_counts(regions_list)
    # Split data into training, validation, and testing sets
    train_data, val_data, test_data = split_data(regions_list, train_val_test_split, split_on)
    # Standardize features of the training set and apply the same transformation to validation and testing sets
    if scaling:
        train_data, val_data, test_data, scalers = standardize_data(train_data, val_data, test_data, features)
    else:
        scalers = {}
    # Standardize the count data in the training set and apply the same transformation to validation and testing sets
    return train_data, val_data, test_data, scalers



if __name__ == "__main__":
    input_folder = "Data/test"
    gene_file = "SGD_API/architecture_info/yeast_genes_with_info.json"
    train_data, val_data, test_data, scalers = preprocess_data(input_folder, gene_file, train_val_test_split=[1, 0, 0], scaling=True)
    # This gives 9153 regions in total
    
    # print the shapes of the datasets
    # for dataset in train_data:
    #     for chrom in train_data[dataset]:
    #         for region_dict in train_data[dataset][chrom]:
    #             print(f"Dataset: {dataset}, Chromosome: {chrom}, "
    #                   f"Region: {region_dict['start']}-{region_dict['end']}, "
    #                   f"Shape: {region_dict['data'].shape}, "
    #                   f"Actual length: {region_dict['actual_length']}")
    
    # Save the training data as a json file after converting numpy arrays to lists
    # train_data_json = {}
    # for dataset in train_data:
    #     train_data_json[dataset] = {}
    #     for chrom in train_data[dataset]:
    #         train_data_json[dataset][chrom] = [
    #             {
    #                 'data': region_dict['data'].tolist(),
    #                 'actual_length': region_dict['actual_length'],
    #                 'start': region_dict['start'],
    #                 'end': region_dict['end']
    #             }
    #             for region_dict in train_data[dataset][chrom]
    #         ]
    # with open("Data/train_data_with_scalers.json", "w") as f:
    #     json.dump(train_data_json, f, indent=4)

        
    