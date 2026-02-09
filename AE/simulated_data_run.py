import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import glob
from AE.preprocessing.preprocessing import preprocess_counts, split_data, process_data, remove_empty_datapoints
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array
from AE.training.training import train, test
import gc


# Preprocessing parameters
SIMULATED_DATA_FOLDER = "Data/simulated_small"
BIN_SIZE = 1
MOVING_AVERAGE = False
DATA_POINT_LENGTH = 2000
STEP_SIZE = 0.25
SAMPLE_FRACTION = 1.0

# Split parameters (split by file/chromosome)
TRAIN_VAL_TEST_SPLIT = [0.7, 0, 0.3]

# Model parameters
USE_CONV = True
CONV_CHANNEL = 64
POOL_SIZE = 2
KERNEL_SIZE = 5
PADDING = 'same'
STRIDE = 1

EPOCHS = 30
BATCH_SIZE = 64
NOISE_LEVEL = 0.3
PI_THRESHOLD = 0.5
MASKED_RECON_WEIGHT = 1.0
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
LAYERS = [512, 256, 128]
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 0
lambda_mu = 0.2
lambda_pi = 0.2
GLOBAL_THETA = False

# Output configuration
OUTPUT_NAME = "Simulated_data_fused"  # Name for results folder
PLOT = True


def load_simulated_data(data_folder):
    """Load all simulated CSV files, treating each as a separate chromosome.
    
    Wraps data in nested structure {dataset: {chrom: DataFrame}} to match
    the format expected by existing preprocessing functions.
    
    Args:
        data_folder (str): Path to folder containing realistic_data_*.csv files
        
    Returns:
        dict: Dictionary with structure {"SimulatedData": {chrom_name: DataFrame}}
    """
    print(f"Loading simulated data from {data_folder}...")
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_folder, "realistic_data_*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No realistic_data_*.csv files found in {data_folder}")
    
    print(f"Found {len(csv_files)} simulated data files")
    
    # Load each file as a separate "chromosome"
    # Wrap in a single "dataset" to match expected structure
    chromosomes = {}
    for file_path in csv_files:
        # Extract file ID from filename (e.g., "realistic_data_42.csv" -> "file_42")
        basename = os.path.basename(file_path)
        file_id = basename.replace("realistic_data_", "").replace(".csv", "")
        chrom_name = f"SimChrom_{file_id}"
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        if 'Position' not in df.columns or 'Value' not in df.columns:
            raise ValueError(f"File {file_path} must have 'Position' and 'Value' columns")
        
        chromosomes[chrom_name] = df[['Position', 'Value']].copy()
    
    # Wrap in dataset structure
    data = {"SimulatedData": chromosomes}
    
    print(f"Loaded {len(chromosomes)} simulated chromosomes")
    return data


def preprocess_simulated(
    data_folder=SIMULATED_DATA_FOLDER,
    bin_size=BIN_SIZE,
    moving_average=MOVING_AVERAGE,
    data_point_length=DATA_POINT_LENGTH,
    step_size=STEP_SIZE,
    train_val_test_split=TRAIN_VAL_TEST_SPLIT,
    zinb_mode=True
):
    """Full preprocessing pipeline for simulated data.
    
    Reuses existing preprocessing functions by wrapping data in expected format.
    
    Returns:
        train, val, test: numpy arrays
        stats: preprocessing statistics
    """
    # Load data (wrapped in {dataset: {chrom: DataFrame}} structure)
    data = load_simulated_data(data_folder)
    
    # Preprocess counts using existing function
    data, count_stats = preprocess_counts(data, zinb_mode=zinb_mode)
    
    # Split by chromosome using existing function
    train_data, val_data, test_data = split_data(
        data, train_val_test_split, split_on='Chrom'
    )
    
    # Calculate step size
    if not moving_average:
        adjusted_data_point_length = data_point_length // bin_size
    else:
        adjusted_data_point_length = data_point_length
    
    step_size_bp = int(adjusted_data_point_length * step_size)
    
    # Features: empty list since we only use 'Value' (no Nucl, Centr, etc.)
    features = []
    
    # Process each split using existing function
    print("\n=== Processing Training Data ===")
    train = process_data(train_data, features, bin_size, moving_average, step_size_bp, 
                        adjusted_data_point_length, split_on='Chrom', zinb_mode=zinb_mode)
    train = remove_empty_datapoints(train)
    gc.collect()
    
    print("\n=== Processing Validation Data ===")
    if val_data:
        val = process_data(val_data, features, bin_size, moving_average, step_size_bp,
                          adjusted_data_point_length, split_on='Chrom', zinb_mode=zinb_mode)
        val = remove_empty_datapoints(val)
    else:
        val = None
    gc.collect()
    
    print("\n=== Processing Test Data ===")
    if test_data:
        test = process_data(test_data, features, bin_size, moving_average, step_size_bp,
                           adjusted_data_point_length, split_on='Chrom', zinb_mode=zinb_mode)
        test = remove_empty_datapoints(test)
    else:
        test = None
    gc.collect()
    
    return train, val, test, count_stats


def main_simulated(
    data_folder=SIMULATED_DATA_FOLDER,
    bin_size=BIN_SIZE,
    moving_average=MOVING_AVERAGE,
    data_point_length=DATA_POINT_LENGTH,
    step_size=STEP_SIZE,
    sample_fraction=SAMPLE_FRACTION,
    train_val_test_split=TRAIN_VAL_TEST_SPLIT,
    use_conv=USE_CONV,
    conv_channel=CONV_CHANNEL,
    pool_size=POOL_SIZE,
    kernel_size=KERNEL_SIZE,
    padding=PADDING,
    stride=STRIDE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    noise_level=NOISE_LEVEL,
    pi_threshold=PI_THRESHOLD,
    masked_recon_weight=MASKED_RECON_WEIGHT,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE,
    layers=LAYERS,
    regularizer=REGULARIZER,
    regularization_weight=REGULARIZATION_WEIGHT,
    lambda_mu=lambda_mu,
    lambda_pi=lambda_pi,
    global_theta=GLOBAL_THETA,
    plot=PLOT,
    output_name=OUTPUT_NAME
):
    """Main function for training on simulated data."""
    
    # Adjust data point length if not using moving average
    if not moving_average:
        adjusted_data_point_length = data_point_length // bin_size
    else:
        adjusted_data_point_length = data_point_length
    
    # Preprocess data
    print("\n" + "="*60)
    print("PREPROCESSING SIMULATED DATA")
    print("="*60)
    train_set, val_set, test_set, count_stats = preprocess_simulated(
        data_folder=data_folder,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=adjusted_data_point_length,
        step_size=step_size,
        train_val_test_split=train_val_test_split,
        zinb_mode=True
    )
    
    print(f"\nFinal dataset shapes:")
    print(f"  Train: {train_set.shape}")
    if val_set is not None:
        print(f"  Val: {val_set.shape}")
    if test_set is not None:
        print(f"  Test: {test_set.shape}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    
    train_dataloader = dataloader_from_array(
        train_set, batch_size=batch_size, shuffle=True, zinb=True, 
        chrom=False, sample_fraction=sample_fraction, denoise_percentage=noise_level
    )
    
    if val_set is not None:
        val_dataloader = dataloader_from_array(
            val_set, batch_size=batch_size, shuffle=False, zinb=True,
            chrom=False, sample_fraction=1.0, denoise_percentage=noise_level
        )
    else:
        val_dataloader = None
    
    if test_set is not None:
        test_dataloader = dataloader_from_array(
            test_set, batch_size=batch_size, shuffle=False, zinb=True,
            chrom=False, sample_fraction=1.0, denoise_percentage=noise_level
        )
    else:
        test_dataloader = None
    
    # Determine feature dimension (Value + mask indicator)
    feature_dim = train_dataloader.dataset.tensors[0].shape[2]
    feature_dim += 1  # Add mask indicator
    
    # Get actual sequence length from the data
    actual_seq_length = train_dataloader.dataset.tensors[0].shape[1]
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Actual sequence length from data: {actual_seq_length}")
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    zinbae_model = ZINBAE(
        seq_length=actual_seq_length,
        feature_dim=feature_dim,
        layers=layers,
        use_conv=use_conv,
        conv_channels=conv_channel,
        pool_size=pool_size,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dropout=dropout_rate,
        global_theta=global_theta
    )
    
    print(f"Model created with seq_length={actual_seq_length}, feature_dim={feature_dim}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    _, train_metrics = train(
        model=zinbae_model,
        dataloader=train_dataloader,
        num_epochs=epochs,
        pi_threshold=pi_threshold,
        learning_rate=learning_rate,
        regularizer=regularizer,
        alpha=regularization_weight,
        denoise_percent=noise_level,
        gamma=masked_recon_weight,
        lambda_mu=lambda_mu,
        lambda_pi=lambda_pi,
        chrom=False,
        plot=plot,
        name=output_name,  # Pass output name for results folder
    )
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Evaluate on validation set if available, otherwise test set
    eval_dataloader = val_dataloader if val_dataloader is not None else test_dataloader
    eval_name = "Validation" if val_dataloader is not None else "Test"
    
    if eval_dataloader is not None:
        print(f"Evaluating on {eval_name} set...")
        _, _, eval_metrics = test(
            model=zinbae_model,
            dataloader=eval_dataloader,
            pi_threshold=pi_threshold,
            chrom=False,
            chrom_embedding=None,
            plot=plot,
            denoise_percent=noise_level,
            alpha=regularization_weight,
            gamma=masked_recon_weight,
            regularizer=regularizer,
            name=output_name,  # Pass output name for results folder
        )
        
        print(f"\n{eval_name} Metrics:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value}")
    else:
        eval_metrics = None
        print("No validation or test set available for evaluation")
    
    return train_metrics, eval_metrics


if __name__ == "__main__":
    print("Starting simulated data run...")
    train_metrics, eval_metrics = main_simulated()
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)