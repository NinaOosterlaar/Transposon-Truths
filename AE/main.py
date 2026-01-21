import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess
from AE.architectures.ZINBAE import ZINBAE
from AE.training.training_utils import dataloader_from_array, ChromosomeEmbedding
from AE.training.training import train, test

# Preprocessing
INPUT_FOLDER = "Data/test/strain_FD"
FEATURES = ['Nucl', 'Centr']
BIN_SIZE = 10
MOVING_AVERAGE = False
DATA_POINT_LENGTH = 2000
STEP_SIZE = 0.25
SAMPLE_FRACTION = 0.01

SPLIT_ON = 'Chrom'
TRAIN_VAL_TEST_SPLIT = [0.5, 0, 0.5]

USE_CONV = True
CONV_CHANNEL = 64
POOL_SIZE = 2
POOLING_OPERATION = 'max'
KERNEL_SIZE = 5
PADDING = 'same'
STRIDE = 1

EPOCHS = 100
BATCH_SIZE = 64
NOISE_LEVEL = 0.3
PI_THRESHOLD = 0.5
MASKED_RECON_WEIGHT = 0.0  # gamma: weight for masked reconstruction loss
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
LAYERS = [512, 256, 128]
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-4

PLOT = False


def main_with_datasets(
    train_set,
    val_set,
    test_set,
    features=FEATURES,
    data_point_length=DATA_POINT_LENGTH,
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
    sample_fraction=SAMPLE_FRACTION,
    plot=PLOT,
    eval_on_val=True):
    """
    Main training function that accepts pre-made datasets.
    Used by Bayesian optimization to avoid recreating data every trial.
    
    Parameters:
    -----------
    train_set, val_set, test_set : numpy arrays
        Pre-processed datasets
    eval_on_val : bool
        If True, evaluate on validation set (for hyperparameter tuning).
        If False, evaluate on test set (for final evaluation).
    """
    # Initialize model
    if "Chr" in features: chrom = True
    else: chrom = False
    
    train_dataloader = dataloader_from_array(train_set, batch_size=batch_size, shuffle=True, zinb=True, chrom=chrom, sample_fraction=sample_fraction, denoise_percentage=noise_level)
    if val_set is not None:
        val_dataloader = dataloader_from_array(val_set, batch_size=batch_size, shuffle=False, zinb=True, chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level)
    if test_set is not None:
        test_dataloader = dataloader_from_array(test_set, batch_size=batch_size, shuffle=False, zinb=True, chrom=chrom, sample_fraction=1.0, denoise_percentage=noise_level)
    if chrom: chrom_embedding = ChromosomeEmbedding()
    else: chrom_embedding = None
    
    feature_dim = train_dataloader.dataset.tensors[0].shape[2]
    feature_dim += 1
    if chrom in features: feature_dim += chrom_embedding.embedding.embedding_dim 
    
    zinbae_model = ZINBAE(
        seq_length=data_point_length,
        feature_dim=feature_dim,
        layers=layers,
        use_conv=use_conv,
        conv_channels=conv_channel,
        pool_size=pool_size,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dropout=dropout_rate,
    )
    # Train model
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
        chrom=chrom,
    )
    # Evaluate model on validation or test set
    eval_dataloader = val_dataloader if eval_on_val else test_dataloader
    _, _, eval_metrics = test(
        model=zinbae_model,
        dataloader=eval_dataloader,
        pi_threshold=pi_threshold,
        chrom=chrom,
        chrom_embedding=chrom_embedding,
        plot=plot,
        denoise_percent=noise_level,
        alpha=regularization_weight,
        gamma=masked_recon_weight,
        regularizer=regularizer,
    )
    
    return train_metrics, eval_metrics


def main(
    input_folder=INPUT_FOLDER,
    features=FEATURES, 
    bin_size=BIN_SIZE, 
    moving_average=MOVING_AVERAGE,
    data_point_length=DATA_POINT_LENGTH, 
    step_size=STEP_SIZE,
    sample_fraction=SAMPLE_FRACTION, 
    split_on=SPLIT_ON,
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
    plot=PLOT):
    """
    Original main function that handles data preprocessing.
    For backwards compatibility and standalone usage.
    """
    if not moving_average:
        data_point_length = data_point_length // bin_size
    
    # Preprocess data
    train_set, val_set, test_set, _, _, _ = preprocess(
        input_folder=input_folder,
        features=features,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=data_point_length,
        step_size=step_size,
        split_on=split_on,
        train_val_test_split=train_val_test_split
    )
    
    # Use val_set for evaluation if available, otherwise test_set
    eval_on_val = val_set is not None
    if not eval_on_val:
        val_set = test_set  # Pass test_set as val_set if no val_set
    
    return main_with_datasets(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        features=features,
        data_point_length=data_point_length,
        use_conv=use_conv,
        conv_channel=conv_channel,
        pool_size=pool_size,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        noise_level=noise_level,
        pi_threshold=pi_threshold,
        masked_recon_weight=masked_recon_weight,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        layers=layers,
        regularizer=regularizer,
        regularization_weight=regularization_weight,
        sample_fraction=sample_fraction,
        plot=plot,
        eval_on_val=eval_on_val
    )

if __name__ == "__main__":
    main()