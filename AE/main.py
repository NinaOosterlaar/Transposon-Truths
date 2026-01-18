import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AE.preprocessing.preprocessing import preprocess_data
from AE.architectures.Autoencoder import AE
from AE.training.training import train, test

# Preprocessing
FEATURES = ['Nucl', 'Centr']
BIN_SIZE = 10
MOVING_AVERAGE = False
DATA_POINT_LENGTH = 2000
STEP_SIZE = 0.25
SAMPLE_FRACTION = 1.0

SPLIT_ON = 'Chrom'
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]

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
KL_BETA = 1.0
PI_THRESHOLD = 0.5
MASKED_RECON_WEIGHT = 0.0  # gamma: weight for masked reconstruction loss
OPTIMIZER = 'adam'
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
LAYERS = [512, 256, 128]
MODEL = 'ZINBAE'
REGULARIZER = 'none'
REGULARIZATION_WEIGHT = 1e-4


def main(
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
    pooling_operation=POOLING_OPERATION, 
    kernel_size=KERNEL_SIZE,
    padding=PADDING, 
    stride=STRIDE,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    noise_level=NOISE_LEVEL,
    kl_beta=KL_BETA,
    pi_threshold=PI_THRESHOLD,
    masked_recon_weight=MASKED_RECON_WEIGHT, 
    optimizer=OPTIMIZER, 
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT_RATE, 
    layers=LAYERS, 
    model=MODEL,
    regularizer=REGULARIZER, 
    regularization_weight=REGULARIZATION_WEIGHT):
    # Preprocess data
    train, val, test, scalers, count_stats, clip_stats = preprocess_data(
        features=features,
        bin_size=bin_size,
        moving_average=moving_average,
        data_point_length=data_point_length,
        step_size=step_size,
        split_on=split_on,
        train_val_test_split=train_val_test_split
    )
