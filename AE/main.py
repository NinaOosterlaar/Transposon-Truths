from preprocessing import preprocess_data
from AE import AE, train, test

FEATURES = ['Pos', 'Chrom', 'Nucl', 'Centr']
TRAIN_VAL_TEST_SPLIT = [0.7, 0, 0.3]
SPLIT_ON = 'Chrom'
CHUNK_SIZE = 50000
NORMALIZE = True
BIN_SIZE = 10
MOVING_AVERAGE = True
DATA_POINT_LENGTH = 2000
STEP_SIZE = 500

BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LAYERS = [512, 256, 128]
CHROMOSOME_EMBEDDING_DIM = 4
MODEL = 'AE'  # Options: 'AE' or 'VAE'


def main():
    train_data, val_data, test_data, scalers = preprocess_data(
        features=FEATURES,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        split_on=SPLIT_ON,
        chunk_size=CHUNK_SIZE,
        normalize=NORMALIZE,
        bin_size=BIN_SIZE,
        moving_average=MOVING_AVERAGE,
        data_point_length=DATA_POINT_LENGTH,
        step_size=STEP_SIZE
    )

    train_loader = AE.get_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True, chrom=True)
    val_loader = AE.get_dataloader(val_data, batch_size=BATCH_SIZE, shuffle=False, chrom=True)
    test_loader = AE.get_dataloader(test_data, batch_size=BATCH_SIZE, shuffle=False, chrom=True)

    if "Chrom" in FEATURES:
        feature_dim = len(FEATURES) + CHROMOSOME_EMBEDDING_DIM
    else:
        feature_dim = len(FEATURES) + 1
    if MODEL == 'VAE':
        # Not implemented yet
        raise NotImplementedError("VAE model is not implemented yet.")
    else:
        model = AE(seq_length=DATA_POINT_LENGTH, feature_dim=feature_dim, layers=LAYERS)  # Use the calculated feature_dim

    model = train(
        model,
        train_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        chrom=True
    )
    
    reconstructions, latents = test(
        model,
        test_loader,
        scalers,
        chrom=True
    )
    
    return reconstructions, latents
