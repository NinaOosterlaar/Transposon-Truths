import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from AE.plotting.results_standard import plot_test_results, plot_binary_test_results
from AE.plotting.plot_loss import plot_training_loss, plot_binary_training_loss, plot_zinb_training_loss
from AE.plotting.results_ZINB import plot_zinb_test_results
import argparse
from AE.architectures.Autoencoder import AE, VAE
from AE.architectures.Autoencoder_binary import AE_binary, VAE_binary
from AE.architectures.ZINBAE import ZINBAE, ZINBVAE
from AE.training.loss_functions import zinb_nll
from AE.training.training_utils import ChromosomeEmbedding, add_noise, dataloader_from_array, gaussian_kl

# Preprocessing hyperparameters
FEATURES = [["Centr", "Nucl"], ["Centr"], ["Nucl"]]
SEQUENCE_LENGTHS = [500, 1000, 2000, 3000, 5000, 10000]
STEP_SIZES = [1/20, 1/10, 1/5, 1/2, 1]  # relative sequence lengths
BIN_SIZES = [5, 10, 20, 50, 75, 100]
SAMPLE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]

# Model Architecture hyperparameters
MODEL_TYPE = ["ZINBAE", "ZINBVAE"]
LAYERS = [[512, 256, 128], [256, 128, 64], [128, 64, 32]
          ]

# Convolutional Layer hyperparameters
USE_CONV = [True, False]
CONV_CHANNELS = [16, 32, 64, 128]
POOL_SIZES = [2, 4, 8]
POOLING_OPERATIONS = ['max', 'avg']
KERNEL_SIZES = [3, 5, 7, 9, 11, 13]
PADDING = ['same']
STRIDE = [1, 2, 3]

# Training hyperparameters
EPOCHS = [30, 50, 70, 100, 150]
BATCH_SIZES = [32, 64, 128, 256]
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
KL_BETA = [0.1, 0.5, 1.0, 1.5, 2.0]
OPTIMIZERS = ['adam']
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PI_THRESHOLD = [0.3, 0.5, 0.7]

# Regularization hyperparameters
REGULARIZATIONS = ["l1", "l2", "none"]
REGULARIZATION_WEIGHTS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]



