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

