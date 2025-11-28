import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os, sys
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 


def plot_test_results(all_originals, all_reconstructions, model_type='AE', 
                      save_dir='AE/results/testing', n_examples=5, metrics=None):
    """
    Create comprehensive visualizations of test results.
    
    Parameters:
    -----------
    all_originals : np.ndarray
        Original sequences (shape: [n_samples, seq_length])
    all_reconstructions : np.ndarray
        Reconstructed sequences (shape: [n_samples, seq_length])
    model_type : str
        Type of model ('AE' or 'VAE')
    save_dir : str
        Directory to save plots
    n_examples : int
        Number of example reconstructions to plot
    metrics : dict
        Dictionary of metrics to save
    """
    # Create visualization directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics for plotting
    mae = mean_absolute_error(all_originals.flatten(), all_reconstructions.flatten())
    r2 = r2_score(all_originals.flatten(), all_reconstructions.flatten())
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_type}_{timestamp}"
    
    # 1. Plot actual vs predicted scatter plot and residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample points for scatter plot (use every 10th point to avoid overcrowding)
    sample_indices = np.arange(0, all_originals.size, 10)
    axes[0].scatter(all_originals.flatten()[sample_indices], 
                   all_reconstructions.flatten()[sample_indices], 
                   alpha=0.3, s=1)
    axes[0].plot([all_originals.min(), all_originals.max()], 
                [all_originals.min(), all_originals.max()], 
                'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Log Counts')
    axes[0].set_ylabel('Predicted Log Counts')
    axes[0].set_title(f'{model_type}: Actual vs Predicted (R²={r2:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Plot residuals
    residuals = all_originals.flatten() - all_reconstructions.flatten()
    axes[1].hist(residuals[sample_indices], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_type}: Residual Distribution (MAE={mae:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_test_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot example reconstructions
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(min(n_examples, len(all_originals))):
        axes[i].plot(all_originals[i], label='Original', linewidth=2, alpha=0.7)
        axes[i].plot(all_reconstructions[i], label='Reconstructed', 
                    linewidth=2, alpha=0.7, linestyle='--')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Log Count')
        axes[i].set_title(f'{model_type}: Example {i+1} Reconstruction')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_example_reconstructions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save metrics to JSON file
    if metrics is not None:
        metrics_to_save = {
            'model_type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'summary_statistics': {
                'mean_original': float(np.mean(all_originals)),
                'std_original': float(np.std(all_originals)),
                'mean_reconstruction': float(np.mean(all_reconstructions)),
                'std_reconstruction': float(np.std(all_reconstructions)),
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals))
            }
        }
        
        metrics_file = os.path.join(save_dir, f'{prefix}_test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
    
    print(f"Plots saved to {save_dir}/")


def plot_training_loss(losses, model_type='AE', save_path='AE/results/training/training_loss.png', 
                       save_losses=True):
    """
    Plot training loss over epochs.
    
    Parameters:
    -----------
    losses : list
        List of loss values per epoch
    model_type : str
        Type of model ('AE' or 'VAE')
    save_path : str
        Path to save the plot
    save_losses : bool
        Whether to save loss values to a file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update save path with model type and timestamp
    base_dir = os.path.dirname(save_path)
    prefix = f"{model_type}_{timestamp}"
    plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_type}: Training Loss over Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {plot_path}")
    
    # Save loss values to JSON
    if save_losses:
        loss_data = {
            'model_type': model_type,
            'timestamp': timestamp,
            'num_epochs': len(losses),
            'final_loss': float(losses[-1]),
            'min_loss': float(min(losses)),
            'max_loss': float(max(losses)),
            'losses_per_epoch': [float(loss) for loss in losses]
        }
        
        loss_file = os.path.join(base_dir, f'{prefix}_training_losses.json')
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=4)
        print(f"Training losses saved to {loss_file}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss over Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {save_path}")
