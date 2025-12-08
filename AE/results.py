import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os, sys
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

# Get the directory where this script is located (AE folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of AE folder)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def plot_test_results(all_originals, all_reconstructions, model_type='AE', 
                      save_dir=None, n_examples=5, metrics=None, use_conv=False, name=""):
    """
    Create comprehensive visualizations of test results.
    
    Parameters:
    -----------
    all_originals : np.ndarray
        Original sequences (shape: [n_samples, seq_length])
    all_reconstructions : np.ndarray
        Reconstructed sequences (shape: [n_samples, seq_length])
    model_type : str
        Type of model ('AE', 'VAE', 'ZINBAE', or 'ZINBVAE')
    save_dir : str or None
        Directory to save plots. If None, uses default 'AE/results/testing'
    n_examples : int
        Number of example reconstructions to plot
    metrics : dict
        Dictionary of metrics to save
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    # Use default save directory if not provided
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'results', 'testing')
    
    # Create visualization directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics for plotting
    mae = mean_absolute_error(all_originals.flatten(), all_reconstructions.flatten())
    r2 = r2_score(all_originals.flatten(), all_reconstructions.flatten())
    
    # Determine if ZINB model
    is_zinb = 'ZINB' in model_type
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
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
    
    if is_zinb:
        axes[0].set_xlabel('Actual Log Counts (Normalized)')
        axes[0].set_ylabel('Predicted Mean (μ)')
    else:
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
    
    y_label = 'Normalized Log Count' if is_zinb else 'Log Count'
    
    for i in range(min(n_examples, len(all_originals))):
        axes[i].plot(all_originals[i], label='Original', linewidth=2, alpha=0.7)
        axes[i].plot(all_reconstructions[i], label='Reconstructed (μ)' if is_zinb else 'Reconstructed', 
                    linewidth=2, alpha=0.7, linestyle='--')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel(y_label)
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
            'use_conv': use_conv,
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


def plot_training_loss(losses, model_type='AE', save_path=None, 
                       save_losses=True, use_conv=False, name=""):
    """
    Plot training loss over epochs.
    
    Parameters:
    -----------
    losses : list
        List of loss values per epoch
    model_type : str
        Type of model ('AE', 'VAE', 'ZINBAE', or 'ZINBVAE')
    save_path : str or None
        Path to save the plot. If None, uses default directory
    save_losses : bool
        Whether to save loss values to a file
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
    # Determine loss type based on model
    is_zinb = 'ZINB' in model_type
    loss_label = 'Loss (ZINB NLL)' if is_zinb else 'Loss (MSE)'
    
    # Use default directory if save_path not provided
    if save_path is None:
        base_dir = os.path.join(SCRIPT_DIR, 'results', 'training')
    else:
        base_dir = os.path.dirname(save_path)
    
    os.makedirs(base_dir, exist_ok=True)
    
    plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel(loss_label)
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
            'use_conv': use_conv,
            'loss_type': 'ZINB_NLL' if is_zinb else 'MSE',
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


def plot_binary_test_results(all_originals, all_reconstructions, all_probabilities, 
                             model_type='AE_binary', save_dir=None, 
                             n_examples=5, metrics=None, use_conv=False, name=""):
    """
    Create comprehensive visualizations of binary classification test results.
    
    Parameters:
    -----------
    all_originals : np.ndarray
        Original binary labels (shape: [n_samples, seq_length])
    all_reconstructions : np.ndarray
        Binary predictions (shape: [n_samples, seq_length])
    all_probabilities : np.ndarray
        Predicted probabilities (shape: [n_samples, seq_length])
    model_type : str
        Type of model ('AE_binary' or 'VAE_binary')
    save_dir : str or None
        Directory to save plots. If None, uses default
    n_examples : int
        Number of example reconstructions to plot
    metrics : dict
        Dictionary of metrics to save
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    # Use default save directory if not provided
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'results', 'testing')
    
    # Create visualization directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
    # Flatten for metric calculations
    y_true = all_originals.flatten()
    y_pred = all_reconstructions.flatten()
    y_prob = all_probabilities.flatten()
    
    # 1. Confusion Matrix and ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].figure.colorbar(im, ax=axes[0])
    axes[0].set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Absent (0)', 'Present (1)'],
               yticklabels=['Absent (0)', 'Present (1)'],
               ylabel='True label',
               xlabel='Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16)
    axes[0].set_title(f'{model_type}: Confusion Matrix')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_type}: ROC Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix_roc.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Probability Distribution by True Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of predicted probabilities for true 0s and 1s
    prob_absent = y_prob[y_true == 0]
    prob_present = y_prob[y_true == 1]
    
    axes[0].hist(prob_absent, bins=50, alpha=0.7, label='True Absent (0)', 
                color='blue', edgecolor='black')
    axes[0].hist(prob_present, bins=50, alpha=0.7, label='True Present (1)', 
                color='red', edgecolor='black')
    axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, 
                   label='Threshold (0.5)')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{model_type}: Probability Distribution by True Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot of probabilities by true class
    axes[1].boxplot([prob_absent, prob_present], labels=['Absent (0)', 'Present (1)'])
    axes[1].axhline(y=0.5, color='green', linestyle='--', linewidth=2, 
                   label='Threshold')
    axes[1].set_ylabel('Predicted Probability')
    axes[1].set_xlabel('True Class')
    axes[1].set_title(f'{model_type}: Probability Boxplot by True Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_probability_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot example reconstructions (showing probabilities and binary predictions)
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(min(n_examples, len(all_originals))):
        ax = axes[i]
        positions = np.arange(len(all_originals[i]))
        
        # Plot as bar chart with probabilities
        colors = ['red' if orig == 1 else 'blue' for orig in all_originals[i]]
        ax.bar(positions, all_probabilities[i], alpha=0.6, label='Predicted Probability', 
               color=colors, edgecolor='black', linewidth=0.5)
        
        # Add horizontal line at threshold
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1, 
                  label='Threshold (0.5)', alpha=0.7)
        
        # Mark true positives and true negatives
        true_present = all_originals[i] == 1
        true_absent = all_originals[i] == 0
        
        ax.scatter(positions[true_present], [1.05]*np.sum(true_present), 
                  marker='v', color='red', s=10, alpha=0.7, 
                  label='True Present (1)')
        ax.scatter(positions[true_absent], [1.05]*np.sum(true_absent), 
                  marker='v', color='blue', s=10, alpha=0.7, 
                  label='True Absent (0)')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Predicted Probability')
        ax.set_ylim([0, 1.1])
        ax.set_title(f'{model_type}: Example {i+1} - Binary Reconstruction')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_example_reconstructions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Metrics Summary Bar Chart
    if metrics is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('auc_roc', 0)
        ]
        
        bars = ax.bar(metric_names, metric_values, color='skyblue', 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.set_title(f'{model_type}: Classification Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_metrics_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Save metrics to JSON file
    if metrics is not None:
        # Calculate class-wise statistics
        n_absent = np.sum(y_true == 0)
        n_present = np.sum(y_true == 1)
        
        metrics_to_save = {
            'model_type': model_type,
            'timestamp': timestamp,
            'use_conv': use_conv,
            'metrics': metrics,
            'class_distribution': {
                'n_absent': int(n_absent),
                'n_present': int(n_present),
                'percent_absent': float(n_absent / len(y_true) * 100),
                'percent_present': float(n_present / len(y_true) * 100)
            },
            'probability_statistics': {
                'mean_prob_when_absent': float(np.mean(prob_absent)),
                'std_prob_when_absent': float(np.std(prob_absent)),
                'mean_prob_when_present': float(np.mean(prob_present)),
                'std_prob_when_present': float(np.std(prob_present))
            },
            'confusion_matrix': cm.tolist()
        }
        
        metrics_file = os.path.join(save_dir, f'{prefix}_test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
    
    print(f"Binary classification plots saved to {save_dir}/")


def plot_binary_training_loss(losses, model_type='AE_binary', 
                              save_path=None, 
                              save_losses=True, use_conv=False, name=""):
    """
    Plot training loss over epochs for binary models.
    
    Parameters:
    -----------
    losses : list
        List of loss values per epoch
    model_type : str
        Type of model ('AE_binary' or 'VAE_binary')
    save_path : str or None
        Path to save the plot. If None, uses default directory
    save_losses : bool
        Whether to save loss values to a file
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
    # Use default directory if save_path not provided
    if save_path is None:
        base_dir = os.path.join(SCRIPT_DIR, 'results', 'training')
    else:
        base_dir = os.path.dirname(save_path)
    
    os.makedirs(base_dir, exist_ok=True)
    
    plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title(f'{model_type}: Training Loss (Binary Cross-Entropy) over Epochs')
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
            'use_conv': use_conv,
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


def plot_zinb_test_results(all_originals, all_reconstructions_mu, 
                           all_theta=None, all_pi=None, all_raw_counts=None,
                           model_type='ZINBAE', save_dir=None, 
                           n_examples=5, metrics=None, use_conv=False, name=""):
    """
    Create comprehensive visualizations specifically for ZINB models (ZINBAE/ZINBVAE).
    
    Parameters:
    -----------
    all_originals : np.ndarray
        Original normalized log counts (shape: [n_samples, seq_length])
    all_reconstructions_mu : np.ndarray
        Reconstructed mean parameters μ (shape: [n_samples, seq_length])
    all_theta : np.ndarray or None
        Dispersion parameters θ (shape: [n_samples, seq_length])
    all_pi : np.ndarray or None
        Zero-inflation probabilities π (shape: [n_samples, seq_length])
    all_raw_counts : np.ndarray or None
        Original raw count data before normalization (shape: [n_samples, seq_length])
    model_type : str
        Type of model ('ZINBAE' or 'ZINBVAE')
    save_dir : str or None
        Directory to save plots. If None, uses default 'AE/results/testing'
    n_examples : int
        Number of example reconstructions to plot
    metrics : dict
        Dictionary of metrics to save
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    # Use default save directory if not provided
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, 'results', 'testing')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
    # Calculate basic metrics
    mae = mean_absolute_error(all_originals.flatten(), all_reconstructions_mu.flatten())
    r2 = r2_score(all_originals.flatten(), all_reconstructions_mu.flatten())
    
    # =================================================================================
    # 1. ZINB Parameter Distributions (θ and π)
    # =================================================================================
    if all_theta is not None and all_pi is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Theta distribution
        theta_flat = all_theta.flatten()
        axes[0, 0].hist(theta_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Dispersion (θ)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{model_type}: Distribution of Dispersion Parameter θ')
        axes[0, 0].axvline(x=np.median(theta_flat), color='red', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(theta_flat):.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pi distribution
        pi_flat = all_pi.flatten()
        axes[0, 1].hist(pi_flat, bins=100, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Zero-inflation Probability (π)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{model_type}: Distribution of Zero-inflation π')
        axes[0, 1].axvline(x=np.median(pi_flat), color='red', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(pi_flat):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log Theta distribution (to see small values better)
        axes[1, 0].hist(np.log10(theta_flat + 1e-8), bins=100, alpha=0.7, 
                       color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('log₁₀(θ)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_type}: Distribution of log₁₀(Dispersion θ)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mu distribution
        mu_flat = all_reconstructions_mu.flatten()
        axes[1, 1].hist(mu_flat, bins=100, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Mean (μ)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'{model_type}: Distribution of Mean Parameter μ')
        axes[1, 1].axvline(x=np.median(mu_flat), color='red', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(mu_flat):.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_zinb_parameter_distributions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================================================================================
    # 2. Actual vs Predicted with Density Plot
    # =================================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter plot (sampled for visibility)
    sample_indices = np.arange(0, all_originals.size, 10)
    orig_flat = all_originals.flatten()
    recon_flat = all_reconstructions_mu.flatten()
    
    axes[0].scatter(orig_flat[sample_indices], recon_flat[sample_indices], 
                   alpha=0.3, s=1, c='blue')
    axes[0].plot([orig_flat.min(), orig_flat.max()], 
                [orig_flat.min(), orig_flat.max()], 
                'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual (Normalized Log Counts)')
    axes[0].set_ylabel('Predicted Mean (μ)')
    axes[0].set_title(f'{model_type}: Actual vs Predicted μ\n(R²={r2:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Hexbin density plot for better visualization of dense regions
    axes[1].hexbin(orig_flat[sample_indices], recon_flat[sample_indices], 
                  gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[1].plot([orig_flat.min(), orig_flat.max()], 
                [orig_flat.min(), orig_flat.max()], 
                'b--', lw=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual (Normalized Log Counts)')
    axes[1].set_ylabel('Predicted Mean (μ)')
    axes[1].set_title(f'{model_type}: Density Plot')
    axes[1].legend()
    
    # Residuals
    residuals = orig_flat - recon_flat
    axes[2].hist(residuals[sample_indices], bins=100, alpha=0.7, 
                color='purple', edgecolor='black')
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2].axvline(x=np.median(residuals), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(residuals):.4f}')
    axes[2].set_xlabel('Residuals (Actual - Predicted)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'{model_type}: Residual Distribution\n(MAE={mae:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_prediction_quality.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================================================================================
    # 3. Zero-Inflation Analysis (if π available)
    # =================================================================================
    if all_pi is not None and all_raw_counts is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Identify actual zeros in raw counts
        actual_zeros = (all_raw_counts.flatten() == 0)
        actual_nonzeros = ~actual_zeros
        pi_flat = all_pi.flatten()
        
        # π distribution for actual zeros vs non-zeros
        axes[0, 0].hist(pi_flat[actual_zeros], bins=50, alpha=0.6, 
                       label='Actual Zeros', color='blue', edgecolor='black')
        axes[0, 0].hist(pi_flat[actual_nonzeros], bins=50, alpha=0.6, 
                       label='Actual Non-zeros', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Zero-inflation Probability (π)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'{model_type}: π Distribution by Actual Zeros')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Boxplot comparison
        axes[0, 1].boxplot([pi_flat[actual_zeros], pi_flat[actual_nonzeros]], 
                          labels=['Actual Zeros', 'Actual Non-zeros'])
        axes[0, 1].set_ylabel('Zero-inflation Probability (π)')
        axes[0, 1].set_title(f'{model_type}: π Boxplot by Actual Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Relationship between π and μ for zeros vs non-zeros
        sample_idx = np.random.choice(len(pi_flat), size=min(5000, len(pi_flat)), replace=False)
        zeros_sample = actual_zeros[sample_idx]
        
        axes[1, 0].scatter(all_reconstructions_mu.flatten()[sample_idx][zeros_sample], 
                          pi_flat[sample_idx][zeros_sample],
                          alpha=0.3, s=5, label='Actual Zeros', color='blue')
        axes[1, 0].scatter(all_reconstructions_mu.flatten()[sample_idx][~zeros_sample], 
                          pi_flat[sample_idx][~zeros_sample],
                          alpha=0.3, s=5, label='Actual Non-zeros', color='red')
        axes[1, 0].set_xlabel('Predicted Mean (μ)')
        axes[1, 0].set_ylabel('Zero-inflation Probability (π)')
        axes[1, 0].set_title(f'{model_type}: π vs μ Relationship')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Zero prediction accuracy
        # Predict zero if π > 0.5
        predicted_zeros = pi_flat > 0.5
        zero_accuracy = np.mean(predicted_zeros == actual_zeros)
        
        # Confusion matrix for zero prediction
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual_zeros, predicted_zeros)
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set(xticks=np.arange(cm.shape[1]),
                      yticks=np.arange(cm.shape[0]),
                      xticklabels=['Non-zero', 'Zero'],
                      yticklabels=['Non-zero', 'Zero'],
                      ylabel='Actual',
                      xlabel='Predicted (π > 0.5)')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black",
                              fontsize=14)
        axes[1, 1].set_title(f'{model_type}: Zero Prediction\n(Accuracy={zero_accuracy:.4f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_zero_inflation_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================================================================================
    # 4. Example Reconstructions with ZINB Parameters
    # =================================================================================
    fig, axes = plt.subplots(n_examples, 1, figsize=(15, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for i in range(min(n_examples, len(all_originals))):
        ax = axes[i]
        positions = np.arange(len(all_originals[i]))
        
        # Plot original and reconstructed μ
        ax.plot(positions, all_originals[i], label='Actual (Normalized)', 
               linewidth=2, alpha=0.8, color='blue')
        ax.plot(positions, all_reconstructions_mu[i], label='Predicted μ', 
               linewidth=2, alpha=0.8, color='red', linestyle='--')
        
        # If we have π, show where model predicts zeros (π > 0.5)
        if all_pi is not None:
            zero_pred_mask = all_pi[i] > 0.5
            if np.any(zero_pred_mask):
                ax.scatter(positions[zero_pred_mask], 
                          all_reconstructions_mu[i][zero_pred_mask],
                          marker='x', s=50, color='orange', 
                          label='Predicted Zero (π>0.5)', zorder=5)
        
        # If we have raw counts, mark actual zeros
        if all_raw_counts is not None:
            actual_zero_mask = all_raw_counts[i] == 0
            if np.any(actual_zero_mask):
                ax.scatter(positions[actual_zero_mask], 
                          all_originals[i][actual_zero_mask],
                          marker='o', s=30, color='green', alpha=0.5,
                          label='Actual Zero', zorder=4)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title(f'{model_type}: Example {i+1} - Reconstruction')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_example_reconstructions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================================================================================
    # 5. ZINB Parameters Heatmaps (for first few examples)
    # =================================================================================
    if all_theta is not None and all_pi is not None:
        n_heatmap_examples = min(3, len(all_originals))
        fig, axes = plt.subplots(n_heatmap_examples, 3, figsize=(18, 4*n_heatmap_examples))
        if n_heatmap_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_heatmap_examples):
            # μ heatmap
            im0 = axes[i, 0].imshow(all_reconstructions_mu[i:i+1], aspect='auto', 
                                   cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f'Example {i+1}: Mean (μ)')
            axes[i, 0].set_ylabel('Sample')
            axes[i, 0].set_xlabel('Position')
            plt.colorbar(im0, ax=axes[i, 0])
            
            # θ heatmap
            im1 = axes[i, 1].imshow(all_theta[i:i+1], aspect='auto', 
                                   cmap='plasma', interpolation='nearest')
            axes[i, 1].set_title(f'Example {i+1}: Dispersion (θ)')
            axes[i, 1].set_ylabel('Sample')
            axes[i, 1].set_xlabel('Position')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # π heatmap
            im2 = axes[i, 2].imshow(all_pi[i:i+1], aspect='auto', 
                                   cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Example {i+1}: Zero-inflation (π)')
            axes[i, 2].set_ylabel('Sample')
            axes[i, 2].set_xlabel('Position')
            plt.colorbar(im2, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_parameter_heatmaps.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================================================================================
    # 6. Metrics Summary
    # =================================================================================
    if metrics is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_names.append(key.upper().replace('_', ' '))
                metric_values.append(value)
        
        bars = ax.bar(metric_names, metric_values, color='steelblue', 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.6f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Value')
        ax.set_title(f'{model_type}: Test Metrics Summary')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_metrics_summary.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================================================================================
    # 7. Save comprehensive metrics to JSON
    # =================================================================================
    if metrics is not None:
        metrics_to_save = {
            'model_type': model_type,
            'timestamp': timestamp,
            'use_conv': use_conv,
            'metrics': metrics,
            'summary_statistics': {
                'original': {
                    'mean': float(np.mean(all_originals)),
                    'std': float(np.std(all_originals)),
                    'min': float(np.min(all_originals)),
                    'max': float(np.max(all_originals)),
                    'median': float(np.median(all_originals))
                },
                'predicted_mu': {
                    'mean': float(np.mean(all_reconstructions_mu)),
                    'std': float(np.std(all_reconstructions_mu)),
                    'min': float(np.min(all_reconstructions_mu)),
                    'max': float(np.max(all_reconstructions_mu)),
                    'median': float(np.median(all_reconstructions_mu))
                },
                'residuals': {
                    'mean': float(np.mean(orig_flat - recon_flat)),
                    'std': float(np.std(orig_flat - recon_flat)),
                    'median': float(np.median(orig_flat - recon_flat))
                }
            }
        }
        
        # Add ZINB-specific statistics
        if all_theta is not None:
            metrics_to_save['zinb_parameters'] = {
                'theta': {
                    'mean': float(np.mean(all_theta)),
                    'std': float(np.std(all_theta)),
                    'min': float(np.min(all_theta)),
                    'max': float(np.max(all_theta)),
                    'median': float(np.median(all_theta))
                }
            }
        
        if all_pi is not None:
            metrics_to_save['zinb_parameters'] = metrics_to_save.get('zinb_parameters', {})
            metrics_to_save['zinb_parameters']['pi'] = {
                'mean': float(np.mean(all_pi)),
                'std': float(np.std(all_pi)),
                'min': float(np.min(all_pi)),
                'max': float(np.max(all_pi)),
                'median': float(np.median(all_pi))
            }
        
        # Add zero-inflation statistics
        if all_pi is not None and all_raw_counts is not None:
            actual_zeros = (all_raw_counts.flatten() == 0)
            actual_nonzeros = ~actual_zeros
            pi_flat = all_pi.flatten()
            
            metrics_to_save['zero_inflation_analysis'] = {
                'percent_actual_zeros': float(np.mean(actual_zeros) * 100),
                'mean_pi_when_zero': float(np.mean(pi_flat[actual_zeros])),
                'mean_pi_when_nonzero': float(np.mean(pi_flat[actual_nonzeros])),
                'zero_prediction_accuracy': float(np.mean((pi_flat > 0.5) == actual_zeros))
            }
        
        metrics_file = os.path.join(save_dir, f'{prefix}_test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"ZINB metrics saved to {metrics_file}")
    
    print(f"ZINB-specific plots saved to {save_dir}/")


def plot_zinb_training_loss(losses, recon_losses=None, kl_losses=None, 
                            model_type='ZINBAE', save_path=None, 
                            save_losses=True, use_conv=False, name=""):
    """
    Plot training loss over epochs for ZINB models, with optional separate 
    reconstruction and KL loss plots for ZINBVAE.
    
    Parameters:
    -----------
    losses : list
        List of total loss values per epoch
    recon_losses : list or None
        List of reconstruction loss values per epoch (for ZINBVAE)
    kl_losses : list or None
        List of KL divergence values per epoch (for ZINBVAE)
    model_type : str
        Type of model ('ZINBAE' or 'ZINBVAE')
    save_path : str or None
        Path to save the plot. If None, uses default directory
    save_losses : bool
        Whether to save loss values to a file
    use_conv : bool
        Whether Conv1D was used in the model
    name : str
        Name prefix for saving plots/results
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_suffix = "conv" if use_conv else "no_conv"
    prefix = f"{name}_{model_type}_{timestamp}_{conv_suffix}" if name else f"{model_type}_{timestamp}_{conv_suffix}"
    
    # Use default directory if save_path not provided
    if save_path is None:
        base_dir = os.path.join(SCRIPT_DIR, 'results', 'training')
    else:
        base_dir = os.path.dirname(save_path)
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Determine if this is ZINBVAE (has KL loss)
    is_zinbvae = recon_losses is not None and kl_losses is not None
    
    if is_zinbvae:
        # Create 3-panel plot for ZINBVAE
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Total loss
        axes[0].plot(range(1, len(losses) + 1), losses, marker='o', 
                    linewidth=2, color='purple', label='Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_type}: Total Training Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Reconstruction loss (ZINB NLL)
        axes[1].plot(range(1, len(recon_losses) + 1), recon_losses, marker='s', 
                    linewidth=2, color='blue', label='Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ZINB NLL')
        axes[1].set_title(f'{model_type}: Reconstruction Loss (ZINB NLL)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # KL divergence
        axes[2].plot(range(1, len(kl_losses) + 1), kl_losses, marker='^', 
                    linewidth=2, color='red', label='KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title(f'{model_type}: KL Divergence')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(base_dir, f'{prefix}_training_losses.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a combined plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(range(1, len(losses) + 1), losses, marker='o', 
               linewidth=2, color='purple', label='Total Loss')
        ax.plot(range(1, len(recon_losses) + 1), recon_losses, marker='s', 
               linewidth=2, color='blue', alpha=0.7, label='Recon Loss')
        ax.plot(range(1, len(kl_losses) + 1), kl_losses, marker='^', 
               linewidth=2, color='red', alpha=0.7, label='KL Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_type}: Training Losses over Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        combined_plot_path = os.path.join(base_dir, f'{prefix}_training_losses_combined.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training loss plots saved to {base_dir}/")
    else:
        # Simple plot for ZINBAE
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (ZINB NLL)')
        plt.title(f'{model_type}: Training Loss (ZINB NLL) over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training loss plot saved to {plot_path}")
    
    # Save loss values to JSON
    if save_losses:
        loss_data = {
            'model_type': model_type,
            'timestamp': timestamp,
            'use_conv': use_conv,
            'loss_type': 'ZINB_NLL',
            'num_epochs': len(losses),
            'final_loss': float(losses[-1]),
            'min_loss': float(min(losses)),
            'max_loss': float(max(losses)),
            'losses_per_epoch': [float(loss) for loss in losses]
        }
        
        if is_zinbvae:
            loss_data['final_recon_loss'] = float(recon_losses[-1])
            loss_data['final_kl_loss'] = float(kl_losses[-1])
            loss_data['min_recon_loss'] = float(min(recon_losses))
            loss_data['min_kl_loss'] = float(min(kl_losses))
            loss_data['recon_losses_per_epoch'] = [float(loss) for loss in recon_losses]
            loss_data['kl_losses_per_epoch'] = [float(loss) for loss in kl_losses]
        
        loss_file = os.path.join(base_dir, f'{prefix}_training_losses.json')
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=4)
        print(f"Training losses saved to {loss_file}")