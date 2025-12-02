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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'AE/results/training/training_loss_{model_type}_{timestamp}.png'
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


def plot_binary_test_results(all_originals, all_reconstructions, all_probabilities, 
                             model_type='AE_binary', save_dir='AE/results/testing', 
                             n_examples=5, metrics=None):
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
    save_dir : str
        Directory to save plots
    n_examples : int
        Number of example reconstructions to plot
    metrics : dict
        Dictionary of metrics to save
    """
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    # Create visualization directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_type}_{timestamp}"
    
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
                              save_path='AE/results/training/training_loss.png', 
                              save_losses=True):
    """
    Plot training loss over epochs for binary models.
    
    Parameters:
    -----------
    losses : list
        List of loss values per epoch
    model_type : str
        Type of model ('AE_binary' or 'VAE_binary')
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
