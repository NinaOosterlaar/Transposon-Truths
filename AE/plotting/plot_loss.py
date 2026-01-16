import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS
from AE.plotting.plot_helper import generate_prefix, prepare_output_dirs
import matplotlib.pyplot as plt
import json

# Set up standardized plot style
setup_plot_style()


def plot_training_loss(losses, model_type='AE', save_path=None, 
                       save_losses=True, use_conv=False, name="", reg_losses=None):
    """
    Plot training loss over epochs for continuous models (AE, VAE).
    
    Inputs from training.py:
    - From train() function: epoch_losses list, model attributes (model_type, use_conv), name parameter
    
    Parameters:
    -----------
    losses : list or np.ndarray
        Loss values per epoch
    model_type : str
        Model type ('AE' or 'VAE'). Default='AE'
    save_path : str or None
        Path to save plot. If None, uses default 'AE/results/training'
    save_losses : bool
        Whether to save loss values to JSON. Default=True
    use_conv : bool
        Whether Conv1D was used in model. Default=False
    name : str
        Name prefix for saved files. Default=""
    reg_losses : list or np.ndarray or None
        Regularization loss values per epoch. Default=None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = generate_prefix(model_type, timestamp, use_conv, name)
    
    base_dir = prepare_output_dirs(save_path, subdir='training', name=name)
    
    # Create multi-panel plot if regularization is used
    if reg_losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color=COLORS['black'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'{model_type}: Training Loss over Epochs')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(range(1, len(reg_losses) + 1), reg_losses, marker='s', linewidth=2, color=COLORS['red'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Regularization Loss')
        axes[1].set_title(f'{model_type}: Regularization Loss over Epochs')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(base_dir, f'{prefix}_training_losses.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{model_type}: Training Loss over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Training loss plot saved to {plot_path}")
    
    if save_losses:
        loss_data = {
            'model_type': model_type,
            'timestamp': timestamp,
            'use_conv': use_conv,
            'loss_type': 'MSE',
            'num_epochs': len(losses),
            'final_loss': float(losses[-1]),
            'min_loss': float(min(losses)),
            'max_loss': float(max(losses)),
            'losses_per_epoch': [float(loss) for loss in losses]
        }
        
        if reg_losses is not None:
            loss_data['has_regularization'] = True
            loss_data['final_reg_loss'] = float(reg_losses[-1])
            loss_data['min_reg_loss'] = float(min(reg_losses))
            loss_data['reg_losses_per_epoch'] = [float(loss) for loss in reg_losses]
        else:
            loss_data['has_regularization'] = False
        
        loss_file = os.path.join(base_dir, f'{prefix}_training_losses.json')
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=4)
        print(f"Training losses saved to {loss_file}")


def plot_binary_training_loss(losses, model_type='AE_binary', 
                              save_path=None, 
                              save_losses=True, use_conv=False, name="", reg_losses=None):
    """
    Plot training loss over epochs for binary models (AE_binary, VAE_binary).
    
    Inputs from training.py:
    - From train() function: epoch_losses list, model attributes (model_type, use_conv), name parameter
    
    Parameters:
    -----------
    losses : list or np.ndarray
        Loss values per epoch (Binary Cross-Entropy)
    model_type : str
        Model type ('AE_binary' or 'VAE_binary'). Default='AE_binary'
    save_path : str or None
        Path to save plot. If None, uses default directory
    save_losses : bool
        Whether to save loss values to JSON. Default=True
    use_conv : bool
        Whether Conv1D was used. Default=False
    name : str
        Name prefix for saved files. Default=""
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = generate_prefix(model_type, timestamp, use_conv, name)
    base_dir = prepare_output_dirs(save_path, subdir='training', name=name)
    plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color=COLORS['pink'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title(f'{model_type}: Training Loss (Binary Cross-Entropy) over Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {plot_path}")
    
    if save_losses:
        loss_data = {
            'model_type': model_type,
            'timestamp': timestamp,
            'use_conv': use_conv,
            'loss_type': 'BCE',
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


def plot_zinb_training_loss(losses, recon_losses=None, kl_losses=None, 
                            model_type='ZINBAE', save_path=None, 
                           save_losses=True, use_conv=False, name="", reg_losses=None):
    """
    Plot training loss over epochs for ZINB models (ZINBAE, ZINBVAE).
    
    For ZINBVAE: Shows total loss, reconstruction loss (ZINB NLL), and KL divergence separately.
    For ZINBAE: Shows only reconstruction loss (ZINB NLL).
    
    Inputs from training.py:
    - ZINBAE: epoch_losses (ZINB NLL only)
    - ZINBVAE: epoch_losses (total), epoch_recon_losses, epoch_kl_losses
    
    Parameters:
    -----------
    losses : list or np.ndarray
        Total loss values per epoch
    recon_losses : list or np.ndarray or None
        Reconstruction loss values per epoch (ZINB NLL for ZINBVAE)
    kl_losses : list or np.ndarray or None
        KL divergence values per epoch (for ZINBVAE)
    model_type : str
        Model type ('ZINBAE' or 'ZINBVAE'). Default='ZINBAE'
    save_path : str or None
        Path to save plot. If None, uses default directory
    save_losses : bool
        Whether to save loss values to JSON. Default=True
    use_conv : bool
        Whether Conv1D was used. Default=False
    name : str
        Name prefix for saved files. Default=""
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = generate_prefix(model_type, timestamp, use_conv, name)
    base_dir = prepare_output_dirs(save_path, subdir='training', name=name)
    
    is_zinbvae = recon_losses is not None and kl_losses is not None
    
    if is_zinbvae:
        # Create 3 or 4-panel plot for ZINBVAE (add reg panel if regularization is used)
        num_panels = 4 if reg_losses is not None else 3
        fig, axes = plt.subplots(1, num_panels, figsize=(6*num_panels, 5))
        
        axes[0].plot(range(1, len(losses) + 1), losses, marker='o', 
                    linewidth=2, color=COLORS['pink'], label='Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_type}: Total Training Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(range(1, len(recon_losses) + 1), recon_losses, marker='s', 
                    linewidth=2, color=COLORS['blue'], label='Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ZINB NLL')
        axes[1].set_title(f'{model_type}: Reconstruction Loss (ZINB NLL)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(range(1, len(kl_losses) + 1), kl_losses, marker='^', 
                    linewidth=2, color=COLORS['red'], label='KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title(f'{model_type}: KL Divergence')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        if reg_losses is not None:
            axes[3].plot(range(1, len(reg_losses) + 1), reg_losses, marker='d', 
                        linewidth=2, color=COLORS['orange'], label='Regularization')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Regularization Loss')
            axes[3].set_title(f'{model_type}: Regularization Loss')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(base_dir, f'{prefix}_training_losses.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a combined plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(range(1, len(losses) + 1), losses, marker='o', 
               linewidth=2, color=COLORS['pink'], label='Total Loss')
        ax.plot(range(1, len(recon_losses) + 1), recon_losses, marker='s', 
               linewidth=2, color=COLORS['blue'], alpha=0.7, label='Recon Loss')
        ax.plot(range(1, len(kl_losses) + 1), kl_losses, marker='^', 
               linewidth=2, color=COLORS['red'], alpha=0.7, label='KL Loss')
        if reg_losses is not None:
            ax.plot(range(1, len(reg_losses) + 1), reg_losses, marker='d', 
                   linewidth=2, color=COLORS['orange'], alpha=0.7, label='Reg Loss')
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
        # ZINBAE: recon_losses contains NLL, losses contains total (NLL + reg if applicable)
        # If we have recon_losses, show total vs NLL vs reg
        # If no recon_losses, it's the old behavior (just total loss)
        
        if recon_losses is not None:
            # ZINBAE with separate NLL tracking
            if reg_losses is not None:
                # Show 3 panels: Total, NLL, Regularization
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                axes[0].plot(range(1, len(losses) + 1), losses, marker='o', 
                            linewidth=2, color=COLORS['pink'], label='Total Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Total Loss')
                axes[0].set_title(f'{model_type}: Total Training Loss')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                axes[1].plot(range(1, len(recon_losses) + 1), recon_losses, marker='s', 
                            linewidth=2, color=COLORS['blue'], label='ZINB NLL')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('ZINB NLL')
                axes[1].set_title(f'{model_type}: Reconstruction Loss (ZINB NLL)')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                axes[2].plot(range(1, len(reg_losses) + 1), reg_losses, marker='d', 
                            linewidth=2, color=COLORS['red'], label='Regularization')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Regularization Loss')
                axes[2].set_title(f'{model_type}: Regularization Loss')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
                
                plt.tight_layout()
                plot_path = os.path.join(base_dir, f'{prefix}_training_losses.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Show just NLL (no regularization)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(recon_losses) + 1), recon_losses, marker='o', 
                        linewidth=2, color=COLORS['blue'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss (ZINB NLL)')
                plt.title(f'{model_type}: Training Loss (ZINB NLL) over Epochs')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        else:
            # Old behavior for backward compatibility (if recon_losses not provided)
            if reg_losses is not None:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                axes[0].plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color=COLORS['black'])
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss (ZINB NLL)')
                axes[0].set_title(f'{model_type}: Training Loss (ZINB NLL) over Epochs')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(range(1, len(reg_losses) + 1), reg_losses, marker='s', linewidth=2, color=COLORS['red'])
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Regularization Loss')
                axes[1].set_title(f'{model_type}: Regularization Loss over Epochs')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(base_dir, f'{prefix}_training_losses.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, color=COLORS['black'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss (ZINB NLL)')
                plt.title(f'{model_type}: Training Loss (ZINB NLL) over Epochs')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = os.path.join(base_dir, f'{prefix}_training_loss.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Training loss plot saved to {plot_path}")
    
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
        
        if reg_losses is not None:
            loss_data['has_regularization'] = True
            loss_data['final_reg_loss'] = float(reg_losses[-1])
            loss_data['min_reg_loss'] = float(min(reg_losses))
            loss_data['reg_losses_per_epoch'] = [float(loss) for loss in reg_losses]
        else:
            loss_data['has_regularization'] = False
        
        loss_file = os.path.join(base_dir, f'{prefix}_training_losses.json')
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=4)
        print(f"Training losses saved to {loss_file}")