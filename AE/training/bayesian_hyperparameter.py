import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
import numpy as np
import json
from datetime import datetime
import gc
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import dump, load
from AE.main import main_with_datasets
from AE.preprocessing.preprocessing import preprocess
import argparse

# Force numpy to not use memory mapping for large arrays (prevents bus errors)
os.environ['NUMPY_MMAP_MODE'] = 'c'  # Copy mode instead of mmap

# ============================================================================
# BAYESIAN OPTIMIZATION PARAMETER SPACE
# ============================================================================

# Preprocessing hyperparameters - CATEGORICAL
# Note: Encode features as strings to avoid skopt Categorical distance calculation issues
FEATURES_OPTIONS = ["Centr_Nucl", "Centr", "Nucl"]  # Will be decoded to lists later
MOVING_AVERAGE_OPTIONS = [True, False]

# Convolutional Layer hyperparameters
USE_CONV_OPTIONS = [True, False]
# Padding is fixed to 0 (valid padding) to avoid dimension mismatch issues
FIXED_PADDING = 0

# Regularization hyperparameters
REGULARIZATIONS = ["l1", "l2", "none"]

# ============================================================================
# BAYESIAN OPTIMIZATION SEARCH SPACE
# ============================================================================
search_space = [
    # Preprocessing
    Categorical(FEATURES_OPTIONS, name='features'),
    Integer(500, 10000, name='data_point_length'),  # SEQUENCE_LENGTHS range
    Real(0.05, 1.0, name='step_size'),  # STEP_SIZES as continuous
    Integer(5, 100, name='bin_size'),  # BIN_SIZES range
    Real(0.25, 1.0, name='sample_fraction'),  # SAMPLE_FRACTIONS as continuous
    Categorical(MOVING_AVERAGE_OPTIONS, name='moving_average'),
    
    # Model Architecture (parameterized, layer sizes divisible by 16)
    Integer(4, 200, name='first_layer_size_factor'),  # Multiplied by 16: 64-3200
    Integer(1, 6, name='num_layers'),  # Number of layers (each divides by 2)
    
    # Convolutional Layers
    Categorical(USE_CONV_OPTIONS, name='use_conv'),
    Integer(16, 128, name='conv_channel'),  # CONV_CHANNELS range (powers of 2 will be sampled)
    Integer(2, 8, name='pool_size'),  # POOL_SIZES range
    Categorical([3, 5, 7, 9, 11, 13], name='kernel_size'),  # Odd numbers only for symmetry
    
    # Training
    Integer(30, 150, name='epochs'),  # EPOCHS range
    Categorical([32, 64, 128, 256], name='batch_size'),  # Powers of 2 only
    Real(0.0, 0.9, name='noise_level'),  # NOISE_LEVELS as continuous
    Real(0.3, 0.7, name='pi_threshold'),  # PI_THRESHOLD as continuous
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),  # Log scale for learning rate
    Real(0.0, 0.5, name='dropout_rate'),  # DROPOUT_RATES as continuous
    
    # Loss weights (log-scale since it can span orders of magnitude)
    Real(1e-3, 10.0, prior='log-uniform', name='masked_recon_weight'),  # gamma: weight for masked reconstruction loss
    
    # Regularization
    Categorical(REGULARIZATIONS, name='regularizer'),
    Real(1e-5, 100, prior='log-uniform', name='regularization_weight'),  # Log scale
]

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'input_folder': "Data/test/strain_FD",
    'split_on': 'Chrom',
    'train_val_test_split': [0.6, 0.2, 0.2],  # Proper train/val/test split
    'plot': False,
    'stride': 1,  # Fixed to avoid dimension mismatch issues
    'padding': 0,  # Fixed to valid padding
}

# Optimization metric: which loss to minimize from VALIDATION set
OPTIMIZATION_METRIC = 'zinb_nll'

# Budget for optimization
N_CALLS = 50  # Number of Bayesian optimization iterations 
RANDOM_STATE = 42  # For reproducibility
N_INITIAL_POINTS = 10  # Random exploration before Bayesian optimization

# Results directory
RESULTS_DIR = "AE/results/bayesian_optimization"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================
@use_named_args(search_space)
def objective(**params):
    """
    Objective function for Bayesian optimization.
    Creates datasets with trial-specific preprocessing hyperparameters,
    trains model, and returns validation loss.
    
    Returns:
        float: Validation loss metric to minimize (specified by OPTIMIZATION_METRIC)
    """
    # Merge with fixed parameters
    all_params = {**FIXED_PARAMS, **params}
    
    print(f"\n{'='*80}")
    print(f"Trial with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    try:
        # Extract preprocessing parameters for this trial
        features_str = all_params['features']
        bin_size = all_params['bin_size']
        moving_average = all_params['moving_average']
        data_point_length = all_params['data_point_length']
        step_size = all_params['step_size']
        
        # Decode features string to list
        if features_str == "Centr_Nucl":
            features = ["Centr", "Nucl"]
        elif features_str == "Centr":
            features = ["Centr"]
        elif features_str == "Nucl":
            features = ["Nucl"]
        else:
            features = [features_str]  # Fallback
        
        # Construct layers from parametric representation (divisible by 16)
        first_layer_size_factor = all_params['first_layer_size_factor']
        first_layer_size = first_layer_size_factor * 16  # Ensure divisible by 16
        num_layers = all_params['num_layers']
        layers = [first_layer_size // (2**i) for i in range(num_layers)]
        
        # Adjust data_point_length BEFORE preprocessing (if not using moving average)
        preprocessing_data_length = data_point_length
        if not moving_average:
            preprocessing_data_length = data_point_length // bin_size
        
        print(f"Creating datasets with:")
        print(f"  features: {features}")
        print(f"  bin_size: {bin_size}")
        print(f"  moving_average: {moving_average}")
        print(f"  data_point_length: {preprocessing_data_length} (from {data_point_length})")
        print(f"  step_size: {step_size}")
        print(f"  layers: {layers} (first={first_layer_size}, num={num_layers})")
        print(f"  stride: {all_params['stride']} (fixed), padding: {all_params['padding']} (fixed)\n")
        
        # Preprocess data with trial-specific parameters
        train_set, val_set, test_set, _, _, _ = preprocess(
            input_folder=all_params['input_folder'],
            features=features,
            bin_size=bin_size,
            moving_average=moving_average,
            data_point_length=preprocessing_data_length,
            step_size=step_size,
            split_on=all_params['split_on'],
            train_val_test_split=all_params['train_val_test_split']
        )
        
        # CRITICAL: Ensure arrays are in-memory copies, not memory-mapped
        # This prevents bus errors in parallel execution
        if train_set is not None and hasattr(train_set, 'flags') and not train_set.flags['OWNDATA']:
            train_set = np.array(train_set, copy=True)
        if val_set is not None and hasattr(val_set, 'flags') and not val_set.flags['OWNDATA']:
            val_set = np.array(val_set, copy=True)
        if test_set is not None and hasattr(test_set, 'flags') and not test_set.flags['OWNDATA']:
            test_set = np.array(test_set, copy=True)
        
        print(f"Datasets created:")
        print(f"  Train: {train_set.shape}, Val: {val_set.shape if val_set is not None else 'None'}, Test: {test_set.shape if test_set is not None else 'None'}\n")
        
        try:
            # Call main function with the created datasets
            # eval_on_val=True means it evaluates on validation set (not test)
            train_metrics, val_metrics = main_with_datasets(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                features=features,
                data_point_length=preprocessing_data_length,
                use_conv=all_params['use_conv'],
                conv_channel=int(all_params['conv_channel']),
                pool_size=int(all_params['pool_size']),
                kernel_size=int(all_params['kernel_size']),
                padding=int(all_params['padding']),
                stride=int(all_params['stride']),
                epochs=int(all_params['epochs']),
                batch_size=int(all_params['batch_size']),
                noise_level=all_params['noise_level'],
                pi_threshold=all_params['pi_threshold'],
                masked_recon_weight=all_params['masked_recon_weight'],
                learning_rate=all_params['learning_rate'],
                dropout_rate=all_params['dropout_rate'],
                layers=layers,  # Use converted list
                regularizer=all_params['regularizer'],
                regularization_weight=all_params['regularization_weight'],
                sample_fraction=all_params['sample_fraction'],
                plot=all_params['plot'],
                eval_on_val=True  # Use validation set for optimization
            )
        finally:
            # Explicitly delete datasets to free memory after training
            del train_set, val_set, test_set
            gc.collect()
        
        # Extract the metric to optimize from VALIDATION metrics
        if OPTIMIZATION_METRIC not in val_metrics:
            print(f"Warning: Metric '{OPTIMIZATION_METRIC}' not found in val_metrics.")
            print(f"Available metrics: {list(val_metrics.keys())}")
            # Fallback to total_loss or first available metric
            if 'total_loss' in val_metrics:
                loss = val_metrics['total_loss']
                print(f"Using 'total_loss' instead: {loss:.6f}")
            else:
                loss = float(list(val_metrics.values())[0])
                print(f"Using first metric '{list(val_metrics.keys())[0]}': {loss:.6f}")
        else:
            loss = val_metrics[OPTIMIZATION_METRIC]
        
        print(f"\n>>> Optimizing {OPTIMIZATION_METRIC} on VALIDATION set: {loss:.6f}")
        print(f">>> Full validation metrics: {val_metrics}\n")
        
        # Explicitly delete metrics to free memory
        del train_metrics, val_metrics
        
        return loss
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        print(f"Parameters that caused error: {params}\n")
        import traceback
        traceback.print_exc()
        
        # Clean up any allocated memory before returning error
        try:
            if 'train_set' in locals():
                del train_set
            if 'val_set' in locals():
                del val_set
            if 'test_set' in locals():
                del test_set
            if 'train_metrics' in locals():
                del train_metrics
            if 'val_metrics' in locals():
                del val_metrics
            gc.collect()
        except:
            pass
        
        # Return a large penalty value instead of crashing
        return 1e6


# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================
def run_bayesian_optimization(n_calls=N_CALLS, random_state=RANDOM_STATE, 
                              n_initial_points=N_INITIAL_POINTS, n_jobs=1):
    """
    Run Bayesian hyperparameter optimization using scikit-optimize.
    
    Args:
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        n_initial_points: Number of random evaluations before Gaussian Process
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        
    Returns:
        result: OptimizeResult object from skopt
    """
    print(f"\n{'#'*80}")
    print(f"# Starting Bayesian Hyperparameter Optimization")
    print(f"# Optimizing metric: {OPTIMIZATION_METRIC} on VALIDATION set")
    print(f"# Number of trials: {n_calls}")
    print(f"# Initial random points: {n_initial_points}")
    print(f"# Random state: {random_state}")
    print(f"# Parallel jobs: {n_jobs}")
    print(f"{'#'*80}\n")
    
    # Set environment variables to prevent each worker from spawning multiple threads
    # Without this, 10 workers × multiple threads each = memory explosion
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        verbose=True,
        n_jobs=n_jobs,
    )
    
    # Force cleanup of any remaining resources
    gc.collect()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"bayesian_opt_result_{timestamp}.pkl")
    dump(result, result_file)
    print(f"\nOptimization result saved to: {result_file}")
    
    # Extract best parameters
    best_params = {search_space[i].name: result.x[i] for i in range(len(search_space))}
    
    # Extract ALL trial results from the optimization result
    all_trials_data = {
        'optimization_info': {
            'optimization_metric': OPTIMIZATION_METRIC,
            'n_calls': n_calls,
            'n_initial_points': n_initial_points,
            'random_state': random_state,
            'timestamp': timestamp,
            'best_score': float(result.fun),
        },
        'best_parameters': {},
        'all_trials': []
    }
    
    # Save best parameters
    for key, value in best_params.items():
        if isinstance(value, (list, tuple)):
            all_trials_data['best_parameters'][key] = list(value)
        elif isinstance(value, (np.integer, np.int_)):
            all_trials_data['best_parameters'][key] = int(value)
        elif isinstance(value, (np.floating, np.float_)):
            all_trials_data['best_parameters'][key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            all_trials_data['best_parameters'][key] = bool(value)
        else:
            all_trials_data['best_parameters'][key] = value
    
    # Save all trial results from result object
    # Cache space names to avoid recreating list for each trial
    space_names = [space.name for space in search_space]
    
    for i, (params_list, score) in enumerate(zip(result.x_iters, result.func_vals)):
        trial_data = {
            'trial_number': i + 1,
            'score': float(score),
            'parameters': {}
        }
        
        # Convert parameters
        for j, param_name in enumerate(space_names):
            value = params_list[j]
            if isinstance(value, (list, tuple)):
                trial_data['parameters'][param_name] = list(value)
            elif isinstance(value, (np.integer, np.int_)):
                trial_data['parameters'][param_name] = int(value)
            elif isinstance(value, (np.floating, np.float_)):
                trial_data['parameters'][param_name] = float(value)
            elif isinstance(value, (np.bool_, bool)):
                trial_data['parameters'][param_name] = bool(value)
            else:
                trial_data['parameters'][param_name] = value
        
        all_trials_data['all_trials'].append(trial_data)
    
    # Save all results to single JSON file
    all_results_file = os.path.join(RESULTS_DIR, f"all_trials_{timestamp}.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_trials_data, f, indent=4)
    
    # Clear large data structure from memory
    del all_trials_data
    
    print(f"All trial results saved to: {all_results_file}")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"# Optimization Complete!")
    print(f"# Optimized metric: {OPTIMIZATION_METRIC} on VALIDATION set")
    print(f"# Best validation {OPTIMIZATION_METRIC}: {result.fun:.6f}")
    print(f"# Total trials: {len(result.x_iters)}")
    print(f"# Best parameters:")
    for key, value in best_params.items():
        print(f"#   {key}: {value}")
    print(f"{'#'*80}\n")
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization')
    parser.add_argument('--n_calls', type=int, default=N_CALLS,
                       help='Number of optimization iterations')
    parser.add_argument('--n_initial_points', type=int, default=N_INITIAL_POINTS,
                       help='Number of random initial evaluations')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (-1 for all cores, 1 for sequential)')
    
    args = parser.parse_args()
    
    # Run optimization
    result = run_bayesian_optimization(
        n_calls=args.n_calls,
        random_state=args.random_state,
        n_initial_points=args.n_initial_points,
        n_jobs=args.n_jobs
    )

