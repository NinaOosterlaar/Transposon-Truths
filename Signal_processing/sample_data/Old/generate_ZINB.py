import numpy as np
import pandas as pd
import os


def generate_zinb_sample(n, mu, theta, pi):
    """
    Generate a sample from a Zero-Inflated Negative Binomial distribution.
    
    Parameters:
    -----------
    n : int
        Sample size
    mu : float
        Mean parameter of the NB distribution
    theta : float
        Dispersion parameter of the NB distribution
    pi : float
        Zero-inflation probability (0 <= pi < 1)
    
    Returns:
    --------
    numpy.ndarray : Array of ZINB samples
    """
    # Generate zero-inflation indicators
    zero_inflation = np.random.binomial(1, pi, size=n)
    
    # Generate NB samples
    # NB parameterization: p = theta / (theta + mu)
    p = theta / (theta + mu)
    nb_samples = np.random.negative_binomial(theta, p, size=n)
    
    # Combine: if zero_inflation[i] == 1, use 0, otherwise use nb_samples[i]
    zinb_samples = np.where(zero_inflation == 1, 0, nb_samples)
    
    return zinb_samples


def generate_zinb_datasets(sample_size=1000, output_dir='ZINB'):
    """
    Generate ZINB datasets with various parameter combinations and save to files.
    
    Parameters:
    -----------
    sample_size : int
        Number of samples per dataset
    output_dir : str
        Directory to save the generated datasets (relative to script location)
    """
    # Define parameter ranges
    pi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    mu_values = [1, 2, 3, 4, 5, 10, 20, 50]
    theta_values = [0.01, 0.1, 1, 5, 10, 50, 75, 100, 200]
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Create Data subdirectory
    data_dir = os.path.join(full_output_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Store all parameter combinations and their results
    all_results = []
    
    # Generate datasets for each parameter combination
    for pi in pi_values:
        for mu in mu_values:
            for theta in theta_values:
                # Generate ZINB data
                data = generate_zinb_sample(sample_size, mu, theta, pi)
                
                # Create filename
                filename = f"zinb_pi{pi:.1f}_mu{mu}_theta{theta}.csv"
                # Save to Data subdirectory
                filepath = os.path.join(data_dir, filename)
                
                # Save data to CSV
                df = pd.DataFrame({'count': data})
                df.to_csv(filepath, index=False)
                
                
                all_results.append({
                    'filename': filename,
                    'pi': pi,
                    'mu': mu,
                    'theta': theta,
                    'sample_size': sample_size,
                })
                
                print(f"Generated: {filename}")
    
    # Save parameter summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(full_output_dir, 'dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Total datasets generated: {len(all_results)}")


if __name__ == "__main__":
    generate_zinb_datasets(sample_size=1000)
