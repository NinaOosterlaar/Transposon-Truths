import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ZINB_MLE.estimate_ZINB import estimate_zinb


def process_all_strains(window_size=2000):
    """
    Process all CSV files in Data/combined_strains/, split into windows,
    estimate ZINB parameters for each window, and save results with window locations.
    
    Parameters:
    -----------
    window_size : int
        Size of windows to split each chromosome into (default: 2000)
    """
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "Data" / "combined_strains"
    output_dir = base_dir / "Signal_processing" / "results" / "ZINB_estimates"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results list
    results = []
    
    # Get all strain folders
    strain_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    print(f"Found {len(strain_folders)} strain folders")
    print(f"Window size: {window_size}")
    
    # Process each strain
    for strain_folder in strain_folders:
        strain_name = strain_folder.name
        print(f"\nProcessing {strain_name}...")
        
        # Get all CSV files in this strain folder
        csv_files = sorted(strain_folder.glob("*.csv"))
        
        total_windows = 0
        
        for csv_file in csv_files:
            chromosome = csv_file.stem.replace("_distances", "")
            print(f"  Processing {chromosome}...")
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Extract Position and Value columns
                positions = df['Position'].values
                values = df['Value'].values
                rounded_values = np.round(values).astype(int)
                
                # Split into windows
                n_positions = len(rounded_values)
                n_windows = int(np.ceil(n_positions / window_size))
                
                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = min((i + 1) * window_size, n_positions)
                    window_data = rounded_values[start_idx:end_idx]
                    window_positions = positions[start_idx:end_idx]
                    
                    # Get actual start and end positions
                    start_pos = int(window_positions[0])
                    end_pos = int(window_positions[-1])
                    
                    # Check if more than 95% are zeros
                    zero_fraction = np.sum(window_data == 0) / len(window_data)
                    if zero_fraction > 0.95:
                        print(f"    Window {i+1}: pos {start_pos}-{end_pos} | "
                              f"SKIPPED (>95% zeros: {zero_fraction:.2%})")
                        continue
                    
                    # Exclude top 1% highest counts
                    percentile_99 = np.percentile(window_data, 99)
                    filtered_data = window_data[window_data <= percentile_99]
                    
                    # Check if we have enough data left
                    if len(filtered_data) < 10:
                        print(f"    Window {i+1}: pos {start_pos}-{end_pos} | "
                              f"SKIPPED (insufficient data after filtering)")
                        continue
                    
                    # Estimate ZINB parameters for this window
                    try:
                        estimates = estimate_zinb(filtered_data, max_iter=1000)
                        
                        # Store results
                        result = {
                            'strain': strain_name,
                            'chromosome': chromosome,
                            'window_id': i + 1,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'pi': estimates['pi'],
                            'mu': estimates['mu'],
                            'theta': estimates['theta'],
                            'log_likelihood': estimates['log_likelihood'],
                            'iterations': estimates['iterations'],
                            'converged': estimates['converged'],
                            'n_observations': len(window_data),
                            'n_filtered': len(filtered_data),
                            'zero_fraction': zero_fraction
                        }
                        results.append(result)
                        total_windows += 1
                        
                        # Print window location and results
                        print(f"    Window {i+1}: pos {start_pos}-{end_pos} | "
                              f"pi={estimates['pi']:.4f}, mu={estimates['mu']:.4f}, "
                              f"theta={estimates['theta']:.4f}, converged={estimates['converged']}")
                        
                    except Exception as e:
                        print(f"    ERROR estimating window {i+1}: {str(e)}")
                        continue
                
                print(f"    Processed {n_windows} windows")
                
            except Exception as e:
                print(f"    ERROR reading {chromosome}: {str(e)}")
                continue
        
        print(f"  Total windows processed for {strain_name}: {total_windows}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = output_dir / f"zinb_estimates_windows_size{window_size}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total windows processed: {len(results_df)}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(results_df[['pi', 'mu', 'theta']].describe())
    
    return results_df


if __name__ == "__main__":
    results = process_all_strains(window_size=2000)
