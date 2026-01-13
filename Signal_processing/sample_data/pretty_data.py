import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Utils.plot_config import setup_plot_style, COLORS

# Set up standardized plot style
setup_plot_style()
import pandas as pd


def make_random_gaussian_counts(
    n_regions,
    mean_range=(0, 25),          
    length_mean_range=(40, 1000), 
    sd_scale=(0.5, 1.5),                
    min_sd=0.5,                  
    min_region_length=20,
    probability_zero_region=0.1,
):
    """
    Generate synthetic count data composed of multiple regions, each with counts drawn from a Gaussian distribution.
    Args:
        n_regions (int): Number of distinct regions to generate.
        mean_range (tuple): Min and max mean values for the Gaussian distributions of each region.
        length_mean_range (tuple): Min and max mean lengths for each region.
        sd_scale (float): Scaling factor for the standard deviation of counts relative to the mean.
        min_sd (float): Minimum standard deviation for counts.
        min_region_length (int): Minimum length for each region.
        probability_zero_region (float): Probability that a region has a mean of zero.
    Returns:
        positions (np.ndarray): Array of positions corresponding to the counts.
        counts (np.ndarray): Array of generated counts.
        region_boundaries (list): List of end positions for each region.
        region_params (dict): Dictionary containing parameters used for each region.

    """
    
    region_means = np.random.uniform(*mean_range, size=n_regions)
    zero_region_mask = np.random.rand(n_regions) < probability_zero_region
    region_means[zero_region_mask] = 0
    region_lengths = np.random.uniform(*length_mean_range, size=n_regions)
    region_sds = np.random.uniform(*sd_scale, size=n_regions) * np.sqrt(region_means) 
    region_sds = np.maximum(region_sds, min_sd)
    region_params = {
        "region_means": region_means,
        "region_sds": region_sds,
        "region_lengths": region_lengths,
    }

    counts = []
    region_boundaries = []
    pos = 0

    for n in range(n_regions):
        L = int(np.round(region_lengths[n]))
        L = max(L, min_region_length)

        region_vals = np.random.normal(
            loc=region_means[n],
            scale=region_sds[n],
            size=L,
        )
        region_vals = np.clip(region_vals, 0, None)
        region_vals = np.rint(region_vals).astype(int)

        counts.append(region_vals)
        pos += L
        region_boundaries.append(pos - 1)

    counts = np.concatenate(counts)

    return counts, region_boundaries, region_params


if __name__ == "__main__":
    plot = False
    generate = True
    
    if generate:
        number_of_regions = 2000
        output_path = f"Signal_processing/pretty_data.csv"
        counts, boundaries, params = make_random_gaussian_counts(
            n_regions=number_of_regions,
            mean_range=(1, 15),
            length_mean_range=(40, 1000),
            sd_scale=(0.5, 1.5),
            min_sd=0.3,
        )
        positions = np.arange(len(counts))
        df = pd.DataFrame({"Position": positions, "Count": counts})
        df.to_csv(output_path, index=False)
        # Save params
        params_output_path = f"Signal_processing/pretty_data_params.csv"
        params_df = pd.DataFrame(params)
        params_df.to_csv(params_output_path, index=False)
    
    if plot:
        counts, boundaries, params = make_random_gaussian_counts(
            n_regions=5,
            mean_range=(1, 15),
            length_mean_range=(40, 1000),
            sd_scale=(0.5, 1.5),
        min_sd=0.3,
    )
        positions = np.arange(len(counts))
        plt.figure(figsize=(12, 3))
        plt.bar(positions, counts, width=1.0, color=COLORS['black'])
        for b in boundaries[:-1]:
            plt.axvline(b + 0.5, linestyle="--", color=COLORS['red'], linewidth=1)
        plt.xlabel("Position")
        plt.ylabel("Count")
        plt.title("Randomly Generated Gaussian Counts")
        plt.tight_layout()
        plt.show()

        print("Region means:", params["region_means"])
        print("Region SDs:", params["region_sds"])
        
    
