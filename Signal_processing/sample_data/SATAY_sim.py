# Combined (n=7611 windows across all strains):
#   mu:    mean = 4.447154, std = 6.322336
#   theta: mean = 0.522436, std = 0.202211
# Nucleosomes:
#   Mean:   166.95 bp
#   Std:    33.77 bp
#   Median: 163.00 bp
#   Range:  [107, 297] bp
# Baselies:
# Centromere file mean of mean_density: 0.03799867644351568
# Nucleosome file mean of mean_density: 0.1791183904425311

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sample_NB(mu, theta, size):
    """Sample from a Negative Binomial distribution parameterized by mean (mu) and dispersion (theta)."""
    p = theta / (theta + mu)
    r = theta
    return np.random.negative_binomial(r, p, size=size)

def generate_NB_sample(mu_mean, mu_std, theta, size, length_range):
    """Generate a sample of regions with different mean values generated from a normal distribution.
    The length is uniform distributed, the actual sample is created using the NB distribution.
    Each region has a different mean. Also save the region boundaries and parameters for later analysis."""
    counts = []
    region_boundaries = [0]
    means = []
    while len(counts) < size:
        region_length = np.random.randint(*length_range)
        region_mu = max(0.0, np.random.normal(mu_mean, mu_std))  # Ensure mean is positive
        means.append(region_mu)
        region_counts = sample_NB(region_mu, theta, size=region_length)
        counts.extend(region_counts)
        region_boundaries.append(region_boundaries[-1] + region_length)
    return np.array(counts), region_boundaries, means
    
def create_nucleosomes_locations(mean, std, total_size):
    """Create a list of nucleosome locations sampled from a normal distribution."""
    locations = [0]  # Start with a nucleosome at position 0
    length = 0
    while length < total_size:
        # Ensure distance is larger than 1 and smaller than 800
        distance = max(1, min(800, int(np.random.normal(mean, std))))
        locations.append(locations[-1] + distance)
        length += distance
    # print(locations)
    return locations

def create_nucleosomes_distances(mean, std, total_size):
    """From the generated nucleosome locations, create a list of distances to the closest nucleosome."""
    locations = create_nucleosomes_locations(mean, std, total_size)
    distances = []
    for i in range(total_size):
        # Find the distance to the closest nucleosome
        closest_distance = min(abs(i - loc) for loc in locations)
        distances.append(closest_distance)
    return distances

def load_density_lookup_tables(nucleosome_file, centromere_file):
    """Load the density lookup tables from CSV files.
    
    Returns:
        nucleosome_df: DataFrame with 'distance' and 'mean_density' columns
        centromere_df: DataFrame with 'Bin_Center' and 'mean_density' columns
    """
    nucleosome_df = pd.read_csv(nucleosome_file)
    centromere_df = pd.read_csv(centromere_file)
    return nucleosome_df, centromere_df


def interpolate_density(distance, lookup_df, distance_col):
    """Interpolate density value for a given distance using linear interpolation.
    
    Args:
        distance: The distance value to interpolate for
        lookup_df: DataFrame containing distance and mean_density columns
        distance_col: Name of the distance column ('distance' or 'Bin_Center')
    
    Returns:
        Interpolated mean_density value
    
    Raises:
        ValueError: If distance is outside the range of available data
    """
    distances = lookup_df[distance_col].values
    densities = lookup_df['mean_density'].values
    
    # Check bounds
    if distance < distances.min() or distance > distances.max():
        raise ValueError(f"Distance {distance} is outside the range [{distances.min()}, {distances.max()}]")
    
    # Use numpy's interp for linear interpolation
    return np.interp(distance, distances, densities)


def genereate_pi_values(baseline_pi, baseline_centromere_density, baseline_nucleosome_density, 
                        centromere_distances, nucleosome_distances,
                        nucleosome_file, centromere_file):
    """Generate pi values based on the distances to the centromere and nucleosomes.
    
    Since mean_density = 1 - pi, we work with q = 1 - pi throughout and convert back to pi at the end.
    
    Args:
        baseline_pi: Baseline pi value (e.g., 0.6)
        baseline_centromere_density: Baseline centromere density (mean of mean_density)
        baseline_nucleosome_density: Baseline nucleosome density (mean of mean_density)
        centromere_distances: Array of distances to centromere for each position
        nucleosome_distances: Array of distances to nearest nucleosome for each position
        nucleosome_file: Path to nucleosome density CSV file
        centromere_file: Path to centromere density CSV file
    
    Returns:
        DataFrame with columns: Position, pi_value, centromere_distance, nucleosome_distance
    """
    # Load lookup tables
    nucleosome_df, centromere_df = load_density_lookup_tables(nucleosome_file, centromere_file)
    
    # Work with q = 1 - pi (since mean_density = 1 - pi)
    baseline_q = 1 - baseline_pi
    baseline_centromere_q = baseline_centromere_density
    baseline_nucleosome_q = baseline_nucleosome_density
    
    # Initialize arrays
    size = len(centromere_distances)
    pi_values = np.zeros(size)
    
    # Compute pi for each position
    for i in range(size):
        # Get densities (q values) from interpolation
        centromere_q = interpolate_density(abs(centromere_distances[i]), centromere_df, 'Bin_Center')
        nucleosome_q = interpolate_density(nucleosome_distances[i], nucleosome_df, 'distance')
        
        # Compute relative influences
        centromere_influence = centromere_q / baseline_centromere_q
        nucleosome_influence = nucleosome_q / baseline_nucleosome_q
        
        # Combine influences additively
        q_i = baseline_q * centromere_influence * nucleosome_influence
        
        # Convert back to pi and clip to [0, 1]
        pi_i = 1 - q_i
        pi_values[i] = np.clip(pi_i, 0, 1)
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        'Position': np.arange(size),
        'pi_value': pi_values,
        'centromere_distance': centromere_distances,
        'nucleosome_distance': nucleosome_distances
    })
    
    return result_df


def apply_pi_to_counts(counts, pi_values):
    """Apply pi values to the counts to simulate dropout.
    
    Args:
        counts: Array of original counts
        pi_values: Array of pi values (dropout probabilities) for each position
    
    Returns:
        Array of counts after applying dropout
    """
    assert len(counts) == len(pi_values), "Counts and pi_values must have the same length"
    
    # Simulate dropout by setting counts to zero with probability pi
    dropout_mask = np.random.rand(len(counts)) < pi_values
    modified_counts = np.where(dropout_mask, 0, counts)
    
    return modified_counts

# Plot the density of non_zero values against nucleosome and centromere distances (for centromere using bins of 10000)
# Thus for each distance to the centromere and nucleosome, you compute how many values are non-zero and plot this against the distance. This will show how the dropout (pi) changes with distance to centromere and nucleosomes.
def plot_density_vs_distance(file_path, counts, centromere_distances, nucleosome_distances, bin_size=10000):

    
    # Create bins for centromere distances
    centromere_bins = np.arange(0, max(centromere_distances) + bin_size, bin_size)
    centromere_bin_indices = np.digitize(centromere_distances, centromere_bins)
    
    # Calculate non-zero density for each centromere distance bin
    centromere_density = []
    for i in range(1, len(centromere_bins)):
        bin_mask = (centromere_bin_indices == i)
        if np.sum(bin_mask) > 0:
            density = np.mean(counts[bin_mask] > 0)
            centromere_density.append((centromere_bins[i-1], density))
    
    # Calculate non-zero density for each nucleosome distance
    nucleosome_distance_bins = np.arange(0, max(nucleosome_distances) + 1, 1)  # Bin size of 1 for nucleosome distances
    nucleosome_bin_indices = np.digitize(nucleosome_distances, nucleosome_distance_bins)
    nucleosome_density = []
    for i in range(1, len(nucleosome_distance_bins)):
        bin_mask = (nucleosome_bin_indices == i)
        if np.sum(bin_mask) > 0:
            density = np.mean(counts[bin_mask] > 0)
            nucleosome_density.append((nucleosome_distance_bins[i-1], density))
            
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(*zip(*centromere_density), marker='o')
    plt.title('Non-zero Density vs Centromere Distance')
    plt.xlabel('Distance to Centromere (bp)')
    plt.ylabel('Density of Non-zero Values')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(*zip(*nucleosome_density), marker='o')
    plt.title('Non-zero Density vs Nucleosome Distance')
    plt.xlabel('Distance to Nearest Nucleosome (bp)')
    plt.ylabel('Density of Non-zero Values')
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_path)

if __name__ == "__main__":
    plot = True
    generate_NB = True
    
    total_size = 500000
    mu_mean = 4.4
    mu_std = 6.3
    theta = 0.52
    length_range = (40, 1000)
    
    if generate_NB:
        output_path = f"Signal_processing/sample_data/SATAY_synthetic/SATAY_without_pi.csv"
        counts, region_boundaries, means = generate_NB_sample(mu_mean, mu_std, theta, size=total_size, length_range=length_range)
        print(f"Generated {len(counts)} counts with mean={mu_mean}, std={mu_std}, theta={theta}")
        positions = np.arange(len(counts))
        new_total_size = len(counts)
        df = pd.DataFrame({"Position": positions, "Value": counts})
        df.to_csv(output_path, index=False)
        # Save params
        params_df = pd.DataFrame({
            "region_start": region_boundaries[:-1],
            "region_end": region_boundaries[1:],
            "region_mean": means
        })
        params_df.to_csv(f"Signal_processing/sample_data/SATAY_synthetic/SATAY_without_pi_params.csv", index=False)

    nucl_mean = 166.95
    nucl_std = 33.77
    generate_nucl = True
    if generate_nucl:
        distances = create_nucleosomes_distances(nucl_mean, nucl_std, new_total_size)
        output_path = f"Signal_processing/sample_data/SATAY_synthetic/nucleosome_distances.csv"
        pd.DataFrame({"Distance": distances}).to_csv(output_path, index=False)
        
    generate_centr = True
    middle_position = total_size // 2
    if generate_centr:
        centromere_distances = [abs(pos - middle_position) for pos in range(new_total_size)]
        output_path = f"Signal_processing/sample_data/SATAY_synthetic/centromere_distances.csv"
        pd.DataFrame({"Distance": centromere_distances}).to_csv(output_path, index=False)
        
    baseline_pi = 0.6
    baseline_centromere_density = 0.038
    baseline_nucleosome_density = 0.179
    nucleosome_file = "Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv"
    centromere_file = "Data_exploration/results/densities/centromere/combined_All_Boolean_True_bin_10000_absolute/ALL_combined_centromere_density_Boolean_True_bin_10000_absolute.csv"
    
    # Generate pi values based on centromere and nucleosome distances
    generate_pi = True
    if generate_pi:
        print("Generating pi values...")
        pi_df = genereate_pi_values(
            baseline_pi=baseline_pi,
            baseline_centromere_density=baseline_centromere_density,
            baseline_nucleosome_density=baseline_nucleosome_density,
            centromere_distances=centromere_distances,
            nucleosome_distances=distances,
            nucleosome_file=nucleosome_file,
            centromere_file=centromere_file
        )
        output_path = "Signal_processing/sample_data/SATAY_synthetic/pi_values.csv"
        pi_df.to_csv(output_path, index=False)
        print(f"Pi values saved to {output_path}")
        print(f"Pi value statistics: mean={pi_df['pi_value'].mean():.4f}, std={pi_df['pi_value'].std():.4f}, min={pi_df['pi_value'].min():.4f}, max={pi_df['pi_value'].max():.4f}")
        
    # Assert that each created array has the same length
    assert len(counts) == len(distances) == len(centromere_distances) == len(pi_df), "Generated arrays have different lengths!"
    
    generate_final_counts = True
    if generate_final_counts:
        final_counts = apply_pi_to_counts(counts, pi_df['pi_value'].values)
        output_path = f"Signal_processing/sample_data/SATAY_synthetic/SATAY_with_pi.csv"
        # Save Position, Value, Centromere_distance, Nucleosome_distance
        final_df = pd.DataFrame({
            "Position": np.arange(len(final_counts)),
            "Value": final_counts,
            "Centromere_distance": centromere_distances,
            "Nucleosome_distance": distances,
        })
        final_df.to_csv(output_path, index=False)
        print(f"Final counts with pi applied saved to {output_path}")
    
    if plot:
        plot_density_vs_distance(
            file_path="Signal_processing/sample_data/SATAY_synthetic/density_vs_distance.png",
            counts=final_counts,
            centromere_distances=centromere_distances,
            nucleosome_distances=distances,
            bin_size=10000
        )