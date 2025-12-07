import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sample_zinb(mu, theta, pi, size):
    """
    Sample from a Zero-Inflated Negative Binomial (ZINB).

    Args:
        mu (float): Mean of the NB component (>0).
        theta (float): Dispersion (>0). Var = mu + mu^2 / theta.
        pi (float): Zero probability in [0, 1).
        size (int): Number of samples.

    Returns:
        counts (np.ndarray): ZINB-sampled integer counts.
    """
    min_mu = 0.05
    mu = max(mu, min_mu)
    p = theta / (theta + mu)
    nb_counts = np.random.negative_binomial(theta, p, size=size)

    # Zero inflation
    zero_mask = np.random.rand(size) < pi
    counts = np.where(zero_mask, 0, nb_counts)

    return counts.astype(int)


def make_random_zinb_counts(
    n_regions,
    mean_range=(0.1, 40.0),
    length_mean_range=(40, 1000),
    theta_range=(5.0, 60.0),
    zero_prob_range=(0.1, 0.8),
    min_region_length=20,
    probability_zero_region=0.1,
):
    """
    Generate synthetic count data composed of multiple regions, each with counts
    drawn from a Zero-Inflated Negative Binomial (ZINB) distribution.

    Args:
        n_regions (int): Number of distinct regions to generate.
        mean_range (tuple): Min and max mean (mu) values for the NB component
            of each region.
        length_mean_range (tuple): Min and max mean lengths for each region.
        theta_range (tuple): Min and max values for the NB dispersion (theta).
        zero_prob_range (tuple): Min and max zero-inflation probabilities (pi).
        min_region_length (int): Minimum length for each region.
        probability_zero_region (float): Probability that a region has mean=0.

    Returns:
        counts (np.ndarray): Array of generated counts across all regions.
        region_boundaries (list): End positions (indices) for each region.
        region_params (dict): Parameters used for each region.
    """

    # Sample per-region means (mu)
    region_means = np.random.uniform(*mean_range, size=n_regions)

    # Some regions are forced to be "zero regions"
    zero_region_mask = np.random.rand(n_regions) < probability_zero_region
    region_means[zero_region_mask] = 0.0

    # Sample per-region lengths
    region_lengths = np.random.uniform(*length_mean_range, size=n_regions)

    # Sample per-region dispersion (theta) and zero-inflation probs (pi)
    region_thetas = np.random.uniform(*theta_range, size=n_regions)
    region_zero_probs = np.random.uniform(*zero_prob_range, size=n_regions)

    region_params = {
        "region_means": region_means,
        "region_thetas": region_thetas,
        "region_zero_probs": region_zero_probs,
        "region_lengths": region_lengths,
        "zero_region_mask": zero_region_mask,
    }

    counts = []
    region_boundaries = []
    pos = 0

    for n in range(n_regions):
        L = int(np.round(region_lengths[n]))
        L = max(L, min_region_length)

        region_vals = sample_zinb(
            mu=region_means[n],
            theta=region_thetas[n],
            pi=region_zero_probs[n],
            size=L,
        )

        counts.append(region_vals)
        pos += L
        region_boundaries.append(pos - 1)

    counts = np.concatenate(counts)

    return counts, region_boundaries, region_params


if __name__ == "__main__":
    plot = True
    generate = True

    if generate:
        number_of_regions = 2000
        output_path = f"Signal_processing/realistic_data.csv"

        counts, boundaries, params = make_random_zinb_counts(
            n_regions=number_of_regions,
            mean_range=(0.1, 40.0),
            length_mean_range=(40, 1000),
            theta_range=(5.0, 60.0),
            zero_prob_range=(0.7, 0.95),
            min_region_length=20,
            probability_zero_region=0.1,
        )

        positions = np.arange(len(counts))
        df = pd.DataFrame({"Position": positions, "Count": counts})
        df.to_csv(output_path, index=False)
        # Save params
        params_output_path = f"Signal_processing/realistic_data_params.csv"
        params_df = pd.DataFrame(params)
        params_df.to_csv(params_output_path, index=False)

    if plot:
        counts, boundaries, params = make_random_zinb_counts(
            n_regions=5,
            mean_range=(0.1, 40.0),
            length_mean_range=(40, 1000),
            theta_range=(5.0, 60.0),
            zero_prob_range=(0.7, 0.95),
            min_region_length=20,
            probability_zero_region=0.1,
        )

        positions = np.arange(len(counts))
        plt.figure(figsize=(12, 3))
        plt.bar(positions, counts, width=1.0, color="black")
        for b in boundaries[:-1]:
            plt.axvline(b + 0.5, linestyle="--", color="gray", linewidth=1)
        plt.xlabel("Position")
        plt.ylabel("Count")
        plt.title("Randomly Generated ZINB Counts")
        plt.tight_layout()
        plt.show()

        print("Region means (mu):", params["region_means"])
        print("Region thetas:", params["region_thetas"])
        print("Region zero probs (pi):", params["region_zero_probs"])
