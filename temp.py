import numpy as np
import pandas as pd

# Read the basic statistics file
file_path = "Data_exploration/results/basic_statistics.txt"
df = pd.read_csv(file_path, sep='\t')

# Extract density values
densities = df['Density'].values

# Extract mean non-zero count and standard deviation values
mean_non_zero_counts = df['Mean_Non_Zero_Count'].values
std_devs = df['Std_Dev'].values

# Calculate mean and standard deviation for density
mean_density = np.mean(densities)
std_density = np.std(densities, ddof=1)  # Using sample standard deviation (ddof=1)

# Calculate statistics for Mean_Non_Zero_Count
mean_of_mean_non_zero = np.mean(mean_non_zero_counts)
std_mean_non_zero = np.std(mean_non_zero_counts, ddof=1)

# Calculate statistics for Standard Deviation
mean_std_dev = np.mean(std_devs)
std_of_std_dev = np.std(std_devs, ddof=1)

# Print results
print("=" * 60)
print("DENSITY STATISTICS FROM BASIC_STATISTICS.TXT")
print("=" * 60)
print(f"Number of samples: {len(densities)}")
print(f"Mean Density: {mean_density:.6f}")
print(f"Standard Deviation: {std_density:.6f}")
print(f"Min Density: {np.min(densities):.6f}")
print(f"Max Density: {np.max(densities):.6f}")
print(f"Median Density: {np.median(densities):.6f}")
print("=" * 60)

print("\n" + "=" * 60)
print("MEAN NON-ZERO COUNT STATISTICS")
print("=" * 60)
print(f"Number of samples: {len(mean_non_zero_counts)}")
print(f"Mean of Mean Non-Zero Counts: {mean_of_mean_non_zero:.6f}")
print(f"Standard Deviation: {std_mean_non_zero:.6f}")
print(f"Min Mean Non-Zero Count: {np.min(mean_non_zero_counts):.6f}")
print(f"Max Mean Non-Zero Count: {np.max(mean_non_zero_counts):.6f}")
print(f"Median Mean Non-Zero Count: {np.median(mean_non_zero_counts):.6f}")
print("=" * 60)

print("\n" + "=" * 60)
print("STANDARD DEVIATION STATISTICS")
print("=" * 60)
print(f"Number of samples: {len(std_devs)}")
print(f"Mean of Std Dev: {mean_std_dev:.6f}")
print(f"Standard Deviation of Std Dev: {std_of_std_dev:.6f}")
print(f"Min Std Dev: {np.min(std_devs):.6f}")
print(f"Max Std Dev: {np.max(std_devs):.6f}")
print(f"Median Std Dev: {np.median(std_devs):.6f}")
print("=" * 60)

# Print density values by sample for reference
print("\nDensity values by sample:")
print("-" * 60)
for idx, row in df.iterrows():
    print(f"{row['Sample']:20s}: {row['Density']:.6f}")

print("\nMean Non-Zero Count values by sample:")
print("-" * 60)
for idx, row in df.iterrows():
    print(f"{row['Sample']:20s}: {row['Mean_Non_Zero_Count']:.6f}")

print("\nStandard Deviation values by sample:")
print("-" * 60)
for idx, row in df.iterrows():
    print(f"{row['Sample']:20s}: {row['Std_Dev']:.6f}")
    
    

