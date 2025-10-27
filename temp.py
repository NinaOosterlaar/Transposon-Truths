import numpy as np
import pandas as pd

# Read the basic statistics file
file_path = "Data_exploration/results/basic_statistics.txt"
df = pd.read_csv(file_path, sep='\t')

# Extract density values
densities = df['Density'].values

# Calculate mean and standard deviation
mean_density = np.mean(densities)
std_density = np.std(densities, ddof=1)  # Using sample standard deviation (ddof=1)

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

# Print density values by sample for reference
print("\nDensity values by sample:")
print("-" * 60)
for idx, row in df.iterrows():
    print(f"{row['Sample']:20s}: {row['Density']:.6f}")
