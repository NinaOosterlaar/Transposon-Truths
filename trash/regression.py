import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from reader import read_csv_file_with_distances
from numba import njit, prange



def ZINB_regression(Y, X, Z = None, offset = None, p = 2, exposure = None, method = 'lbfgs', regularized = None, alpha = 1.0):
    """
    Perform Zero-Inflated Negative Binomial regression.
    
    Parameters:
    Y : array-like
        Dependent variable (counts).
    X : array-like
        Independent variables for the count model.
    Z : array-like
        Independent variables for the zero-inflation model. If None, uses X.
    offset : array-like, optional
        Offset for the count model.
    p : int, optional
        Power parameter for the Negative Binomial distribution.
    exposure : array-like, optional
        Exposure variable for the count model.
    method : str, optional
        Optimization method for fitting the model.
    regularized : str, optional
        Regularization method if regularization is desired.
    alpha : float, optional
        Regularization strength.
        
    Returns:
    model : statsmodels object
        The ZINB model instance.
    result : statsmodels object
        Fitted ZINB regression model.
    """
    print(X.mean(), X.std(ddof=0))
    X = (X - X.mean()) / X.std(ddof=0)  # Standardize X)
    
    if Z is None:
        Z = X
    
    # Add constant to independent variables
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)

    # Fit the ZINB model
    model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=X, exog_infl=Z, offset=offset, p=p, exposure=exposure)
    if regularized is not None:
        # 4) pen_weight: 0 for inflate_const, 1 for inflate slopes,
        #                0 for count const, 1 for count slopes,
        #                0 for alpha (last param)
        k_infl = Z.shape[1]           # includes inflate_const
        k_exog = X.shape[1]           # includes const
        pen_weight = np.r_[
            [0.0],                    # inflate_const
            np.ones(k_infl - 1),      # inflate slopes
            [0.0],                    # count const
            np.ones(k_exog - 1),      # count slopes
            [0.0]                     # alpha (dispersion)
        ]

        result = model.fit_regularized(
            method="l1_cvxopt_cp",    # use CVXOPT backend if installed
            alpha=alpha,              # start small (e.g., 1e-4 or 1e-3)
            pen_weight=pen_weight,
            maxiter=5000,
            cnvrg_tol=1e-8,
            trim_mode="size",
            size_trim_tol=1e-6,
            refit=True,
            disp=False
        )
    else:
        result = model.fit(method=method, maxiter=1000, disp=1)

    return model, result

def perform_regression_on_datasets(input_folder="Data_exploration/results/distances_with_zeros", range=None, combine_all=False):
    """
    Perform ZINB regression on datasets read from CSV files in the input folder.
    
    Returns:
    results : dict
        { dataset_name: fitted ZINB result }
    Parameters:
    input_folder : str
        Path to the folder containing CSV files with distance data.
    range : int, optional
        Distance from centromere in base pairs to filter data. If None, no filtering is applied.
    combine_all : bool, optional
        If True combines all the datasets into one before fitting.
    """
    datasets = read_csv_file_with_distances(input_folder)
    if range is not None:
        datasets = set_centromere_range(datasets, range)
    results = {}
    
    if combine_all:
        # Initialize one big dataframe
        combined_df = pd.DataFrame()
        for dataset_name in datasets:
            # Combine all chromosomes in all datasets into one dataframe
            combined_df = pd.concat([combined_df] + list(datasets[dataset_name].values()), ignore_index=True)
        datasets = {"combined": {"all_chromosomes": combined_df}}
        
    for dataset_name in datasets:
        # Print size of dataset
        print(f"\n--- Fitting ZINB for dataset: {dataset_name} ---")

        # Combine all chromosomes into one dataframe
        df_all = pd.concat(datasets[dataset_name].values(), ignore_index=True)
        # Print the size of the combined dataframe
        print(f"Combined dataframe size: {df_all.shape}")

        # Drop any NaNs (shouldn't exist, but just in case)
        df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"])
        # Scale centromere distance from bp to kbp
        df_all["Centromere_Distance"] = df_all["Centromere_Distance"] / 1000.0

        # Dependent and independent variables
        Y = df_all["Value"].astype(int)
        X = df_all[["Nucleosome_Distance", "Centromere_Distance"]]

        # Fit ZINB
        model, result = ZINB_regression(Y, X, Z=None, regularized='l1_cvxopt_cp', alpha=1e-4)
        
        results[dataset_name] = result

    return results

def set_centromere_range(datasets, distance_from_centromere):
    """
    Filter datasets to only include data within a certain distance from the centromere.
    
    Parameters:
    datasets : dict
        { dataset_name: { chromosome_name: dataframe } }
    distance_from_centromere : int
        Distance from centromere in base pairs.
    """
    filtered_datasets = {}
    for dataset_name in datasets:
        filtered_datasets[dataset_name] = {}
        for chrom in datasets[dataset_name]:
            df = datasets[dataset_name][chrom]
            df_filtered = df[np.abs(df["Centromere_Distance"]) <= distance_from_centromere]
            filtered_datasets[dataset_name][chrom] = df_filtered
    return filtered_datasets

if __name__ == "__main__":
    # regression_results = perform_regression_on_datasets("Data_exploration/results/test_dataset")
    combined_results = perform_regression_on_datasets("Data_exploration/results/test_dataset", combine_all=True)
    # Optionally, save or further process regression_results
    output_file = "Data_exploration/results/regression_results.txt"
    with open(output_file, "w") as f:
        # for dataset, result in regression_results.items():
        #     print("params:", result.params)
        #     print("loglike:", result.llf, "AIC:", result.aic, "BIC:", result.bic)
        #     f.write(f"--- {dataset} ---\n")
        #     f.write("params:\n")
        #     f.write(result.params.to_string())
        #     f.write("\n")
        #     f.write(f"loglike: {result.llf}, AIC: {result.aic}, BIC: {result.bic}\n")
        #     f.write("\n")
        f.write("=== Combined Results ===\n")
        for dataset, result in combined_results.items():
            print("params:", result.params)
            print("loglike:", result.llf, "AIC:", result.aic, "BIC:", result.bic)
            f.write(f"--- {dataset} ---\n")
            f.write("params:\n")
            f.write(result.params.to_string())
            f.write("\n")
            f.write(f"loglike: {result.llf}, AIC: {result.aic}, BIC: {result.bic}\n")
            f.write("\n")


