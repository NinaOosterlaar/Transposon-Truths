import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from reader import read_csv_file_with_distances, fill_data_with_zeros



def ZINB_regression(Y, X, Z = None, offset = None, p = 2, exposure = None, method = 'bfgs', regularized = None, alpha = 1.0):
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
        Regularization method ('l1', 'l2', etc.) if regularization is desired.
    alpha : float, optional
        Regularization strength.
        
    Returns:
    model : statsmodels object
        The ZINB model instance.
    result : statsmodels object
        Fitted ZINB regression model.
    """
    print(X.head())
    X = (X - X.mean()) / X.std(ddof=0)  # Standardize X)
    
    if Z is None:
        Z = X
    
    # Add constant to independent variables
    X = sm.add_constant(X)
    Z = sm.add_constant(Z)

    # Fit the ZINB model
    model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=X, exog_infl=Z, offset=offset, p=p, exposure=exposure)
    if regularized is not None:
        result = model.fit_regularized(method=regularized, alpha=alpha, maxiter=100, disp=0)
    else:
        result = model.fit(method='bfgs', maxiter=100, disp=1)

    return model, result


def perform_regression_on_datasets(input_folder="Data_exploration/results/distances"):
    """
    Perform ZINB regression on datasets read from CSV files in the input folder.
    
    Returns:
    results : dict
        { dataset_name: fitted ZINB result }
    """
    datasets = read_csv_file_with_distances(input_folder)
    results = {}
    for dataset_name in datasets:
        print(f"\n--- Fitting ZINB for dataset: {dataset_name} ---")

        # Combine all chromosomes into one dataframe
        df_all = pd.concat(datasets[dataset_name].values(), ignore_index=True)

        # Drop any NaNs (shouldn't exist, but just in case)
        df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"])
        # Scale centromere distance from bp to kbp
        df_all["Centromere_Distance"] = df_all["Centromere_Distance"] / 1000.0

        # Dependent and independent variables
        Y = df_all["Value"].astype(int)
        X = df_all[["Nucleosome_Distance", "Centromere_Distance"]]

        # Fit ZINB
        model, result = ZINB_regression(Y, X, Z=None)
        
        results[dataset_name] = result

        return results

if __name__ == "__main__":
    regression_results = perform_regression_on_datasets("Data_exploration/results/test_dataset")
    # Optionally, save or further process regression_results
    for dataset, result in regression_results.items():
        print(f"Results for {dataset}:")
        print(result.summary())


