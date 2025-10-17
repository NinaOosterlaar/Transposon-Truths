import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from reader import read_csv_file_with_distances
from multiprocessing import Pool, Process
import pickle
import gc  # For explicit memory cleanup



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
        result = model.fit(method=method, maxiter=5000, disp=1)

    return model, result

def _fit_worker(args):
    """Run one fit in a separate process and return pickled result."""
    # avoid BLAS oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    dataset_name, Y, X = args
    # Call your existing function (unchanged)
    _, result = ZINB_regression(Y, X, Z=None, regularized='l1_cvxopt_cp', alpha=1e-4)
    return dataset_name, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def _write_result_immediately(dataset_name: str, result, output_dir: str) -> None:
    """
    Persist results for one dataset immediately to disk (pickle + text summary).
    Falls back to raw pickle if statsmodels save() is unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save full result
    pickle_path = os.path.join(output_dir, f"{dataset_name}_poly_result.pickle")
    try:
        result.save(pickle_path)
    except Exception:
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) Save quick human-readable summary
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(f"--- {dataset_name} ---\n")
            f.write("params:\n")
            try:
                f.write(result.params.to_string())
                f.write("\n")
            except Exception:
                f.write("<params unavailable>\n")

            # Fit metrics if available
            llf = getattr(result, 'llf', 'NA')
            aic = getattr(result, 'aic', 'NA')
            bic = getattr(result, 'bic', 'NA')
            f.write(f"loglike: {llf}, AIC: {aic}, BIC: {bic}\n")
    except Exception:
        # Best-effort; if summary writing fails, continue
        pass

def perform_regression_on_datasets(
    input_folder: str = "Data_exploration/results/distances_with_zeros",
    range=None,
    combine_all: bool = False,
    output_dir: str = "Data_exploration/results/regression/linear",
    write_immediately: bool = True,
):
    datasets = read_csv_file_with_distances(input_folder)
    if range is not None:
        datasets = set_centromere_range(datasets, range)
    results = {}

    # Ensure output directory exists when immediate writing is enabled
    if write_immediately:
        os.makedirs(output_dir, exist_ok=True)

    if combine_all:
        combined_df = pd.DataFrame()
        for dataset_name in datasets:
            combined_df = pd.concat([combined_df] + list(datasets[dataset_name].values()), ignore_index=True)
        datasets = {"combined": {"all_chromosomes": combined_df}}

    # build tasks
    tasks = []
    for dataset_name in datasets:
        print(f"\n--- Fitting ZINB for dataset: {dataset_name} ---")
        df_all = pd.concat(datasets[dataset_name].values(), ignore_index=True)
        print(f"Combined dataframe size: {df_all.shape}")

        df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"]).copy()
        df_all["Centromere_Distance"] = df_all["Centromere_Distance"] / 1000.0

        Y = df_all["Value"].astype(int)
        X = df_all[["Nucleosome_Distance", "Centromere_Distance"]]
        tasks.append((dataset_name, Y, X))

    # parallel fit across datasets
    if tasks:
        with Pool(processes=os.cpu_count()) as pool:
            for dataset_name, result_bytes in pool.imap_unordered(_fit_worker, tasks):
                result = pickle.loads(result_bytes)

                # Save immediately for crash resilience
                if write_immediately:
                    _write_result_immediately(dataset_name, result, output_dir)
                    print(f"✓ Saved result for {dataset_name}", flush=True)

                # Optionally keep in memory (set to None to minimize memory when writing immediately)
                results[dataset_name] = None if write_immediately else result

                # Free memory ASAP
                del result
                gc.collect()

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
    out_dir = "Data_exploration/results/regression/linear"
    # Run per-dataset with immediate writes
    regression_results = perform_regression_on_datasets(
        "Data_exploration/results/distances_with_zeros",
        combine_all=False,
        output_dir=out_dir,
        write_immediately=True,
    )

    # Run combined (note: this may use more memory by design)
    combined_results = perform_regression_on_datasets(
        "Data_exploration/results/distances_with_zeros",
        combine_all=True,
        output_dir=out_dir,
        write_immediately=True,
    )

    # Compose a final summary file from individual summaries (memory-safe)
    summary_path = os.path.join(out_dir, "regression_results.txt")
    with open(summary_path, "w") as f:
        # Per-dataset summaries
        for dataset in sorted(regression_results.keys()):
            summary_file = os.path.join(out_dir, f"{dataset}_summary.txt")
            if os.path.exists(summary_file):
                with open(summary_file, "r") as sf:
                    f.write(sf.read())
                    f.write("\n")

        # Combined result summary (if present)
        for dataset, result in combined_results.items():
            if result is not None:
                f.write(f"--- {dataset} ---\n")
                try:
                    f.write("params:\n")
                    f.write(result.params.to_string())
                    f.write("\n")
                except Exception:
                    f.write("<params unavailable>\n")
                llf = getattr(result, 'llf', 'NA')
                aic = getattr(result, 'aic', 'NA')
                bic = getattr(result, 'bic', 'NA')
                f.write(f"loglike: {llf}, AIC: {aic}, BIC: {bic}\n\n")

    print(f"\nSaved per-dataset results and summary to: {out_dir}")


