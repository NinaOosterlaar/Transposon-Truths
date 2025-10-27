import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from math import exp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.reader import read_csv_file_with_distances
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


def _logistic(x: float) -> float:
    """Safely compute logistic(x) = 1 / (1 + exp(-x)) for possibly large |x|."""
    # prevent overflow in exp()
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def plot_regression_results(folder, output_file, transform: bool = False) -> None:
    """
    Generate plots for regression results stored in the specified folder.

    Args:
        folder (str): Path to the folder containing regression result txt files
        output_file (str): Base path (".png" will be modified for each figure)
        transform (bool): If False (default), plot raw model coefficients.
                          If True, convert:
                            - inflate_const -> structural zero probability (pi)
                            - other inflate_* -> odds ratio (exp)
                            - count_const -> baseline mean (exp)
                            - other count_* -> fold-change (exp)
    """

    # ---------- 1. Read all summaries ----------
    regression_results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith("_summary.txt"):
            dataset_name = file_name.replace("_summary.txt", "")
            with open(os.path.join(folder, file_name), 'r') as f:
                lines = f.readlines()
                params = {}
                for line in lines:
                    if line.startswith("params:"):
                        continue
                    elif line.startswith("---") or line.startswith("loglike:"):
                        continue
                    else:
                        key_value = line.strip().split()
                        if len(key_value) == 2:
                            key, value = key_value
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
                regression_results[dataset_name] = params

    # ---------- 2. Extract parameters ----------
    datasets = list(regression_results.keys())

    # raw values from file
    raw_pi_const = [regression_results[ds].get('inflate_const', np.nan) for ds in datasets]
    raw_pi_nuc   = [regression_results[ds].get('inflate_Nucleosome_Distance', np.nan) for ds in datasets]
    raw_pi_cent  = [regression_results[ds].get('inflate_Centromere_Distance', np.nan) for ds in datasets]

    raw_cnt_const = [regression_results[ds].get('const', np.nan) for ds in datasets]
    raw_cnt_nuc   = [regression_results[ds].get('Nucleosome_Distance', np.nan) for ds in datasets]
    raw_cnt_cent  = [regression_results[ds].get('Centromere_Distance', np.nan) for ds in datasets]

    # ---------- 3. Optionally transform ----------
    if transform:
        # Zero inflation part
        # inflate_const is on log-odds scale -> convert to probability pi
        pi_const = [_logistic(v) if np.isfinite(v) else np.nan for v in raw_pi_const]

        # other inflate_* coefficients are changes in log-odds per unit -> odds ratios
        pi_nuc  = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_pi_nuc]
        pi_cent = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_pi_cent]

        # Count part
        # const is log(mean) -> mean
        count_const = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_const]

        # slopes are log fold-change per unit -> fold-change multiplier
        count_nuc  = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_nuc]
        count_cent = [np.exp(v) if np.isfinite(v) else np.nan for v in raw_cnt_cent]

        # Names for axes / legends in transformed mode
        param_names = [
            'Inflate: Baseline π (prob)',
            'Inflate: Nucleosome OR',
            'Inflate: Centromere OR',
            'Count: Baseline mean',
            'Count: Nucleosome FC',
            'Count: Centromere FC'
        ]

        heatmap_label = 'Transformed Value'
        detailed_xlabel = [
            'Baseline π (probability of structural zero)',
            'Odds ratio per unit nucleosome distance',
            'Odds ratio per unit centromere distance',
            'Baseline expected mean (growth/count)',
            'Fold-change per unit nucleosome distance',
            'Fold-change per unit centromere distance'
        ]

        zero_line_for_bars = [None, 1.0, 1.0, None, 1.0, 1.0]
        # (In transformed space, "neutral" is 1.0 for ratios/FC, not 0.)

    else:
        # No transform: just plot raw coefficients
        pi_const     = raw_pi_const
        pi_nuc       = raw_pi_nuc
        pi_cent      = raw_pi_cent
        count_const  = raw_cnt_const
        count_nuc    = raw_cnt_nuc
        count_cent   = raw_cnt_cent

        param_names = [
            'Inflate: Const',
            'Inflate: Nucleosome',
            'Inflate: Centromere',
            'Count: Const',
            'Count: Nucleosome',
            'Count: Centromere'
        ]

        heatmap_label = 'Parameter Value'

        detailed_xlabel = [
            'Value',
            'Value',
            'Value',
            'Value',
            'Value',
            'Value'
        ]

        zero_line_for_bars = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # We'll reuse these lists below
    all_param_lists = [
        pi_const, pi_nuc, pi_cent,
        count_const, count_nuc, count_cent
    ]

    # ---------- 4. Heatmap ----------
    param_data = np.array(all_param_lists, dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.4), 8))
    im = ax.imshow(param_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticklabels(param_names)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(heatmap_label, rotation=270, labelpad=20)

    # annotate heatmap cells
    for i in range(len(param_names)):
        for j in range(len(datasets)):
            val = param_data[i, j]
            txt = f'{val:.2f}' if np.isfinite(val) else 'NA'
            ax.text(j, i, txt,
                    ha="center", va="center", color="black", fontsize=8)

    title_suffix = " (transformed)" if transform else " (raw coef)"
    ax.set_title('ZINB Regression Parameters by Dataset - Heatmap' + title_suffix,
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_heatmap_{transform}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- 5. Dot plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(datasets) * 0.3)))
    y_pos = np.arange(len(datasets))

    # Inflate
    ax1.plot(pi_const, y_pos, 'o', label=param_names[0], markersize=8, alpha=0.7)
    ax1.plot(pi_nuc,   y_pos, 's', label=param_names[1], markersize=8, alpha=0.7)
    ax1.plot(pi_cent,  y_pos, '^', label=param_names[2], markersize=8, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(datasets)
    ax1.set_xlabel(heatmap_label)
    ax1.set_title('Inflate Parameters' + title_suffix)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')

    # show reference line:
    if transform:
        # neutral odds/fold-change is 1
        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    else:
        ax1.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)

    # Count
    ax2.plot(count_const, y_pos, 'o', label=param_names[3], markersize=8, alpha=0.7)
    ax2.plot(count_nuc,   y_pos, 's', label=param_names[4], markersize=8, alpha=0.7)
    ax2.plot(count_cent,  y_pos, '^', label=param_names[5], markersize=8, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(datasets)
    ax2.set_xlabel(heatmap_label)
    ax2.set_title('Count Parameters' + title_suffix)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    if transform:
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    else:
        ax2.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('ZINB Regression Parameters by Dataset' + title_suffix,
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_dotplot_{transform}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- 6. Detailed bar subplots ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    params_to_plot = [
        (pi_const, 0, param_names[0]),
        (pi_nuc,   1, param_names[1]),
        (pi_cent,  2, param_names[2]),
        (count_const, 3, param_names[3]),
        (count_nuc,   4, param_names[4]),
        (count_cent,  5, param_names[5]),
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))

    for (values, idx, title), ax in zip(params_to_plot, axes):
        x_pos = np.arange(len(datasets))
        ax.barh(x_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(detailed_xlabel[idx])

        # reference line (0 in raw space, 1 in transformed ratio space)
        zl = zero_line_for_bars[idx]
        if zl is not None:
            ax.axvline(x=zl, color='gray', linestyle='--', alpha=0.5)

        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('ZINB Regression Parameters - Detailed View' + title_suffix,
                 fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(output_file.replace(f".png", f"_detailed_{transform}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated 3 visualization types:",
          "_heatmap.png, _dotplot.png, _detailed.png")
    
def retrieve_average_parameters(folder):
    """
    Retrieve average regression parameters across datasets in the specified folder.
    
    Args:
        folder (str): Path to the folder containing regression result txt files.

    Returns:
        dict: A dictionary containing average regression parameters for each dataset.
    """
    regression_results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith("_summary.txt"):
            dataset_name = file_name.replace("_summary.txt", "")
            with open(os.path.join(folder, file_name), 'r') as f:
                lines = f.readlines()
                params = {}
                for line in lines:
                    if line.startswith("params:"):
                        continue
                    elif line.startswith("---") or line.startswith("loglike:"):
                        continue
                    else:
                        key_value = line.strip().split()
                        if len(key_value) == 2:
                            key, value = key_value
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
                regression_results[dataset_name] = params
    # Transform parameters as in plot_regression_results
    for ds in regression_results:
        # Zero inflation part
        regression_results[ds]['inflate_const'] = _logistic(regression_results[ds]['inflate_const'])
        regression_results[ds]['inflate_Nucleosome_Distance'] = exp(regression_results[ds]['inflate_Nucleosome_Distance'])
        regression_results[ds]['inflate_Centromere_Distance'] = exp(regression_results[ds]['inflate_Centromere_Distance'])
        # Count part
        regression_results[ds]['const'] = exp(regression_results[ds]['const'])
        regression_results[ds]['Nucleosome_Distance'] = exp(regression_results[ds]['Nucleosome_Distance'])
        regression_results[ds]['Centromere_Distance'] = exp(regression_results[ds]['Centromere_Distance'])
    # Compute the mean and standard deviation for each parameter across datasets
    average_params = {}
    for param in regression_results[next(iter(regression_results))].keys():
        values = [regression_results[ds][param] for ds in regression_results if param in regression_results[ds]]
        average_params[param] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return average_params
            

if __name__ == "__main__":
    # plot_regression_results("Data_exploration/results/regression/linear", "Data_exploration/results/regression/linear/regression_parameters.png", transform=False)
    print(retrieve_average_parameters("Data_exploration/results/regression/linear"))
    # out_dir = "Data_exploration/results/regression/linear"
    # # Run per-dataset with immediate writes
    # regression_results = perform_regression_on_datasets(
    #     "Data_exploration/results/distances_with_zeros",
    #     combine_all=False,
    #     output_dir=out_dir,
    #     write_immediately=True,
    # )

    # Run combined (note: this may use more memory by design)
    # combined_results = perform_regression_on_datasets(
    #     "Data_exploration/results/distances_with_zeros",
    #     combine_all=True,
    #     output_dir=out_dir,
    #     write_immediately=True,
    # )

   