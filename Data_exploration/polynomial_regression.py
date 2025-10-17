import os
import pickle
from multiprocessing import Pool
import gc

import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse
from reader import label_from_filename

print("Available CPUs:", os.cpu_count())


# =========================
# Feature helpers
# =========================

def _center_and_powers(s, name, max_deg=3):
    """
    Center a Series and return dict of powers up to max_deg.
    Keys: f"{name}_c", f"{name}_c2", f"{name}_c3"
    """
    c = s - s.mean()
    out = {f"{name}_c": c}
    if max_deg >= 2:
        out[f"{name}_c2"] = c ** 2
    if max_deg >= 3:
        out[f"{name}_c3"] = c ** 3
    return out

def _standardize_df(df, skip_cols=()):
    """
    Z-score standardize columns except those in skip_cols.
    Returns standardized df, plus dicts of (means, stds).
    """
    df = df.copy()
    means, stds = {}, {}
    for col in df.columns:
        if col in skip_cols:
            continue
        m = df[col].mean()
        s = df[col].std(ddof=0)
        if not np.isfinite(s) or s == 0:
            s = 1.0
        df[col] = (df[col] - m) / s
        means[col], stds[col] = m, s
    return df, means, stds


# =========================
# Design builders
# =========================

def build_polynomial_design(
    df,
    nuc_col="Nucleosome_Distance",   # bp
    cen_col="Centromere_Distance",   # bp or kb (auto-convert to kb)
    piecewise=False,                 # reference-coding: global + High deviations
    threshold_kb=200.0,              # only used if piecewise=True
    max_deg=3,
):
    """
    Reference-coding (default):
      - Build centered polynomials up to cubic for Nuc (bp) and Cen (kb).
      - If piecewise=True: add High (0/1) and interactions term*High
        to allow different intercept/slope/curvature in High region.
      - One global intercept will be added later in the fitter.

    Returns:
      X_std (count design), Z_std (inflation design), scaler_info (dict)
    """
    d = df.copy()

    # Ensure centromere distance is in kb
    if d[cen_col].abs().max() > 5_000:  # looks like bp
        d[cen_col] = d[cen_col] / 1000.0

    # Base centered powers (up to cubic)
    nuc_feat = _center_and_powers(d[nuc_col], "Nuc", max_deg=max_deg)  # bp
    cen_feat = _center_and_powers(d[cen_col], "Cen", max_deg=max_deg)  # kb

    X = pd.DataFrame({**nuc_feat, **cen_feat})

    if piecewise:
        High = (d[cen_col] > threshold_kb).astype(int)
        X["High"] = High

        inter = {}
        for col in nuc_feat.keys():
            inter[f"{col}_x_High"] = X[col] * High
        for col in cen_feat.keys():
            inter[f"{col}_x_High"] = X[col] * High
        X = pd.concat([X, pd.DataFrame(inter)], axis=1)

    # Standardize continuous columns, skip binary 'High' if present
    skip = ("High",) if piecewise else tuple()
    X_std, means, stds = _standardize_df(X, skip_cols=skip)

    Z_std = X_std.copy()  # same features for inflation by default
    scaler_info = {"means": means, "stds": stds, "threshold_kb": threshold_kb, "mode": "reference"}
    return X_std, Z_std, scaler_info


def build_polynomial_design_separate(
    df,
    nuc_col="Nucleosome_Distance",   # bp
    cen_col="Centromere_Distance",   # bp or kb (auto-convert to kb)
    threshold_kb=200.0,
    max_deg=3,
):
    """
    Fully separate piecewise design (Low & High are independent):
      - Low  = 1{Cen <= threshold_kb}
      - High = 1{Cen >  threshold_kb}
      - Two region-specific intercepts (const_low, const_high)
      - Every centered polynomial is duplicated with _low/_high
      - IMPORTANT: Do NOT add a global constant later (already have two)!

    Returns:
      X (count design), Z (inflation design), scaler_info (dict)
    """
    d = df.copy()

    # Ensure centromere is in kb
    if d[cen_col].abs().max() > 5_000:
        d[cen_col] = d[cen_col] / 1000.0

    High = (d[cen_col] > threshold_kb).astype(int)
    Low  = 1 - High

    # Center BEFORE powers (on whole sample for comparability)
    def feats(s, name):
        c = s - s.mean()
        out = {f"{name}_c": c}
        if max_deg >= 2: out[f"{name}_c2"] = c**2
        if max_deg >= 3: out[f"{name}_c3"] = c**3
        return out

    nuc = feats(d[nuc_col], "Nuc")   # bp
    cen = feats(d[cen_col], "Cen")   # kb
    base = pd.DataFrame({**nuc, **cen})

    # Region-specific intercepts
    X = pd.DataFrame({
        "const_low": Low,
        "const_high": High,
    })

    # Duplicate every polynomial by region
    for col in base.columns:
        X[f"{col}_low"]  = base[col] * Low
        X[f"{col}_high"] = base[col] * High

    # Standardize ONLY continuous cols (leave intercept dummies as 0/1)
    cont_cols = [c for c in X.columns if c not in ("const_low", "const_high")]
    means, stds = {}, {}
    for c in cont_cols:
        m, s = X[c].mean(), X[c].std(ddof=0)
        if not np.isfinite(s) or s == 0: s = 1.0
        X[c] = (X[c] - m) / s
        means[c], stds[c] = m, s

    Z = X.copy()  # same features for inflation by default
    scaler_info = {"means": means, "stds": stds, "threshold_kb": threshold_kb, "mode": "separate"}
    return X, Z, scaler_info


# =========================
# ZINB fitter
# =========================

def ZINB_regression(
    Y,
    X,
    Z=None,
    offset=None,
    p=2,
    exposure=None,
    method="lbfgs",
    regularized=None,   # 'l1' or 'l1_cvxopt_cp' or None
    alpha=1e-4,
    maxiter=5000,
    disp=False,
    add_intercept=True,   # NEW: skip when using separate design
):
    """
    Fit Zero-Inflated Negative Binomial with provided X/Z design matrices.
    Assumes X/Z are already prepared (centered/standardized if desired).
    """
    if Z is None:
        Z = X.copy()

    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
        Z = sm.add_constant(Z, has_constant="add")

    model = sm.ZeroInflatedNegativeBinomialP(
        endog=Y,
        exog=X,
        exog_infl=Z,
        offset=offset,
        exposure=exposure,
        p=p,
        missing="drop",
    )

    if regularized is not None:
        # Do not penalize intercepts or alpha
        k_infl = Z.shape[1]
        k_cnt  = X.shape[1]
        pen_weight = np.r_[
            [0.0],              # inflate_const
            np.ones(k_infl - 1),
            [0.0],              # count const
            np.ones(k_cnt - 1),
            [0.0],              # alpha (dispersion)
        ]
        res = model.fit_regularized(
            method=regularized,
            alpha=alpha,
            pen_weight=pen_weight,
            refit=True,
            maxiter=maxiter,
            cnvrg_tol=1e-8,
            trim_mode="size",
            size_trim_tol=1e-6,
            disp=disp,
        )
    else:
        res = model.fit(method=method, maxiter=maxiter, disp=disp)

    return model, res


# =========================
# Worker & Orchestrator
# =========================

def _load_and_filter_dataset(args):
    """
    Load a single dataset folder in parallel and optionally filter by centromere range.
    Returns (dataset_name, df_all) or None if empty.
    """
    dataset_path, dataset_name, range_bp = args
    
    genome = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            file_path = os.path.join(dataset_path, file)
            df = pd.read_csv(file_path)
            chrom = file.split("_")[0]  # Extract chromosome from filename
            if chrom == "ChrM":
                continue
            
            # Apply centromere filter immediately if specified
            if range_bp is not None:
                mask = np.abs(df["Centromere_Distance"]) <= range_bp
                df = df.loc[mask].copy()  # Copy to avoid fragmentation
            
            if not df.empty:
                genome[chrom] = df
    
    if not genome:
        return None
    
    # Concatenate all chromosomes for this dataset
    df_all = pd.concat(genome.values(), ignore_index=True)
    del genome  # Free chromosome dict
    gc.collect()  # Force garbage collection
    
    return dataset_name, df_all


def _fit_worker(args):
    """
    Load, filter, and fit a single dataset in one worker process.
    This avoids loading all data in the main process.
    """
    try:
        # Avoid BLAS oversubscription per worker
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        (
            dataset_path,
            dataset_name,
            range_bp,
            piecewise,            # reference-coding flag
            separate_piecewise,   # fully separate Low/High flag
            threshold_kb,
            regularized,
            alpha,
            maxiter,
            disp,
            degree
        ) = args

        # Load the dataset in this worker process
        result = _load_and_filter_dataset((dataset_path, dataset_name, range_bp))
        if result is None:
            return dataset_name, None
        
        _, df_all = result
        
        # Clean & units handled in builders
        df_all = df_all.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"]).copy()
        Y = df_all["Value"].astype(int).values

        if piecewise and separate_piecewise:
            # Fully separate design: DO NOT add a global constant later
            X_poly, Z_poly, _ = build_polynomial_design_separate(
                df_all, threshold_kb=threshold_kb, max_deg=degree
            )
            add_int = False
        else:
            # Reference coding (global + High deviations if piecewise=True)
            X_poly, Z_poly, _ = build_polynomial_design(
                df_all, piecewise=piecewise, threshold_kb=threshold_kb, max_deg=degree
            )
            add_int = True

        # Free memory before fitting
        del df_all
        gc.collect()
        
        _, result = ZINB_regression(
            Y=Y,
            X=X_poly,
            Z=Z_poly,
            p=2,
            method="lbfgs",
            regularized=regularized,   # e.g. 'l1_cvxopt_cp' or None
            alpha=alpha,
            maxiter=maxiter,
            disp=disp,
            add_intercept=add_int,
        )
        
        # Free design matrices after fitting
        del Y, X_poly, Z_poly
        gc.collect()
        
        return dataset_name, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error processing dataset at {dataset_path}: {e}")
        return dataset_name, None


def _discover_dataset_paths(input_folder):
    """
    Discover all dataset paths without loading any data.
    Returns list of (dataset_path, dataset_name) tuples.
    """
    dataset_paths = []
    
    # Walk through strain folders
    for strain_folder in os.listdir(input_folder):
        strain_path = os.path.join(input_folder, strain_folder)
        if not os.path.isdir(strain_path):
            continue
        
        # Each subfolder in strain_xxx is a separate dataset
        for dataset_folder in os.listdir(strain_path):
            dataset_path = os.path.join(strain_path, dataset_folder)
            if not os.path.isdir(dataset_path):
                continue
            
            # Check if it contains CSV files
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
            if csv_files:
                # Use label_from_filename from reader to get consistent naming
                
                dataset_name = label_from_filename(dataset_folder)
                dataset_paths.append((dataset_path, dataset_name))
    
    return dataset_paths


def perform_regression_on_datasets(
    input_folder="Data_exploration/results/distances_with_zeros",
    range_bp=None,                 # e.g., 200_000 to restrict to ±200 kb
    combine_all=False,
    piecewise=False,               # reference coding toggle
    separate_piecewise=False,      # fully separate Low/High toggle
    threshold_kb=200.0,            # split at 200 kb (in kb units)
    regularized=None,              # 'l1_cvxopt_cp' or None
    alpha=1e-4,
    maxiter=5000,
    disp=False,
    degree=3,                      # polynomial degree (1, 2, or 3)
    output_dir="Data_exploration/results/regression/poly",  # NEW: output directory
    write_immediately=True,        # NEW: write results as they complete
):
    """
    Fits a polynomial (up to cubic) ZINB per dataset.
    Choose:
      - piecewise=False, separate_piecewise=False : global polynomials (no split)
      - piecewise=True,  separate_piecewise=False : reference coding (global + High deviations)
      - piecewise=True,  separate_piecewise=True  : fully separate Low/High
      
    MEMORY EFFICIENT: Loads and processes each dataset in parallel workers,
    never holding all data in main process memory.
    
    If write_immediately=True, results are written to disk as each worker completes,
    reducing memory usage and providing fault tolerance.
    """
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover dataset paths WITHOUT loading data
    dataset_paths = _discover_dataset_paths(input_folder)
    print(f"Found {len(dataset_paths)} datasets to process")
    
    if combine_all:
        # For combine_all mode, we need to load all data
        # Do this in parallel then combine
        print("\n--- Loading all datasets for combined analysis ---")
        load_tasks = [(path, name, range_bp) for path, name in dataset_paths]
        
        all_dfs = []
        with Pool(processes=os.cpu_count()) as pool:
            for result in pool.imap_unordered(_load_and_filter_dataset, load_tasks):
                if result is not None:
                    _, df = result
                    all_dfs.append(df)
        
        if not all_dfs:
            return {}
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs  # Free memory
        gc.collect()
        
        # Build single task for combined dataset
        tasks = [(
            "combined",  # dataset_path (not used in fit_worker for pre-loaded data)
            "combined",  # dataset_name
            None,        # range_bp (already applied)
            piecewise,
            separate_piecewise,
            threshold_kb,
            regularized,
            alpha,
            maxiter,
            disp,
            degree,
        )]
        
        # We need a special path for combined - pass the dataframe directly
        # This requires modifying the approach slightly
        print(f"Combined dataset rows: {len(combined_df):,}")
        
        # Process combined directly (can't pickle DataFrame easily in args)
        combined_df = combined_df.dropna(subset=["Value", "Nucleosome_Distance", "Centromere_Distance"]).copy()
        Y = combined_df["Value"].astype(int).values
        
        if piecewise and separate_piecewise:
            X_poly, Z_poly, _ = build_polynomial_design_separate(
                combined_df, threshold_kb=threshold_kb, max_deg=degree
            )
            add_int = False
        else:
            X_poly, Z_poly, _ = build_polynomial_design(
                combined_df, piecewise=piecewise, threshold_kb=threshold_kb, max_deg=degree
            )
            add_int = True
        
        del combined_df
        gc.collect()
        
        _, result = ZINB_regression(
            Y=Y, X=X_poly, Z=Z_poly, p=2,
            method="lbfgs", regularized=regularized, alpha=alpha,
            maxiter=maxiter, disp=disp, add_intercept=add_int,
        )
        
        del Y, X_poly, Z_poly
        gc.collect()
        
        # Write combined result immediately if requested
        if write_immediately:
            result.save(os.path.join(output_dir, "combined_poly_result.pickle"))
            print(f"✓ Saved: combined")
        
        return {"combined": result}
    
    else:
        # Process each dataset independently in parallel
        # Build tasks with dataset paths (loading happens in worker)
        tasks = [
            (
                dataset_path,
                dataset_name,
                range_bp,
                piecewise,
                separate_piecewise,
                threshold_kb,
                regularized,
                alpha,
                maxiter,
                disp,
                degree,
            )
            for dataset_path, dataset_name in dataset_paths
        ]
        
        # Parallel load + fit with immediate writing
        results = {}
        completed_count = 0
        if tasks:
            with Pool(processes=os.cpu_count()) as pool:
                for dataset_name, result_bytes in pool.imap_unordered(_fit_worker, tasks):
                    if result_bytes is not None:
                        result = pickle.loads(result_bytes)
                        results[dataset_name] = result
                        completed_count += 1
                        
                        # Write immediately to disk
                        if write_immediately:
                            result.save(os.path.join(output_dir, f"{dataset_name}_poly_result.pickle"))
                            # Also write summary to text file
                            summary_file = os.path.join(output_dir, f"{dataset_name}_summary.txt")
                            with open(summary_file, "w") as f:
                                f.write(f"Dataset: {dataset_name}\n")
                                f.write("=" * 60 + "\n\n")
                                f.write("Parameters:\n")
                                f.write(result.params.to_string())
                                f.write("\n\n")
                                f.write(f"Log-Likelihood: {result.llf}\n")
                                f.write(f"AIC: {result.aic}\n")
                                f.write(f"BIC: {result.bic}\n")
                            
                            print(f"✓ Completed ({completed_count}/{len(tasks)}): {dataset_name}")
                            
                            # Free memory immediately after writing
                            del result
                            results[dataset_name] = None  # Keep track but don't hold in memory
                            gc.collect()
                    else:
                        print(f"✗ Skipped (empty): {dataset_name}")
        
        return results

def parse_args():
    p = argparse.ArgumentParser(description="ZINB with polynomial features (optional piecewise splitting).")
    p.add_argument("--piecewise", action="store_true", help="Enable piecewise split at threshold (reference coding unless --separate).")
    p.add_argument("--separate", action="store_true", help="Use fully separate Low/High (requires --piecewise).")
    p.add_argument("--degree", type=int, default=3, choices=[1,2,3], help="Polynomial degree (max 3).")
    return p.parse_args()


# =========================
# Main
# =========================

if __name__ == "__main__":
    args = parse_args()
    
    # Define output directory
    output_dir = "Data_exploration/results/regression/poly"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: per-dataset, fully separate Low/High (your requested mode)
    # Results are written immediately as each worker completes
    print("\n" + "="*60)
    print("PROCESSING INDIVIDUAL DATASETS")
    print("="*60)
    regression_results = perform_regression_on_datasets(
        input_folder="Data_exploration/results/distances_with_zeros",
        range_bp=None,
        combine_all=False,
        piecewise=args.piecewise,
        separate_piecewise=args.separate,    # << fully separate regimes
        threshold_kb=200.0,         # split at 200 kb
        regularized='l1',           # try None first; then 'l1_cvxopt_cp' with small alpha
        alpha=1e-6,
        maxiter=5000,
        disp=False,
        degree=args.degree,
        output_dir=output_dir,
        write_immediately=True,  # Write results as they complete
    )

    # Example 2 (optional): combined, reference coding (shared baseline + High tweaks)
    print("\n" + "="*60)
    print("PROCESSING COMBINED DATASET")
    print("="*60)
    combined_results = perform_regression_on_datasets(
        input_folder="Data_exploration/results/distances_with_zeros",
        range_bp=None,
        combine_all=True,
        piecewise=True,
        separate_piecewise=False,   # << reference coding
        threshold_kb=200.0,
        regularized='l1',  # try regularization on combined
        alpha=1e-4=6,
        maxiter=5000,
        disp=False,
        degree=args.degree,
        output_dir=output_dir,
        write_immediately=True,  # Write result when complete
    )

    # Write a final summary file with all results
    summary_file = os.path.join(
        output_dir, 
        f"all_results_piecewise_{args.piecewise}_separate_{args.separate}_degree_{args.degree}.txt"
    )
    print(f"\nWriting final summary to: {summary_file}")
    
    with open(summary_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("POLYNOMIAL REGRESSION RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Piecewise: {args.piecewise}\n")
        f.write(f"Separate: {args.separate}\n")
        f.write(f"Degree: {args.degree}\n")
        f.write("="*60 + "\n\n")
        
        f.write("INDIVIDUAL DATASETS\n")
        f.write("-"*60 + "\n")
        for dataset_name in sorted(regression_results.keys()):
            # Read from individual summary files since we freed memory
            indiv_summary = os.path.join(output_dir, f"{dataset_name}_summary.txt")
            if os.path.exists(indiv_summary):
                with open(indiv_summary, "r") as sf:
                    f.write(sf.read())
                    f.write("\n")
            else:
                f.write(f"Dataset: {dataset_name} - No summary found\n\n")

        f.write("\n" + "="*60 + "\n")
        f.write("COMBINED RESULTS\n")
        f.write("-"*60 + "\n")
        for dataset_name, result in combined_results.items():
            if result is not None:
                f.write(f"Dataset: {dataset_name}\n")
                f.write("Parameters:\n")
                f.write(result.params.to_string())
                f.write("\n\n")
                f.write(f"Log-Likelihood: {result.llf}\n")
                f.write(f"AIC: {result.aic}\n")
                f.write(f"BIC: {result.bic}\n\n")
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE")
    print("="*60)
    print(f"Individual results saved to: {output_dir}/")
    print(f"Summary file: {summary_file}")