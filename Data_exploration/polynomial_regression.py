import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse

from reader import read_csv_file_with_distances  # your existing loader


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

def _fit_worker(args):
    """
    Run one fit in a separate process and return (dataset_name, pickled_result).
    """
    # Avoid BLAS oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    (
        dataset_name,
        df_all,
        piecewise,            # reference-coding flag
        separate_piecewise,   # fully separate Low/High flag
        threshold_kb,
        regularized,
        alpha,
        maxiter,
        disp,
        degree
    ) = args

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
    return dataset_name, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def set_centromere_range(datasets, distance_from_centromere_bp):
    """Filter to |Centromere_Distance| <= threshold (bp)."""
    filtered = {}
    for ds in datasets:
        filtered[ds] = {}
        for chrom, df in datasets[ds].items():
            m = np.abs(df["Centromere_Distance"]) <= distance_from_centromere_bp
            filtered[ds][chrom] = df.loc[m]
    return filtered


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
):
    """
    Fits a polynomial (up to cubic) ZINB per dataset.
    Choose:
      - piecewise=False, separate_piecewise=False : global polynomials (no split)
      - piecewise=True,  separate_piecewise=False : reference coding (global + High deviations)
      - piecewise=True,  separate_piecewise=True  : fully separate Low/High
    """
    datasets = read_csv_file_with_distances(input_folder)
    if range_bp is not None:
        datasets = set_centromere_range(datasets, range_bp)

    # Optionally combine all datasets into a single "combined"
    if combine_all:
        combined_df = pd.concat(
            [df for ds in datasets.values() for df in ds.values()],
            ignore_index=True,
        )
        datasets = {"combined": {"all_chromosomes": combined_df}}

    # Build tasks
    tasks = []
    for dataset_name, chrom_dict in datasets.items():
        print(f"\n--- Fitting ZINB (poly ≤ x^3) for dataset: {dataset_name} ---")
        df_all = pd.concat(chrom_dict.values(), ignore_index=True)
        print(f"  rows: {len(df_all):,}")
        tasks.append((
            dataset_name,
            df_all,
            piecewise,
            separate_piecewise,
            threshold_kb,
            regularized,
            alpha,
            maxiter,
            disp,
            degree,
        ))

    # Parallel fits
    results = {}
    if tasks:
        with Pool(processes=os.cpu_count()) as pool:
            for dataset_name, result_bytes in pool.imap_unordered(_fit_worker, tasks):
                results[dataset_name] = pickle.loads(result_bytes)

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
    # Example 1: per-dataset, fully separate Low/High (your requested mode)
    regression_results = perform_regression_on_datasets(
        input_folder="Data_exploration/results/distances_with_zeros",
        range_bp=None,
        combine_all=False,
        piecewise=args.piecewise,
        separate_piecewise=args.separate,    # << fully separate regimes
        threshold_kb=200.0,         # split at 200 kb
        regularized=None,           # try None first; then 'l1_cvxopt_cp' with small alpha
        alpha=1e-4,
        maxiter=5000,
        disp=False,
        degree=args.degree,
    )

    # Example 2 (optional): combined, reference coding (shared baseline + High tweaks)
    combined_results = perform_regression_on_datasets(
        input_folder="Data_exploration/results/distances_with_zeros",
        range_bp=None,
        combine_all=True,
        piecewise=True,
        separate_piecewise=False,   # << reference coding
        threshold_kb=200.0,
        regularized=None,
        alpha=1e-4,
        maxiter=5000,
        disp=False,
    )

    # Write compact outputs
    output_file = f"Data_exploration/results/regression/poly/poly_piecewise_{args.degree}_separate_{args.separate}_degree_{args.degree}.txt"
    with open(output_file, "w") as f:
        for dataset, result in regression_results.items():
            print(f"\nResults for {dataset}:")
            print("params:", result.params)
            print("loglike:", result.llf, "AIC:", result.aic, "BIC:", result.bic)
            f.write(f"--- {dataset} ---\n")
            f.write("params:\n")
            f.write(result.params.to_string())
            f.write("\n")
            f.write(f"loglike: {result.llf}, AIC: {result.aic}, BIC: {result.bic}\n\n")

        f.write("=== Combined Results ===\n")
        for dataset, result in combined_results.items():
            print(f"\nResults for {dataset}:")
            print("params:", result.params)
            print("loglike:", result.llf, "AIC:", result.aic, "BIC:", result.bic)
            f.write(f"--- {dataset} ---\n")
            f.write("params:\n")
            f.write(result.params.to_string())
            f.write("\n")
            f.write(f"loglike: {result.llf}, AIC: {result.aic}, BIC: {result.bic}\n\n")
        
    # Now save the result instance to be sure as pickle file 
    for dataset, result in regression_results.items():
        result.save(f"Data_exploration/results/regression/poly/{dataset}_poly_result.pickle")
    for dataset, result in combined_results.items():
        result.save(f"Data_exploration/results/regression/poly/{dataset}_poly_result.pickle")