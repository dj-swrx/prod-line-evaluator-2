
#!/usr/bin/env python3
"""
Feature Importance & Optimization Pipeline for Production Configuration Parameters

This script:
  1) Loads a dataset with many configuration parameters (e.g., 2,500 features)
  2) Cleans and scales features
  3) Trains Lasso (L1), Random Forest, and computes Permutation Importance
  4) Produces SHAP interpretability plots (optional)
  5) Combines importance scores into a single ranking
  6) Runs Bayesian optimization with Optuna to minimize percent error (optional)
  7) Saves all plots and CSV artifacts

Usage:
    python pipeline_feature_importance_optimization.py \
        --input production_data.csv \
        --target percent_error \
        --outdir ./outputs \
        --top-n 30 \
        --top-k 50 \
        --n-trials 100 \
        --enable-shap \
        --enable-optuna

Author: M365 Copilot ChatGPT-5
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

# Optional libraries; the script will handle absence gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

RANDOM_STATE = 42


# -----------------------------
# Utility Functions
# -----------------------------
def setup_logging(outdir: Path) -> None:
    """Configure logging to console and file."""
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Logging initialized. Output directory: {outdir}")


def save_fig(path: Path, tight=True):
    """Helper to save current matplotlib figure."""
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0,1], handling NaNs and degenerate ranges."""
    s = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vmin, vmax = s.min(), s.max()
    if np.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - vmin) / (vmax - vmin)


# -----------------------------
# Core Pipeline Steps
# -----------------------------
def load_and_preprocess(input_path: Path, target_col: str, test_size: float):
    """Load CSV, clean, encode, scale features, return train/test splits and scaler."""
    logging.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path)

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Separate features and target
    feature_cols = [c for c in data.columns if c != target_col]
    X_raw = data[feature_cols].copy()
    y = data[target_col].copy()

    # Basic cleaning:
    # - Numeric NaNs -> column mean
    # - Non-numeric -> category codes
    for col in X_raw.columns:
        if X_raw[col].dtype.kind in "biufc":
            X_raw[col] = X_raw[col].fillna(X_raw[col].mean())
        else:
            X_raw[col] = X_raw[col].astype("category").cat.codes

    # Train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # Scale features (Lasso requires scaling; we'll use scaled consistently)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_cols,
        index=X_train_raw.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_cols,
        index=X_test_raw.index,
    )

    logging.info(
        f"Data shapes: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def lasso_importance(X_train: pd.DataFrame, y_train: pd.Series, feature_cols, outdir: Path, top_n: int):
    """Train LassoCV and plot top-N feature importances."""
    logging.info("Training LassoCV for feature selection...")
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_alphas=100, max_iter=5000)
    lasso.fit(X_train, y_train)

    lasso_importance_vals = np.abs(lasso.coef_)
    lasso_df = pd.DataFrame({
        "feature": feature_cols,
        "lasso_importance": lasso_importance_vals,
    }).sort_values("lasso_importance", ascending=False)

    # Visualization
    plt.figure(figsize=(10, 10))
    sns.barplot(
        data=lasso_df.head(top_n),
        y="feature", x="lasso_importance", palette="viridis"
    )
    plt.title("Top Lasso (L1) Feature Importances")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    save_fig(outdir / "lasso_top_features.png")

    # Save CSV
    lasso_df.to_csv(outdir / "feature_importance_lasso.csv", index=False)
    logging.info("Lasso importance computed and saved.")
    return lasso_df


def random_forest_importance(X_train: pd.DataFrame, y_train: pd.Series, feature_cols, outdir: Path, top_n: int):
    """Train RandomForestRegressor and plot top-N feature importances."""
    logging.info("Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_importance_vals = rf.feature_importances_
    rf_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importance_vals,
    }).sort_values("rf_importance", ascending=False)

    # Visualization
    plt.figure(figsize=(10, 10))
    sns.barplot(
        data=rf_df.head(top_n),
        y="feature", x="rf_importance", palette="magma"
    )
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Mean Decrease in Impurity")
    plt.ylabel("Feature")
    save_fig(outdir / "rf_top_features.png")

    # Save CSV
    rf_df.to_csv(outdir / "feature_importance_rf.csv", index=False)
    logging.info("Random Forest importance computed and saved.")
    return rf, rf_df


def permutation_importance_eval(rf, X_test: pd.DataFrame, y_test: pd.Series, feature_cols, outdir: Path, top_n: int, perm_sample_size: int = None):
    """Compute permutation importance on the test set and plot results."""
    logging.info("Computing permutation importance on the test set...")

    # Optional sampling for speed
    if perm_sample_size is not None and perm_sample_size < len(X_test):
        X_perm = X_test.sample(perm_sample_size, random_state=RANDOM_STATE)
        y_perm = y_test.loc[X_perm.index]
    else:
        X_perm, y_perm = X_test, y_test

    perm = permutation_importance(
        rf, X_perm, y_perm, n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)

    # Visualization with error bars
    plt.figure(figsize=(10, 10))
    top_perm = perm_df.head(top_n)
    sns.barplot(
        data=top_perm,
        y="feature", x="perm_importance_mean", palette="coolwarm", orient="h", errorbar=None
    )
    plt.errorbar(
        top_perm["perm_importance_mean"], range(len(top_perm)),
        xerr=top_perm["perm_importance_std"], fmt="none", ecolor="black", capsize=3
    )
    plt.title("Top Permutation Importances (Test Set)")
    plt.xlabel("Mean Importance (± Std)")
    plt.ylabel("Feature")
    save_fig(outdir / "perm_top_features.png")

    # Save CSV
    perm_df.to_csv(outdir / "feature_importance_perm.csv", index=False)
    logging.info("Permutation importance computed and saved.")
    return perm_df


def shap_analysis(rf, X_test: pd.DataFrame, feature_cols, outdir: Path, shap_sample_size: int, top_n_dependence: int):
    """Run SHAP analysis (if available) and save plots."""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not installed. Skipping SHAP analysis.")
        return

    logging.info("Running SHAP analysis (TreeExplainer)...")
    # SHAP can be heavy; sample for speed
    sample_size = min(shap_sample_size, len(X_test))
    X_test_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)

    # SHAP may warn about model types & approximations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_sample)

    # Summary (beeswarm)
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, plot_type="dot", show=False)
    save_fig(outdir / "shap_summary_beeswarm.png")

    # Summary (bar)
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, plot_type="bar", show=False)
    save_fig(outdir / "shap_summary_bar.png")

    # Dependence plots for top features by RF importance
    rf_importance_sorted = np.argsort(rf.feature_importances_)[::-1]
    top_features_idx = rf_importance_sorted[:top_n_dependence]
    for idx in top_features_idx:
        f = feature_cols[idx]
        shap.dependence_plot(f, shap_values, X_test_sample, feature_names=feature_cols, show=False)
        save_fig(outdir / f"shap_dependence_{f}.png")

    logging.info("SHAP plots generated and saved.")


def combine_rankings(lasso_df: pd.DataFrame, rf_df: pd.DataFrame, perm_df: pd.DataFrame, outdir: Path, top_k: int):
    """Combine feature importance scores into a single ranking and save plot/CSV."""
    logging.info("Combining feature importance rankings...")
    combined_df = (
        lasso_df[["feature", "lasso_importance"]]
        .merge(rf_df[["feature", "rf_importance"]], on="feature", how="outer")
        .merge(perm_df[["feature", "perm_importance_mean"]], on="feature", how="outer")
        .fillna(0.0)
    )

    combined_df["lasso_n"] = minmax_normalize(combined_df["lasso_importance"])
    combined_df["rf_n"] = minmax_normalize(combined_df["rf_importance"])
    combined_df["perm_n"] = minmax_normalize(combined_df["perm_importance_mean"])

    # Weighted score (tune weights as needed)
    combined_df["combined_score"] = (
        0.2 * combined_df["lasso_n"] +
        0.5 * combined_df["rf_n"] +
        0.3 * combined_df["perm_n"]
    )

    combined_df = combined_df.sort_values("combined_score", ascending=False)
    combined_df.to_csv(outdir / "feature_importance_combined.csv", index=False)

    # Visualization of top-k combined importance
    plt.figure(figsize=(10, 10))
    sns.barplot(
        data=combined_df.head(top_k),
        y="feature", x="combined_score", palette="crest"
    )
    plt.title("Top Combined Feature Importance Score")
    plt.xlabel("Combined (weighted & normalized) Importance")
    plt.ylabel("Feature")
    save_fig(outdir / "combined_top_features.png")

    logging.info("Combined rankings saved.")
    top_features = combined_df.head(top_k)["feature"].tolist()
    return combined_df, top_features


def model_performance_snapshot(rf, X_test: pd.DataFrame, y_test: pd.Series, outdir: Path):
    """Evaluate RF model on test set and save predicted vs. actual plot."""
    y_pred_test = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    logging.info(f"RandomForest Test R^2: {r2:.3f} | MAE: {mae:.3f}")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred_test)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Predicted vs Actual (Test Set)")
    plt.xlabel("Actual Percent Error")
    plt.ylabel("Predicted Percent Error")
    save_fig(outdir / "predicted_vs_actual.png")

    # Save metrics
    pd.DataFrame({"R2": [r2], "MAE": [mae]}).to_csv(outdir / "model_metrics.csv", index=False)


def optuna_optimization(
    rf,
    X_train: pd.DataFrame,
    top_features: list,
    outdir: Path,
    n_trials: int = 100,
):
    """Run Optuna to minimize predicted percent error by adjusting top features (scaled space)."""
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna not installed. Skipping optimization.")
        return None

    logging.info(f"Starting Optuna optimization for top {len(top_features)} features...")

    # Baseline vector in scaled space (median of training)
    baseline_vec = X_train.median()

    # Bounds from observed quantiles to stay realistic; replace with engineering bounds if available
    bounds = {}
    for f in top_features:
        low = float(X_train[f].quantile(0.05))
        high = float(X_train[f].quantile(0.95))
        if np.isclose(low, high):  # fallback to a small window
            low, high = low - 0.5, high + 0.5
        bounds[f] = (low, high)

    def objective(trial):
        x_vec = baseline_vec.copy()
        for f in top_features:
            lo, hi = bounds[f]
            x_vec[f] = trial.suggest_float(f, lo, hi)
        pred = rf.predict(x_vec.values.reshape(1, -1))[0]
        return pred

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save study results
    best_params = study.best_params
    best_value = study.best_value
    logging.info(f"Optuna best predicted percent error: {best_value:.6f}")
    pd.DataFrame([best_params]).to_csv(outdir / "optuna_best_params_scaled.csv", index=False)
    pd.DataFrame({"best_value": [best_value]}).to_csv(outdir / "optuna_best_value.csv", index=False)

    # Basic optimization history plot (matplotlib fallback)
    best_values = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(8, 4))
    plt.plot(best_values, marker="o", linestyle="-", color="blue")
    plt.title("Optuna Optimization History (Objective Values)")
    plt.xlabel("Trial")
    plt.ylabel("Predicted Percent Error")
    plt.grid(True, alpha=0.3)
    save_fig(outdir / "optuna_optimization_history.png")

    return study


def sensitivity_curves(
    rf,
    X_train: pd.DataFrame,
    top_features: list,
    outdir: Path,
    max_plots: int = 10,
):
    """One-at-a-time sensitivity curves around training quantiles for top features."""
    logging.info("Generating sensitivity curves...")
    n_sense = min(max_plots, len(top_features))
    baseline_vec = X_train.median()

    fig, axes = plt.subplots(nrows=n_sense, ncols=1, figsize=(8, 2 * n_sense), sharex=False)

    for i, f in enumerate(top_features[:n_sense]):
        lo = float(X_train[f].quantile(0.05))
        hi = float(X_train[f].quantile(0.95))
        grid = np.linspace(lo, hi, 40)

        preds = []
        x_vec = baseline_vec.copy()
        for g in grid:
            x_vec[f] = g
            preds.append(rf.predict(x_vec.values.reshape(1, -1))[0])

        ax = axes[i] if n_sense > 1 else axes
        ax.plot(grid, preds)
        ax.set_title(f"Sensitivity: {f} (scaled)")
        ax.set_xlabel(f"{f} (scaled units)")
        ax.set_ylabel("Predicted Percent Error")
        ax.grid(True, alpha=0.3)

    save_fig(outdir / "sensitivity_curves.png")
    logging.info("Sensitivity curves saved.")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Feature importance and optimization pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--target", type=str, default="percent_error", help="Target column name.")
    parser.add_argument("--outdir", type=str, default="./outputs", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size fraction.")
    parser.add_argument("--top-n", type=int, default=30, help="Top-N features to show in bar plots.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K features to select for optimization/sensitivity.")
    parser.add_argument("--perm-sample-size", type=int, default=None, help="Optional sample size for permutation importance.")
    parser.add_argument("--shap-sample-size", type=int, default=1000, help="Sample size for SHAP analysis.")
    parser.add_argument("--top-n-dependence", type=int, default=10, help="Number of SHAP dependence plots.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument("--enable-shap", action="store_true", help="Enable SHAP analysis (requires shap).")
    parser.add_argument("--enable-optuna", action="store_true", help="Enable Optuna optimization (requires optuna).")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    setup_logging(outdir)

    # Load & preprocess
    X_train, X_test, y_train, y_test, feature_cols, scaler = load_and_preprocess(
        input_path=Path(args.input),
        target_col=args.target,
        test_size=args.test_size,
    )

    # Lasso
    lasso_df = lasso_importance(
        X_train=X_train, y_train=y_train, feature_cols=feature_cols,
        outdir=outdir, top_n=args.top_n
    )

    # Random Forest
    rf, rf_df = random_forest_importance(
        X_train=X_train, y_train=y_train, feature_cols=feature_cols,
        outdir=outdir, top_n=args.top_n
    )

    # Permutation Importance
    perm_df = permutation_importance_eval(
        rf=rf, X_test=X_test, y_test=y_test, feature_cols=feature_cols,
        outdir=outdir, top_n=args.top_n, perm_sample_size=args.perm_sample_size
    )

    # SHAP (optional)
    if args.enable_shap and SHAP_AVAILABLE:
        shap_analysis(
            rf=rf, X_test=X_test, feature_cols=feature_cols,
            outdir=outdir, shap_sample_size=args.shap_sample_size,
            top_n_dependence=args.top_n_dependence
        )
    elif args.enable_shap and not SHAP_AVAILABLE:
        logging.warning("SHAP requested but not installed. Run `pip install shap` to enable.")

    # Combine rankings & select top-K features
    combined_df, top_features = combine_rankings(
        lasso_df=lasso_df, rf_df=rf_df, perm_df=perm_df,
        outdir=outdir, top_k=args.top_k
    )

    # Model performance snapshot
    model_performance_snapshot(rf=rf, X_test=X_test, y_test=y_test, outdir=outdir)

    # Optimization (optional)
    if args.enable_optuna and OPTUNA_AVAILABLE:
        optuna_optimization(
            rf=rf, X_train=X_train, top_features=top_features,
            outdir=outdir, n_trials=args.n_trials
        )
    elif args.enable_optuna and not OPTUNA_AVAILABLE:
        logging.warning("Optuna requested but not installed. Run `pip install optuna` to enable.")

    # Sensitivity curves
    sensitivity_curves(
        rf=rf, X_train=X_train, top_features=top_features,
        outdir=outdir, max_plots=min(10, len(top_features))
    )

    logging.info("Pipeline completed.")


if __name__ == "__main__":
    main()
