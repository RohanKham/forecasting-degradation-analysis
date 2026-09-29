"""
Shared residual-analysis plotting script.

Works on the residual_export.csv (or .parquet) produced by either
residual_LSTM.py or residual_MLPLSTM.py -- both save the same schema:
    true_<target>, pred_<target>, residual_<target> (= pred - true),
    Irr, temp_C, humidity, precip_mm, precip_indicator, cloud_cover,
    hour_sin, hour_cos, month_sin, month_cos, doy_sin, doy_cos,
    weekday_sin, weekday_cos, days_since_install_<family>,
    bad_day_<target>, _split, pred_<target>_age1

Produces 9 figures per target:
  1. Pred vs True scatter, with R^2 annotated (no 45-degree reference line)
  2. Residual over time (every timestamp, raw scatter/line)
  3. Residual over time, one box per month
  4. Residual vs each environment variable (one subplot each)
  5. First 14 days: true, age=1 predicted, and normal predicted power (lines)
  6. (age1 - true) and (age1 - model pred) over time, vertically stacked
  7. age=1 prediction vs each environment variable (scatter subplots)
  8. Box plots of age=1 prediction per bin of Irr, and per bin of temp_C
  9. age=1 prediction over time near STC (Irr 1000+-50, temp 25+-5, no rain),
     coloured by humidity

Usage:
    python plot_residuals.py --csv /path/to/residual_export.csv --output_dir ./plots
or edit CSV_PATH / OUTPUT_DIR in main() and run directly.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ENV_VARS = ["Irr", "temp_C", "cloud_cover", "humidity", "precip_mm", "precip_indicator"]
BOX_BIN_VARS = ["Irr", "temp_C"]


def load_residual_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=["timestamp"])
    df = df.sort_index()
    print(f"Loaded {csv_path}")
    print(f"  rows: {len(df):,}")
    print(f"  date range: {df.index.min()} -> {df.index.max()}")
    print(f"  columns: {list(df.columns)}")
    return df


def detect_targets(df: pd.DataFrame) -> list[str]:
    """Detect target names from residual_<target> columns."""
    targets = [c.replace("residual_", "") for c in df.columns if c.startswith("residual_")]
    print(f"  Detected targets: {targets}")
    return targets


def apply_bad_day_mask(df: pd.DataFrame, target: str, exclude_bad_days: bool) -> pd.DataFrame:
    """
    Returns a filtered copy of df with bad-day rows and NaN true/pred rows
    dropped, if exclude_bad_days is True. Bad-day column is looked up as
    bad_day_<target> or, failing that, any single bad_day_* column present.
    """
    true_col = f"true_{target}"
    pred_col = f"pred_{target}"

    valid = df[true_col].notna() & df[pred_col].notna()

    if exclude_bad_days:
        bad_col = f"bad_day_{target}"
        if bad_col not in df.columns:
            candidates = [c for c in df.columns if c.startswith("bad_day_")]
            bad_col = candidates[0] if len(candidates) == 1 else None
        if bad_col is not None and bad_col in df.columns:
            valid = valid & (df[bad_col] < 0.5)
            print(f"  Excluding bad days using column: {bad_col}")
        else:
            print(f"  WARNING: could not resolve a bad_day column for target '{target}', "
                  f"not excluding any bad days.")

    out = df.loc[valid].copy()
    print(f"  {target}: {len(out):,} / {len(df):,} rows kept after filtering")
    return out


# --------------------------------------------------------------------------
# Plot 1: Pred vs True scatter with R^2 (no 45-degree line)
# --------------------------------------------------------------------------
def plot_pred_vs_true(df: pd.DataFrame, target: str, output_dir: Path, split: str = "all"):
    true_col = f"true_{target}"
    pred_col = f"pred_{target}"

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=4, alpha=0.35, color="tab:blue")

    ax.set_xlabel(f"True {target} (W)")
    ax.set_ylabel(f"Predicted {target} (W)")
    ax.set_title(f"{target}: Predicted vs True ({split})  n={len(df):,}")
    ax.text(
        0.02, 0.98, f"R² = {r2:.4f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=13,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_pred_vs_true.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")
    return r2


# --------------------------------------------------------------------------
# Plot 2: Residual over time (every timestamp)
# --------------------------------------------------------------------------
def plot_residual_over_time(df: pd.DataFrame, target: str, output_dir: Path, split: str = "all"):
    resid_col = f"residual_{target}"

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(df.index, df[resid_col], s=2, alpha=0.3, color="tab:red")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    mean_resid = df[resid_col].mean()
    ax.axhline(mean_resid, color="green", linewidth=1.5,
               label=f"mean = {mean_resid:+.3f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (pred - true), W")
    ax.set_title(f"{target}: Residual over time ({split})  n={len(df):,}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_residual_over_time.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 3: Residual over time, box plot per month
# --------------------------------------------------------------------------
def plot_residual_monthly_boxplot(df: pd.DataFrame, target: str, output_dir: Path, split: str = "all"):
    resid_col = f"residual_{target}"

    sub = df[[resid_col]].dropna().copy()
    idx = sub.index.tz_localize(None) if sub.index.tz is not None else sub.index
    sub["year_month"] = pd.PeriodIndex(idx, freq="M").astype(str).values

    months = sorted(sub["year_month"].unique())
    data_per_month = [sub.loc[sub["year_month"] == m, resid_col].values for m in months]

    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.6), 6))
    bp = ax.boxplot(data_per_month, labels=months, showfliers=True, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Month")
    ax.set_ylabel("Residual (pred - true), W")
    ax.set_title(f"{target}: Monthly residual distribution ({split})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_residual_monthly_boxplot.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 4: Residual vs each environment variable, one figure, subplots
# --------------------------------------------------------------------------
def plot_residual_vs_env_vars(
    df: pd.DataFrame, target: str, output_dir: Path, split: str = "all",
    env_vars: list[str] = ENV_VARS,
):
    resid_col = f"residual_{target}"

    available_vars = [v for v in env_vars if v in df.columns]
    missing_vars = [v for v in env_vars if v not in df.columns]
    if missing_vars:
        print(f"  WARNING: missing env columns, skipping: {missing_vars}")

    n = len(available_vars)
    if n == 0:
        print("  WARNING: no environment variables available to plot, skipping plot 4.")
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, var in enumerate(available_vars):
        ax = axes[i]
        sub = df[[var, resid_col]].dropna()

        ax.scatter(sub[var], sub[resid_col], s=3, alpha=0.3, color="tab:purple")
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

        if len(sub) > 1:
            corr = np.corrcoef(sub[var], sub[resid_col])[0, 1]
            ax.text(
                0.02, 0.98, f"corr = {corr:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(var)
        ax.set_ylabel("Residual (pred - true), W")
        ax.set_title(f"Residual vs {var}")
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{target}: Residual vs environment variables ({split})  n={len(df):,}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fpath = output_dir / f"{target}_{split}_residual_vs_env_vars.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Helper: resolve the age=1 column, return None if not present
# --------------------------------------------------------------------------
def get_age1_col(df: pd.DataFrame, target: str):
    col = f"pred_{target}_age1"
    if col not in df.columns:
        print(f"  WARNING: column '{col}' not found, skipping age=1 plots for '{target}'.")
        return None
    if df[col].notna().sum() == 0:
        print(f"  WARNING: column '{col}' is all NaN, skipping age=1 plots for '{target}'.")
        return None
    return col


# --------------------------------------------------------------------------
# Plot 5: first 14 days -- true vs age=1 pred vs normal pred (continuous)
# --------------------------------------------------------------------------
def plot_first_14_days(df: pd.DataFrame, target: str, output_dir: Path, split: str = "all",
                       n_days: int = 40):
    age1_col = get_age1_col(df, target)
    if age1_col is None:
        return
    true_col, pred_col = f"true_{target}", f"pred_{target}"

    start = df.index.min()
    window = df.loc[start: start + pd.Timedelta(days=n_days), [true_col, pred_col, age1_col]]
    if len(window) < 2:
        print("  WARNING: not enough rows in first 14 days, skipping plot 5.")
        return

    # Insert NaN rows for gaps (e.g. removed bad days) so lines are not drawn across them
    step = window.index.to_series().diff().median()
    if pd.notna(step) and step > pd.Timedelta(0):
        window = window.asfreq(step)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(window.index, window[true_col], color="black", linewidth=1.2, label="True")
    ax.plot(window.index, window[age1_col], color="tab:orange", linewidth=1.0,
            alpha=0.9, label="Predicted (age=1)")
    ax.plot(window.index, window[pred_col], color="tab:blue", linewidth=1.0,
            alpha=0.9, label="Predicted (model)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"{target}: First {n_days} days ({window.index.min():%Y-%m-%d} -> "
                 f"{window.index.max():%Y-%m-%d}), {split}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_first{n_days}days_power.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 6: (age1 - true) and (age1 - model pred) over time, vertically stacked
# --------------------------------------------------------------------------
def plot_age1_differences_over_time(df: pd.DataFrame, target: str, output_dir: Path,
                                    split: str = "all"):
    age1_col = get_age1_col(df, target)
    if age1_col is None:
        return
    true_col, pred_col = f"true_{target}", f"pred_{target}"

    sub = df[[true_col, pred_col, age1_col]].dropna()
    diff_true = sub[age1_col] - sub[true_col]
    diff_model = sub[age1_col] - sub[pred_col]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    for ax, diff, color, title in [
        (axes[0], diff_true, "tab:red", "age=1 - true"),
        (axes[1], diff_model, "tab:green", "age=1 - model pred"),
    ]:
        ax.scatter(sub.index, diff, s=2, alpha=0.3, color=color)
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        m = diff.mean()
        ax.axhline(m, color="blue", linewidth=1.5, label=f"mean = {m:+.3f}")
        ax.set_ylabel(f"{title} (W)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Time")
    fig.suptitle(f"{target}: age=1 differences over time ({split})  n={len(sub):,}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fpath = output_dir / f"{target}_{split}_age1_differences_over_time.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 7: age=1 prediction vs each environment variable (scatter subplots)
# --------------------------------------------------------------------------
def plot_age1_vs_env_vars(df: pd.DataFrame, target: str, output_dir: Path, split: str = "all",
                          env_vars: list[str] = ENV_VARS):
    age1_col = get_age1_col(df, target)
    if age1_col is None:
        return

    available_vars = [v for v in env_vars if v in df.columns]
    missing_vars = [v for v in env_vars if v not in df.columns]
    if missing_vars:
        print(f"  WARNING: missing env columns, skipping: {missing_vars}")
    n = len(available_vars)
    if n == 0:
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, var in enumerate(available_vars):
        ax = axes[i]
        sub = df[[var, age1_col]].dropna()
        ax.scatter(sub[var], sub[age1_col], s=3, alpha=0.3, color="tab:orange")

        if len(sub) > 1 and sub[var].std() > 0:
            corr = np.corrcoef(sub[var], sub[age1_col])[0, 1]
            ax.text(0.02, 0.98, f"corr = {corr:.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel(var)
        ax.set_ylabel(f"Predicted {target} age=1 (W)")
        ax.set_title(f"age=1 vs {var}")
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{target}: age=1 prediction vs environment variables ({split})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fpath = output_dir / f"{target}_{split}_age1_vs_env_vars.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 8: box plot of age=1 prediction per bin of a variable (Irr, temp_C)
# --------------------------------------------------------------------------
def plot_age1_binned_boxplot(df: pd.DataFrame, target: str, var: str, output_dir: Path,
                             split: str = "all", n_bins: int = 15):
    age1_col = get_age1_col(df, target)
    if age1_col is None:
        return
    if var not in df.columns:
        print(f"  WARNING: column '{var}' missing, skipping binned boxplot.")
        return

    sub = df[[var, age1_col]].dropna().copy()
    if len(sub) == 0 or sub[var].nunique() < 2:
        print(f"  WARNING: not enough data in '{var}' for binned boxplot.")
        return

    sub["bin"] = pd.cut(sub[var], bins=n_bins)
    groups = [(b, g[age1_col].values) for b, g in sub.groupby("bin", observed=True) if len(g) > 0]
    labels = [f"{b.left:.0f}–{b.right:.0f}" if sub[var].abs().max() > 50
              else f"{b.left:.1f}–{b.right:.1f}" for b, _ in groups]
    counts = [len(v) for _, v in groups]
    labels = [f"{l}\n(n={c})" for l, c in zip(labels, counts)]

    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 0.9), 6.5))
    bp = ax.boxplot([v for _, v in groups], showfliers=True, patch_artist=True)
    ax.set_xticklabels(labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("moccasin")
        patch.set_alpha(0.8)

    ax.set_xlabel(f"{var} (binned)")
    ax.set_ylabel(f"Predicted {target} age=1 (W)")
    ax.set_title(f"{target}: age=1 prediction per {var} bin ({split})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_age1_boxplot_by_{var}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Plot 9: age=1 prediction near STC over time, coloured by humidity
# --------------------------------------------------------------------------
def plot_age1_near_stc(
    df: pd.DataFrame, target: str, output_dir: Path, split: str = "all",
    irr_center: float = 1000.0, irr_tol: float = 100.0,
    temp_center: float = 25.0, temp_tol: float = 10.0,
):
    age1_col = get_age1_col(df, target)
    if age1_col is None:
        return

    required = ["Irr", "temp_C", "precip_indicator", "humidity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing}, skipping near-STC plot.")
        return

    mask = (
        (df["Irr"].sub(irr_center).abs() <= irr_tol)
        & (df["temp_C"].sub(temp_center).abs() <= temp_tol)
        & (df["precip_indicator"] == 0)
    )
    sub = df.loc[mask, [age1_col, "humidity"]].dropna()
    print(f"  Near-STC rows: {len(sub):,} / {len(df):,}")
    if len(sub) == 0:
        print("  WARNING: no near-STC rows, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    sc = ax.scatter(sub.index, sub[age1_col], c=sub["humidity"], s=12, alpha=0.8,
                    cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Humidity")

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Predicted {target} age=1 (W)")
    ax.set_title(
        f"{target}: age=1 near STC ({split})  n={len(sub):,}\n"
        f"Irr = {irr_center:.0f} ± {irr_tol:.0f} W/m², "
        f"temp = {temp_center:.0f} ± {temp_tol:.0f} °C, no rain"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fpath = output_dir / f"{target}_{split}_age1_near_STC.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_all_plots(
    csv_path: str,
    output_dir: str = "./residual_plots",
    split: str = "all",
    exclude_bad_days: bool = True,
    env_vars: list[str] = ENV_VARS,
):
    """
    split: "all", "train", or "test" -- filters the _split column if present
    and split != "all".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_residual_csv(csv_path)
    targets = detect_targets(df)

    if split != "all":
        if "_split" not in df.columns:
            print(f"  WARNING: split='{split}' requested but no '_split' column found, using all rows.")
        else:
            df = df[df["_split"] == split].copy()
            print(f"  Filtered to split='{split}': {len(df):,} rows")

    for target in targets:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target}  (split={split})")
        print(f"{'=' * 70}")

        sub = apply_bad_day_mask(df, target, exclude_bad_days=exclude_bad_days)

        if len(sub) == 0:
            print(f"  WARNING: no valid rows for target '{target}' after filtering, skipping plots.")
            continue

        r2 = plot_pred_vs_true(sub, target, output_dir, split=split)
        plot_residual_over_time(sub, target, output_dir, split=split)
        plot_residual_monthly_boxplot(sub, target, output_dir, split=split)
        plot_residual_vs_env_vars(sub, target, output_dir, split=split, env_vars=env_vars)

        # ---- age=1 plots ----
        plot_first_14_days(sub, target, output_dir, split=split)
        plot_age1_differences_over_time(sub, target, output_dir, split=split)
        plot_age1_vs_env_vars(sub, target, output_dir, split=split, env_vars=env_vars)
        for var in BOX_BIN_VARS:
            plot_age1_binned_boxplot(sub, target, var, output_dir, split=split)
        plot_age1_near_stc(sub, target, output_dir, split=split)

        print(f"\n  {target} summary (split={split}):")
        print(f"    R²                 : {r2:.4f}")
        print(f"    Residual mean      : {sub[f'residual_{target}'].mean():+.3f} W")
        print(f"    Residual std       : {sub[f'residual_{target}'].std():.3f} W")


def main():
    CSV_PATH = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_26_133601/lstm_results/residual_export/residual_export.csv"
    OUTPUT_DIR = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_26_133601/lstm_results/residual_plots"
    
    run_all_plots(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        split="all",             # "all", "train", or "test"
        exclude_bad_days=True,
        )

if __name__ == "__main__":
    main()