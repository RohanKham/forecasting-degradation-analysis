import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from scipy.stats import linregress, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PANEL_AREA = 1.44 # active panel area in m*m  (psc: 0.7 | solon: 1.44 | sanyo: 1.125)
DAY_START_HOUR = 4 
DAY_END_HOUR = 22
MIN_IRR_FOR_PCE = 300.0 
MIN_POWER_FOR_PCE = 20 
FREQ = "10min" # expected time series frequency

TEMP_COL = "temp_C" # temperature column to use: "temp_C" (ambient) or "panel_temp"
TEMP_LABEL = "Ambient Temperature (°C)" if TEMP_COL == "temp_C" else "Panel Temp (°C)"

def filter_daylight_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only rows within the daylight window [DAY_START_HOUR, DAY_END_HOUR)
    """
    if df.empty:
        return df
    hours = df.index.hour
    mask = (hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR)
    out = df[mask].copy()
    print(f"Daylight filter ({DAY_START_HOUR}:00-{DAY_END_HOUR}:00): "
          f"kept {len(out)}/{len(df)} rows ({100*len(out)/len(df):.1f}%)")
    return out


def filter_bad_days(df: pd.DataFrame, bad_day_list: list) -> pd.DataFrame:
    """
    Remove all rows belonging to manually flagged bad days
    """
    if not bad_day_list or df.empty:
        return df
    bad_dates = pd.to_datetime(bad_day_list).normalize()
    mask = ~df["date"].isin(bad_dates)
    out = df[mask].copy()
    print(f"Bad day filter: removed {len(df)-len(out)} rows from {len(bad_dates)} days")
    return out


def merge_train_test_parquets(train_path: Path, test_path: Path) -> pd.DataFrame:
    """
    Load train and test parquet splits
    """
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")

    train_df = pd.read_parquet(train_path).copy()
    test_df = pd.read_parquet(test_path).copy()
    train_df["_split"] = "train"
    test_df["_split"] = "test"

    merged = pd.concat([train_df, test_df]).sort_index()
    dup = merged.index.duplicated(keep=False)
    if dup.any():
        print(f"Removing {dup.sum()} duplicate timestamps")
        merged = merged[~merged.index.duplicated(keep="first")]

    print(f"Train rows : {(merged['_split']=='train').sum():,}")
    print(f"Test rows  : {(merged['_split']=='test').sum():,}")
    print(f"Date range : {merged.index.min().date()} → {merged.index.max().date()}")
    return merged


def run_inference(model, dataset, device, batch_size: int = 256) -> np.ndarray:
    """
    Returns array of shape (n_samples, horizon, n_outputs).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_x = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
            preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _inverse_transform(arr, scaler) -> np.ndarray:
    """Invert MinMax scaling"""
    arr = np.asarray(arr).ravel()
    if scaler is None:
        return arr.copy()
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def build_aligned_df(
    dataset, preds_scaled, df_scaled, info,
    target_scalers, feature_scalers, bad_days_list=None,
) -> pd.DataFrame:
    target_cols = info["data_config"]["features"]["target_cols"]
    all_feat = info["data_config"]["features"]["feature_cols"]
    horizon = info["data_config"]["window_horizon"]["horizon"]
    window = info["data_config"]["window_horizon"]["window"]

    # Inverse-transform all feature columns to real units
    feat_real = {}
    for col in all_feat:
        scaler = feature_scalers.get(col)
        feat_real[col] = pd.Series(
            _inverse_transform(df_scaled[col].values, scaler),
            index=df_scaled.index,
        )

    bad_indices: set = set()
    if hasattr(dataset, "bad_day_mask"):
        bad_indices = {i for i, b in enumerate(dataset.bad_day_mask) if b}

    # Build one record per horizon step per sequence
    records = []
    for i in range(len(dataset)):
        if i in bad_indices:
            continue
        sample = dataset[i]
        y_true = sample[1]
        y_pred = preds_scaled[i]

        if hasattr(dataset, "get_target_index"):
            ts_index = dataset.get_target_index(i)
        else:
            start_row = window + i * horizon
            end_row = start_row + horizon
            if end_row > len(df_scaled):
                continue
            ts_index = df_scaled.index[start_row:end_row]

        if len(ts_index) != horizon:
            continue

        for h in range(horizon):
            ts = ts_index[h]
            row = {"timestamp": ts}
            for t_idx, tname in enumerate(target_cols):
                tsc = target_scalers.get(tname)
                row[f"true_{tname}"] = float(_inverse_transform(np.array([y_true[h, t_idx]]), tsc)[0])
                row[f"pred_{tname}"] = float(_inverse_transform(np.array([y_pred[h, t_idx]]), tsc)[0])
            records.append(row)

    df = pd.DataFrame(records).set_index("timestamp").sort_index()

    # Attach inverse-transformed feature columns
    for col, series in feat_real.items():
        df[col] = series.reindex(df.index)

    if "_split" in df_scaled.columns:
        df["_split"] = df_scaled["_split"].reindex(df.index)

    df["date"] = df.index.normalize()
    df = df.resample(FREQ).first()
    df["date"] = df.index.normalize()
    df["dt_hours"] = 10.0 / 60.0  # 10-min step expressed in hours for energy integration

    if bad_days_list:
        df = filter_bad_days(df, bad_days_list)
    df = filter_daylight_hours(df)

    # Drop rows where irradiance or ambient temperature is missing or zero
    df = df[df["Irr"].notna() & df["temp_C"].notna() & (df["Irr"] > 0)].copy()
    return df

def _resolve_temp_col(df: pd.DataFrame) -> tuple[str, str]:
    if TEMP_COL == "panel_temp":
        if "panel_temp" in df.columns and df["panel_temp"].notna().sum() > len(df) * 0.1:
            return "panel_temp", "Panel Temp (°C)"
        else:
            print("WARNING: TEMP_COL='panel_temp' but column is missing or "
                  "mostly NaN, falling back to 'temp_C'.")
            return "temp_C", "Ambient Temp (°C)"
    return "temp_C", "Ambient Temp (°C)"

def calculate_pce(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Power Conversion Efficiency (PCE) for true and predicted power
    """
    true_cols = [c for c in df.columns if c.startswith("true_")]
    target_name = true_cols[0].replace("true_", "")
    print(f"Using target: {target_name}")

    temp_col, temp_label = _resolve_temp_col(df)
    print(f"Using temperature column: '{temp_col}' ({temp_label})")

    denom = df["Irr"] * PANEL_AREA
    power_true = df[f"true_{target_name}"]
    power_pred = df[f"pred_{target_name}"]

    base_valid = (
        df["Irr"].notna() &
        (df["Irr"] >= MIN_IRR_FOR_PCE) &
        df[temp_col].notna() &
        power_true.notna() & (power_true >= MIN_POWER_FOR_PCE) &
        power_pred.notna() & (power_pred >= MIN_POWER_FOR_PCE)
    )
    print(f"Shared valid timestamps (Irr >= {MIN_IRR_FOR_PCE}, "
          f"P >= {MIN_POWER_FOR_PCE} for both): {base_valid.sum()}/{len(df)}")

    df["pce_true"] = np.nan
    df["pce_pred"] = np.nan
    df.loc[base_valid, "pce_true"] = power_true[base_valid] / denom[base_valid]
    df.loc[base_valid, "pce_pred"] = power_pred[base_valid] / denom[base_valid]

    # Zero PCE is physically meaningless
    df.loc[df["pce_true"] == 0, "pce_true"] = np.nan
    df.loc[df["pce_pred"] == 0, "pce_pred"] = np.nan

    print(f"\nPCE (before outlier filter):")
    print(f"pce_true: min={df['pce_true'].min():.4f}, "
          f"max={df['pce_true'].max():.4f}, valid={df['pce_true'].notna().sum()}")
    print(f"pce_pred: min={df['pce_pred'].min():.4f}, "
          f"max={df['pce_pred'].max():.4f}, valid={df['pce_pred'].notna().sum()}")

    count_t_before = df["pce_true"].notna().sum()
    count_p_before = df["pce_pred"].notna().sum()

    # Remove top 1% outliers to suppress sensor spikes
    p99_true = df.loc[df["pce_true"].notna(), "pce_true"].quantile(0.99)
    p99_pred = df.loc[df["pce_pred"].notna(), "pce_pred"].quantile(0.99)
    df.loc[(df["pce_true"] > p99_true) & df["pce_true"].notna(), "pce_true"] = np.nan
    df.loc[(df["pce_pred"] > p99_pred) & df["pce_pred"].notna(), "pce_pred"] = np.nan

    removed_t = count_t_before - df["pce_true"].notna().sum()
    removed_p = count_p_before - df["pce_pred"].notna().sum()
    print(f"\nAfter 99th-pct filter: removed {removed_t} true / {removed_p} pred outliers")

    # Convert to percentage
    df["pce_true"] *= 100
    df["pce_pred"] *= 100

    print(f"\nFINAL PCE (%):")
    print(f"pce_true: min={df['pce_true'].min():.2f}%, "
          f"max={df['pce_true'].max():.2f}%, valid={df['pce_true'].notna().sum()}")
    print(f"pce_pred: min={df['pce_pred'].min():.2f}%, "
          f"max={df['pce_pred'].max():.2f}%, valid={df['pce_pred'].notna().sum()}")

    overlap = (df["pce_true"].notna() & df["pce_pred"].notna()).sum()
    print(f"Final overlap (timestamps with both valid): {overlap}")

    df.attrs["target_name"] = target_name
    return df

def _get_target_name(df: pd.DataFrame) -> str:
    """Extract target module name"""
    if "target_name" in df.attrs:
        return df.attrs["target_name"]
    true_cols = [c for c in df.columns if c.startswith("true_")]
    if true_cols:
        return true_cols[0].replace("true_", "")
    return "Power"


def print_dataframe_stats(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("ALIGNED DATAFRAME STATISTICS")
    print(f"{'='*70}")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Days covered: {df['date'].nunique()}")
    print(f"Frequency: {FREQ}")
    print(f"Temperature: using '{TEMP_COL}' ({TEMP_LABEL})")

    print(f"\n{'Column':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 57)
    for col in ["Irr", "temp_C", "panel_temp"]:
        if col in df.columns and df[col].notna().any():
            print(f"{col:<15} {df[col].min():>10.2f} {df[col].max():>10.2f} "
                  f"{df[col].mean():>10.2f} {df[col].std():>10.2f}")

def plot_temp_correlation(df: pd.DataFrame, output_dir: Path, module_name: str = "Module"):
    """
    Produce two scatter plots showing irradiance vs panel temperature and irradiance vs ambient temperature, with Pearson correlation.
    """
    target = _get_target_name(df)

    if "temp_C" not in df.columns:
        print("WARNING: temp_C column not found in DataFrame")
        return
    if "panel_temp" not in df.columns:
        print("WARNING: panel_temp column not found in DataFrame")
        return

    valid = (
        df["temp_C"].notna() &
        df["panel_temp"].notna() &
        (df["Irr"] > 0)
    )
    if not valid.any():
        print("WARNING: No valid data points for temperature comparison")
        return

    df_valid = df[valid].copy()
    corr_panel = np.corrcoef(df_valid["Irr"], df_valid["panel_temp"])[0, 1]
    corr_ambient = np.corrcoef(df_valid["Irr"], df_valid["temp_C"])[0, 1]

    print(f"\nTemperature Correlation Analysis for {module_name}:")
    print(f"{'='*50}")
    print(f"Correlation (Panel vs Irradiance): r = {corr_panel:.4f}")
    print(f"Correlation (Ambient vs Irradiance): r = {corr_ambient:.4f}")
    print(f"Difference: {corr_panel - corr_ambient:.4f}")

    # Panel temperature scatter
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(df_valid["Irr"], df_valid["panel_temp"],
                color="orange", s=15, alpha=0.5, edgecolors="none", rasterized=True)
    ax1.set_xlabel("Irradiance (W/m²)", fontsize=14, fontweight="bold", labelpad=20)
    ax1.set_ylabel(f"{module_name} Panel Temperature (°C)", fontsize=14, fontweight="bold", labelpad=10)
    ax1.set_title(f"{module_name} Panel Temperature vs Irradiance", fontsize=24, fontweight="bold", pad=15)
    ax1.legend([f"r = {corr_panel:.3f}"], loc="upper left", fontsize=14, framealpha=0.9)
    ax1.grid(False, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=14)
    plt.tight_layout(pad=1.5)
    panel_filename = f"panel_temp_vs_irradiance_{module_name}.png"
    fig1.savefig(output_dir / panel_filename, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {panel_filename}")

    # Ambient temperature scatter
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.scatter(df_valid["Irr"], df_valid["temp_C"],
                color="steelblue", s=15, alpha=0.5, edgecolors="none", rasterized=True)
    ax2.set_xlabel("Irradiance (W/m²)", fontsize=14, fontweight="bold", labelpad=20)
    ax2.set_ylabel("Ambient Temperature (°C)", fontsize=14, fontweight="bold", labelpad=10)
    ax2.set_title("Ambient Temperature vs Irradiance", fontsize=24, fontweight="bold", pad=15)
    ax2.legend([f"r = {corr_ambient:.3f}"], loc="upper left", fontsize=14, framealpha=0.9)
    ax2.grid(False, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=14)
    plt.tight_layout(pad=1.5)
    ambient_filename = "ambient_temp_vs_irradiance.png"
    fig2.savefig(output_dir / ambient_filename, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {ambient_filename}")

def plot_pce_temperature_dependence(df: pd.DataFrame, output_dir: Path):
    """
    Scatter PCE vs temperature for both true and predicted power.
    """
    target = _get_target_name(df)
    temp_col, temp_label = _resolve_temp_col(df)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PCE Temperature Dependence  [Target: {target}]  [{temp_label}]",
                 fontsize=14, fontweight="bold")

    for ax, col, color, title in [
        (ax_t, "pce_true", "b", f"True (Measured): {target}"),
        (ax_p, "pce_pred", "r", f"LSTM (Predicted): {target}"),
    ]:
        valid = df[col].notna() & (df[col] > 0) & df[temp_col].notna()
        sub = df.loc[valid].copy()
        sc = ax.scatter(sub[temp_col], sub[col], c=sub["Irr"],
                        cmap="viridis", alpha=0.6, s=20)
        slope, intercept, r, *_ = linregress(sub[temp_col], sub[col])
        x_range = np.array([sub[temp_col].min(), sub[temp_col].max()])
        ax.plot(x_range, intercept + slope * x_range, f"{color}-", linewidth=2,
                label=f"OLS: {intercept:.2f} + {slope:.3f}*T  (R²={r**2:.3f})")
        ax.set_xlabel(temp_label)
        ax.set_ylabel("PCE (%)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Irradiance (W/m²)")
        print(f"{title}: slope={slope:.3f}%/°C, R²={r**2:.3f}")

    plt.tight_layout()
    plt.savefig(output_dir / "pce_temperature_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pce_temperature_dependence.png")

def plot_pce_irradiance_dependence(df: pd.DataFrame, output_dir: Path):
    """
    Scatter PCE vs irradiance for both true and predicted power.
    """
    target = _get_target_name(df)
    temp_col, temp_label = _resolve_temp_col(df)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PCE Irradiance Dependence  [Target: {target}]",
                 fontsize=14, fontweight="bold")

    for ax, col, color, title in [
        (ax_t, "pce_true", "b", f"True (Measured): {target}"),
        (ax_p, "pce_pred", "r", f"LSTM (Predicted): {target}"),
    ]:
        valid = df[col].notna() & (df[col] > 0)
        sub = df.loc[valid].copy()
        sc = ax.scatter(sub["Irr"], sub[col], c=sub[temp_col],
                        cmap="coolwarm", alpha=0.6, s=20)
        slope, intercept, r, *_ = linregress(sub["Irr"], sub[col])
        x_range = np.array([sub["Irr"].min(), sub["Irr"].max()])
        ax.plot(x_range, intercept + slope * x_range, f"{color}-", linewidth=2,
                label=f"OLS: {intercept:.2f} + {slope:.4f}*Irr (R²={r**2:.3f})")
        ax.set_xlabel("Irradiance (W/m²)")
        ax.set_ylabel("PCE (%)")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label=temp_label)
        print(f"{title}: slope={slope:.4f}%/(W/m²), R²={r**2:.3f}")

    plt.tight_layout()
    plt.savefig(output_dir / "pce_irradiance_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pce_irradiance_dependence.png")

def build_reference_surface(
    df: pd.DataFrame,
    ref_days: int = 120,
    n_irr_bins: int = 10,
    n_temp_bins: int = 10,
    min_irr_ref: float = MIN_IRR_FOR_PCE,
    percentile: float = 75,
) -> dict:
    """
    Build a 2D reference PCE lookup table in (Irradiance * Temperature) space using the first 120 days of data.
    """
    temp_col, temp_label = _resolve_temp_col(df)

    print(f"\n{'='*70}")
    print(f"BUILDING REFERENCE SURFACE, "
          f"p{percentile:.0f})  [T = {temp_label}]")
    print(f"{'='*70}")

    all_dates = sorted(df["date"].unique())
    ref_dates = all_dates[:ref_days]

    df_ref = df[
        df["date"].isin(ref_dates) &
        (df["Irr"] >= min_irr_ref) &
        df["pce_true"].notna() & (df["pce_true"] > 0) &
        df[temp_col].notna()
    ].copy()

    print(f"Reference period: {ref_dates[0].date()} -> {ref_dates[-1].date()}")
    print(f"Reference points: {len(df_ref)} (Irr >= {min_irr_ref:.0f} W/m²)")

    if len(df_ref) < 20:
        raise ValueError(f"Too few reference points above min_irr_ref={min_irr_ref} W/m²")

    G_ref = df_ref["Irr"].values
    T_ref = df_ref[temp_col].values
    y_ref = df_ref["pce_true"].values

    # Fit OLS on median-aggregated bin values
    _irr_e = np.linspace(G_ref.min(), G_ref.max(), 16)
    _tmp_e = np.linspace(T_ref.min(), T_ref.max(), 11)
    _irr_c = (_irr_e[:-1] + _irr_e[1:]) / 2
    _tmp_c = (_tmp_e[:-1] + _tmp_e[1:]) / 2
    _tmp_df = pd.DataFrame({"G": G_ref, "T": T_ref, "y": y_ref})
    _tmp_df["ib"] = pd.cut(_tmp_df["G"], bins=_irr_e, labels=False)
    _tmp_df["tb"] = pd.cut(_tmp_df["T"], bins=_tmp_e, labels=False)
    _agg = _tmp_df.groupby(["ib", "tb"])["y"].median().reset_index().dropna()

    if len(_agg) >= 4:
        G_fit = _irr_c[_agg["ib"].astype(int).values]
        T_fit = _tmp_c[_agg["tb"].astype(int).values]
        y_fit = _agg["y"].values
    else:
        G_fit, T_fit, y_fit = G_ref, T_ref, y_ref

    lm_diag = LinearRegression().fit(np.column_stack([G_fit, T_fit]), y_fit)
    alpha = lm_diag.intercept_
    beta_G = lm_diag.coef_[0]
    beta_T = lm_diag.coef_[1]
    r2_diag = r2_score(y_ref, lm_diag.predict(np.column_stack([G_ref, T_ref])))
    print(f"Linear model : eta = {alpha:.3f} + {beta_G:.5f}*G + {beta_T:.4f}*T  "
          f"[{temp_label}]")
    print(f"R2 on ref pts: {r2_diag:.4f}")

    # Define bin edges spanning the full dataset range
    irr_min = MIN_IRR_FOR_PCE
    irr_max = np.ceil(df["Irr"].max() / 50) * 50
    temp_min = np.floor(df[temp_col].min() / 2) * 2
    temp_max = np.ceil(df[temp_col].max() / 2) * 2

    irr_bins = np.linspace(irr_min, irr_max, n_irr_bins + 1)
    temp_bins = np.linspace(temp_min, temp_max, n_temp_bins + 1)
    irr_centres = (irr_bins[:-1] + irr_bins[1:]) / 2
    temp_centres = (temp_bins[:-1] + temp_bins[1:]) / 2

    all_irr_idx = list(range(n_irr_bins))
    all_temp_idx = list(range(n_temp_bins))

    df_ref["irr_bin"] = pd.cut(df_ref["Irr"], bins=irr_bins, labels=False)
    df_ref["temp_bin"] = pd.cut(df_ref[temp_col], bins=temp_bins, labels=False)

    # Aggregate reference PCE at the chosen percentile per bin
    lookup_obs = (
        df_ref
        .groupby(["irr_bin", "temp_bin"])["pce_true"]
        .quantile(percentile / 100.0)
        .unstack()
        .reindex(index=all_irr_idx, columns=all_temp_idx)
    )
    lookup = lookup_obs.copy()

    lookup_counts = (
        df_ref
        .groupby(["irr_bin", "temp_bin"])["pce_true"]
        .count()
        .unstack()
        .reindex(index=all_irr_idx, columns=all_temp_idx)
    )

    # Full-dataset bin counts (used for coverage heatmap)
    df_full = df.copy()
    df_full["irr_bin"] = pd.cut(df_full["Irr"], bins=irr_bins, labels=False)
    df_full["temp_bin"] = pd.cut(df_full[temp_col], bins=temp_bins, labels=False)
    lookup_counts_full = (
        df_full
        .groupby(["irr_bin", "temp_bin"])["pce_true"]
        .size()
        .unstack()
        .reindex(index=all_irr_idx, columns=all_temp_idx)
    )

    n_observed = lookup.notna().sum().sum()
    n_total = lookup.size
    n_empty = n_total - n_observed
    print(f"  Bins observed      : {n_observed} / {n_total}")
    print(f"  Bins left as NaN   : {n_empty} / {n_total}")
    print(f"  Lookup mean PCE    : {lookup.mean().mean():.2f}%")
    print(f"  Lookup max PCE     : {lookup.max().max():.2f}%")
    print(f"  Ref-period p{percentile:.0f} PCE : {np.percentile(y_ref, percentile):.2f}%")
    print(f"  Ref-period mean PCE: {y_ref.mean():.2f}%")

    return {
        "lookup_table": lookup,
        "lookup_obs": lookup_obs,
        "lookup_counts": lookup_counts,
        "lookup_counts_full": lookup_counts_full,
        "irr_bins": irr_bins,
        "temp_bins": temp_bins,
        "irr_centres": irr_centres,
        "temp_centres": temp_centres,
        "ref_dates": ref_dates,
        "percentile": percentile,
        "temp_col": temp_col,
        "temp_label": temp_label,
        "lin_model": dict(alpha=alpha, beta_G=beta_G, beta_T=beta_T, r2=r2_diag),
    }

def get_reference_pce(df: pd.DataFrame, ref_surface: dict) -> pd.Series:
    """
    Look up the reference PCE for each row in df using the (Irr, temp) bin lookup table.
    """
    lookup = ref_surface["lookup_table"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    temp_col = ref_surface.get("temp_col", TEMP_COL)

    eta_ref = pd.Series(np.nan, index=df.index)
    irr_binned = pd.cut(df["Irr"], bins=irr_bins, labels=False)
    tmp_binned = pd.cut(df[temp_col], bins=temp_bins, labels=False)

    valid = irr_binned.notna() & tmp_binned.notna()
    for idx in df.index[valid]:
        i = int(irr_binned[idx])
        j = int(tmp_binned[idx])
        if i in lookup.index and j in lookup.columns:
            val = lookup.loc[i, j]
            if not pd.isna(val):
                eta_ref[idx] = val

    n_filled = eta_ref.notna().sum()
    n_nan = eta_ref.isna().sum()
    print(f"  get_reference_pce: {n_filled:,} filled  |  {n_nan:,} NaN  [T = {temp_col}]")
    return eta_ref

def plot_reference_surface(ref_surface: dict, output_dir: Path, target: str = "Power"):
    """
    Heatmap of the reference PCE lookup table in (Irradiance * Temperature) space.
    """
    lookup = ref_surface["lookup_table"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    pct = ref_surface.get("percentile", 75)
    temp_label = ref_surface.get("temp_label", TEMP_LABEL)

    irr_labels = [f"{int(irr_bins[i])}-{int(irr_bins[i+1])}"
                  for i in range(len(irr_bins) - 1)]
    temp_labels = [f"{int(temp_bins[i])}-{int(temp_bins[i+1])}"
                   for i in range(len(temp_bins) - 1)]

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f"\n{target}", fontsize=24, fontweight="bold")

    arr = lookup.values.T
    masked = np.ma.masked_invalid(arr)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    vmin = float(lookup.min().min())
    vmax = float(lookup.max().max())
    im = ax.imshow(masked, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax, label=f"p{pct:.0f} PCE (%)")
    tick_values = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_values])
    cbar.mappable.set_clim(vmin, vmax)

    n_irr, n_temp = lookup.shape[0], lookup.shape[1]
    ax.set_xticks(np.arange(n_irr))
    ax.set_xticklabels(irr_labels, rotation=90, ha="right", fontsize=14)
    ax.set_yticks(np.arange(n_temp))
    ax.set_yticklabels(temp_labels, fontsize=14)
    ax.set_xlabel("Irradiance (W/m²)", fontsize=14)
    ax.set_ylabel(f"{temp_label}", fontsize=14)

    # Annotate each observed bin with its PCE value
    norm = plt.Normalize(vmin, vmax)
    for i in range(n_irr):
        for j in range(n_temp):
            v = lookup.values[i, j]
            if not np.isnan(v):
                rgba = cmap(norm(v))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(i, j, f"{v:.1f}", ha="center", va="center",
                        fontsize=14, color="black" if luminance > 0.5 else "white")

    plt.tight_layout()
    plt.savefig(output_dir / "reference_surface_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: reference_surface_heatmap.png")


def plot_coverage_heatmap(ref_surface: dict, output_dir: Path, target: str = "Power"):
    """
    Heatmap showing how many timestamps fall in each (Irr * Temperature) bin.
    """
    lookup_counts_full = ref_surface["lookup_counts_full"]
    lookup_counts_ref = ref_surface["lookup_counts"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    temp_label = ref_surface.get("temp_label", TEMP_LABEL)

    irr_labels = [f"{int(irr_bins[i])}-{int(irr_bins[i+1])}"
                  for i in range(len(irr_bins) - 1)]
    temp_labels = [f"{int(temp_bins[i])}-{int(temp_bins[i+1])}"
                   for i in range(len(temp_bins) - 1)]

    full = lookup_counts_full.values.astype(float)
    ref = lookup_counts_ref.values.astype(float)

    # Bins observed in reference period in blue; bins missed in red
    value = np.full(full.shape, np.nan, dtype=float)
    missed_mask = (full > 0) & ((ref == 0) | np.isnan(ref))
    observed_mask = (ref > 0) & ~np.isnan(ref)
    value[missed_mask] = 0
    value[observed_mask] = ref[observed_mask]

    masked_value = np.ma.masked_invalid(value)
    cmap = plt.cm.Blues.copy()
    cmap.set_under(color="lightcoral")
    cmap.set_bad(color="white")

    max_count = np.nanmax(value)
    norm = plt.Normalize(vmin=0.1, vmax=max_count)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f"{target}", fontsize=24, fontweight="bold")
    im = ax.imshow(masked_value.T, origin="lower", aspect="auto",
                   cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, extend="min")
    cbar.set_label("Number of timestamps", fontsize=11)

    n_irr, n_temp = full.shape
    ax.set_xticks(np.arange(n_irr))
    ax.set_xticklabels(irr_labels, rotation=90, ha="right", fontsize=14)
    ax.set_yticks(np.arange(n_temp))
    ax.set_yticklabels(temp_labels, fontsize=14)
    ax.set_xlabel("Irradiance (W/m²)", fontsize=14)
    ax.set_ylabel(f"{temp_label}", fontsize=14)

    for i in range(n_irr):
        for j in range(n_temp):
            cnt_ref = ref[i, j]
            cnt_full = full[i, j]
            if not np.isnan(cnt_ref) and cnt_ref > 0:
                text_color = "white" if cnt_ref > 0.5 * max_count else "black"
                ax.text(i, j, str(int(cnt_ref)), ha="center", va="center",
                        fontsize=14, color=text_color)
            elif missed_mask[i, j]:
                ax.text(i, j, str(int(cnt_full)), ha="center", va="center",
                        fontsize=14, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / f"coverage_heatmap_{target}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: coverage_heatmap_{target}.png")

def compute_daily_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily energy-based degradation metrics using the relative-delta method.
    """
    temp_col, temp_label = _resolve_temp_col(df)

    print(f"\n{'='*70}")
    print("COMPUTING DEGRADATION ANALYSIS")
    print(f"Temperature column: '{temp_col}' ({temp_label})")
    print(f"{'='*70}")

    results = []
    total_removed_true = 0
    total_removed_pred = 0
    days_fully_removed = 0

    for date, day_group in df.groupby("date"):
        day_clean = day_group[
            (day_group["Irr"] >= MIN_IRR_FOR_PCE) &
            day_group["pce_true"].notna() &
            day_group["eta_ref_true"].notna() &
            day_group["dt_hours"].notna() &
            day_group[temp_col].notna()
        ].copy()

        if len(day_clean) < 5:
            continue

        original_size = len(day_clean)

        # Remove timesteps where measured PCE exceeds reference in either branch!
        above_ref_mask_true = day_clean["pce_true"] > day_clean["eta_ref_true"]
        above_ref_mask_pred = day_clean["pce_pred"] > day_clean["eta_ref_pred"]
        n_removed_true = above_ref_mask_true.sum()
        n_removed_pred = above_ref_mask_pred.sum()
        total_removed_true += n_removed_true
        total_removed_pred += n_removed_pred

        points_to_remove = above_ref_mask_true | above_ref_mask_pred
        day_clean = day_clean[~points_to_remove].copy()

        if len(day_clean) < 5:
            days_fully_removed += 1
            continue

        for branch in ("true", "pred"):
            pce_col = f"pce_{branch}"
            ref_col = f"eta_ref_{branch}"

            # Relative degradation at each timestep (clamp negatives to 0)
            day_clean[f"delta_rel_{branch}"] = (
                (day_clean[ref_col] - day_clean[pce_col]) / day_clean[ref_col]
            ).replace([np.inf, -np.inf], np.nan)

            if (day_clean[f"delta_rel_{branch}"] < 0).any():
                day_clean.loc[
                    day_clean[f"delta_rel_{branch}"] < 0,
                    f"delta_rel_{branch}"
                ] = 0

        day_clean = day_clean.dropna(subset=["delta_rel_true", "delta_rel_pred"])
        if len(day_clean) == 0:
            days_fully_removed += 1
            continue

        min_delta_true = day_clean["delta_rel_true"].min()
        min_delta_pred = day_clean["delta_rel_pred"].min()
        scale_true = np.clip(1.0 + min_delta_true, 1.0, 1.2)
        scale_pred = np.clip(1.0 + min_delta_pred, 1.0, 1.2)

        day_clean["pce_true_adjusted"] = day_clean["pce_true"] * scale_true
        day_clean["pce_pred_adjusted"] = day_clean["pce_pred"] * scale_pred

        # Energy integration (Wh/m² per panel area)
        dt = day_clean["dt_hours"]
        e_ref_true = (day_clean["eta_ref_true"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_ref_pred = (day_clean["eta_ref_pred"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_meas_true = (day_clean["pce_true"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_meas_pred = (day_clean["pce_pred"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_adj_true = (day_clean["pce_true_adjusted"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_adj_pred = (day_clean["pce_pred_adjusted"] / 100.0 * day_clean["Irr"] * dt).sum()

        if e_ref_true < 0.1 or e_ref_pred < 0.1:
            continue

        if e_meas_true > e_ref_true * 1.001 or e_meas_pred > e_ref_pred * 1.001:
            print(f"WARNING: {date}: Energy check failed after filtering - skipping")
            continue

        # Decompose losses into reversible and irreversible components
        total_loss_true = (e_ref_true - e_meas_true) / e_ref_true
        rev_loss_true = (e_ref_true - e_adj_true) / e_ref_true
        irr_loss_true = (e_adj_true - e_meas_true) / e_ref_true
        total_loss_pred = (e_ref_pred - e_meas_pred) / e_ref_pred
        rev_loss_pred = (e_ref_pred - e_adj_pred) / e_ref_pred
        irr_loss_pred = (e_adj_pred - e_meas_pred) / e_ref_pred

        if total_loss_true < -0.001 or total_loss_pred < -0.001:
            continue

        total_loss_true = max(0, total_loss_true)
        total_loss_pred = max(0, total_loss_pred)
        rev_loss_true = max(0, rev_loss_true)
        rev_loss_pred = max(0, rev_loss_pred)
        irr_loss_true = max(0, irr_loss_true)
        irr_loss_pred = max(0, irr_loss_pred)

        pr_true = e_meas_true / e_ref_true
        pr_pred = e_meas_pred / e_ref_pred

        if not (0.1 <= pr_true <= 1.0) or not (0.1 <= pr_pred <= 1.0):
            continue

        results.append({
            "date": date,
            "n_points_original": original_size,
            "n_points_after_filter": len(day_clean),
            "n_removed_true": n_removed_true,
            "n_removed_pred": n_removed_pred,
            "temp_mean": day_clean[temp_col].mean(),
            "temp_median": day_clean[temp_col].median(),
            "irr_mean": day_clean["Irr"].mean(),
            "irr_median": day_clean["Irr"].median(),
            "meas_pce_true": day_clean["pce_true"].median(),
            "meas_pce_pred": day_clean["pce_pred"].median(),
            "pr_true": pr_true,
            "pr_pred": pr_pred,
            "e_ref_true": e_ref_true,
            "e_meas_true": e_meas_true,
            "e_adj_true": e_adj_true,
            "total_loss_true_pct": total_loss_true * 100,
            "reversible_loss_true_pct": rev_loss_true * 100,
            "irreversible_loss_true_pct": irr_loss_true * 100,
            "min_gap_true": min_delta_true,
            "scale_true": scale_true,
            "e_ref_pred": e_ref_pred,
            "e_meas_pred": e_meas_pred,
            "e_adj_pred": e_adj_pred,
            "total_loss_pred_pct": total_loss_pred * 100,
            "reversible_loss_pred_pct": rev_loss_pred * 100,
            "irreversible_loss_pred_pct": irr_loss_pred * 100,
            "min_gap_pred": min_delta_pred,
            "scale_pred": scale_pred,
        })

    daily_df = pd.DataFrame(results).set_index("date").sort_index()

    print(f"\nFiltering Summary:")
    print(f"{'='*50}")
    print(f"Days with valid degradation analysis : {len(daily_df)}")
    print(f"Days completely removed : {days_fully_removed}")
    print(f"Total points removed (PCE > reference):")
    print(f"TRUE branch : {total_removed_true}")
    print(f"LSTM branch : {total_removed_pred}")

    if len(daily_df) > 0:
        avg_pct_true = (daily_df["n_removed_true"] / daily_df["n_points_original"] * 100).mean()
        avg_pct_pred = (daily_df["n_removed_pred"] / daily_df["n_points_original"] * 100).mean()
        print(f"Avg removal % : {avg_pct_true:.1f}% (TRUE), {avg_pct_pred:.1f}% (LSTM)")

        print(f"\nDegradation Summary (after filtering):")
        print(f"{'='*50}")
        for branch, suffix in [("TRUE", "true"), ("LSTM", "pred")]:
            print(f"\n  {branch} branch:")
            print(f"Mean total loss: {daily_df[f'total_loss_{suffix}_pct'].mean():.2f}%")
            print(f"Mean reversible loss: {daily_df[f'reversible_loss_{suffix}_pct'].mean():.2f}%")
            print(f"Mean irreversible loss: {daily_df[f'irreversible_loss_{suffix}_pct'].mean():.2f}%")
            print(f"Mean performance ratio: {daily_df[f'pr_{suffix}'].mean():.4f}")
            print(f"PR range: {daily_df[f'pr_{suffix}'].min():.4f} - {daily_df[f'pr_{suffix}'].max():.4f}")

    return daily_df

def plot_daily_loss_bars_valid_days(
    daily_df: pd.DataFrame,
    output_dir: Path,
    start_date: str = None,
    end_date: str = None,
    target: str = "Power",
):
    """
    Stacked bar chart of daily reversible and irreversible energy losses for both the true (measured) and LSTM (predicted) branches,
    plus two difference panels (TRUE - LSTM) for each loss component.
    """
    if daily_df is None or len(daily_df) == 0:
        print("WARNING: No daily data for loss bars plot")
        return

    plot_df = daily_df.copy()
    if plot_df.index.tz is not None:
        plot_df.index = plot_df.index.tz_localize(None)

    if start_date:
        plot_df = plot_df[plot_df.index >= pd.Timestamp(start_date)]
    if end_date:
        plot_df = plot_df[plot_df.index <= pd.Timestamp(end_date)]

    plot_df = plot_df[plot_df["reversible_loss_true_pct"].notna()].copy()

    if len(plot_df) == 0:
        print("WARNING: No valid days in specified date range")
        return

    plot_df["reversible_diff_pct"] = (plot_df["reversible_loss_true_pct"] - plot_df["reversible_loss_pred_pct"])
    plot_df["irreversible_diff_pct"] = (plot_df["irreversible_loss_true_pct"] - plot_df["irreversible_loss_pred_pct"])

    r2_reversible   = r2_score(plot_df["reversible_loss_true_pct"], plot_df["reversible_loss_pred_pct"])
    r2_irreversible = r2_score(plot_df["irreversible_loss_true_pct"], plot_df["irreversible_loss_pred_pct"])

    print(f"Valid days: {len(plot_df)}")
    print(f"Reversible Loss R²: {r2_reversible:.4f}")
    print(f"Irreversible Loss R²: {r2_irreversible:.4f}")

    fig, (ax_t, ax_p, ax_rev_diff, ax_irr_diff) = plt.subplots(
        4, 1, figsize=(max(14, len(plot_df) * 0.15), 18)
    )
    fig.suptitle(f"Daily Loss Decomposition | {target}", fontsize=24, fontweight="bold", y=0.99)

    x_pos = np.arange(len(plot_df))
    valid_dates = plot_df.index.strftime("%Y-%m-%d")
    tick_step = 7
    tick_positions = x_pos[::tick_step]
    tick_labels = [valid_dates[i] for i in range(0, len(valid_dates), tick_step)]
    bar_width = 0.7

    for ax, (rcol, icol), panel_title in [
        (ax_t, ("reversible_loss_true_pct", "irreversible_loss_true_pct"), "TRUE (Measured)"),
        (ax_p, ("reversible_loss_pred_pct", "irreversible_loss_pred_pct"), "LSTM (Predicted)"),
    ]:
        irr_vals = plot_df[icol].fillna(0).values
        rev_vals = plot_df[rcol].values
        ax.bar(x_pos, irr_vals, color="tomato", alpha=0.85, label="Irreversible",
               width=bar_width, edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, color="mediumseagreen", alpha=0.85,
               label="Reversible", width=bar_width, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(panel_title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        ax.set_ylim(0, max(irr_vals + rev_vals) * 1.25)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5, zorder=2)
        ax.set_xlim(-1.0, len(plot_df) + 0.2)
        ax.set_xticks(tick_positions)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        mean_rev = rev_vals.mean()
        mean_irr = irr_vals.mean()
        ax.text(
            0.02, 0.96,
            f"Mean Rev = {mean_rev:.2f}%\nMean Irr = {mean_irr:.2f}%",
            transform=ax.transAxes, fontsize=20,
            verticalalignment="top", horizontalalignment="left", multialignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    for ax, diff_col, r2_val, diff_title in [
        (ax_rev_diff, "reversible_diff_pct",   r2_reversible, "TRUE - LSTM (Reversible)"),
        (ax_irr_diff, "irreversible_diff_pct", r2_irreversible, "TRUE - LSTM (Irreversible)"),
    ]:
        diff_vals = plot_df[diff_col].values
        for pos, val in zip(x_pos, diff_vals):
            color = "#1f77b4" if val > 0 else "#ff7f0e"
            ax.bar(pos, val, color=color, alpha=0.8, width=bar_width,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.02, 0.98, f"R² = {r2_val:.3f}", transform=ax.transAxes, fontsize=20,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))
        ax.set_title(diff_title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        max_diff = max(abs(diff_vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5, zorder=2)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5, zorder=2)
        ax.set_xlim(-1.0, len(plot_df) + 0.2)
        ax.set_xticks(tick_positions)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_irr_diff.set_xlabel("Day", fontsize=16, fontweight="bold", labelpad=20)

    loss_handles = [
        Patch(facecolor="tomato", alpha=0.85, edgecolor="white", label="Irreversible"),
        Patch(facecolor="mediumseagreen", alpha=0.85, edgecolor="white", label="Reversible"),
    ]
    diff_handles = [
        Patch(facecolor="#1f77b4", alpha=0.8, edgecolor="black", label="TRUE > LSTM"),
        Patch(facecolor="#ff7f0e", alpha=0.8, edgecolor="black", label="TRUE < LSTM"),
    ]
    fig.legend(handles=loss_handles + diff_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=20,
               frameon=True, edgecolor="black")

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.96])

    start_str = plot_df.index.min().strftime("%Y%m%d")
    end_str = plot_df.index.max().strftime("%Y%m%d")
    filename = output_dir / f"daily_loss_bars_valid_only_{start_str}_{end_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename.name}")


def plot_monthly_loss_bars(
    daily_df: pd.DataFrame,
    df_full: pd.DataFrame,
    output_dir: Path,
    target: str = "Power",
):
    """
    Monthly mean loss decomposition as stacked bar charts with n_days annotations.
    """
    dfc = daily_df.copy()
    dfc["year_month"] = dfc.index.strftime("%Y-%m")
    day_counts = dfc.groupby("year_month").size()

    monthly = dfc.groupby("year_month").agg(
        reversible_loss_true_pct=("reversible_loss_true_pct", "mean"),
        irreversible_loss_true_pct=("irreversible_loss_true_pct", "mean"),
        reversible_loss_pred_pct=("reversible_loss_pred_pct", "mean"),
        irreversible_loss_pred_pct=("irreversible_loss_pred_pct", "mean"),
        e_ref_true_sum=("e_ref_true", "sum"),
        e_meas_true_sum=("e_meas_true", "sum"),
        e_ref_pred_sum=("e_ref_pred", "sum"),
        e_meas_pred_sum=("e_meas_pred", "sum"),
    ).reset_index()

    monthly["total_loss_true_pct"] = (monthly["reversible_loss_true_pct"] + monthly["irreversible_loss_true_pct"])
    monthly["total_loss_pred_pct"] = (monthly["reversible_loss_pred_pct"] + monthly["irreversible_loss_pred_pct"])
    monthly["reversible_diff_pct"] = (monthly["reversible_loss_true_pct"] - monthly["reversible_loss_pred_pct"])
    monthly["irreversible_diff_pct"] = (monthly["irreversible_loss_true_pct"] - monthly["irreversible_loss_pred_pct"])

    r2_reversible = r2_score(monthly["reversible_loss_true_pct"], monthly["reversible_loss_pred_pct"])
    r2_irreversible = r2_score(monthly["irreversible_loss_true_pct"], monthly["irreversible_loss_pred_pct"])

    x_pos = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(f"Monthly Mean Loss Decomposition | {target}",
                 fontsize=24, fontweight="bold", y=1.02)
    axes = axes.flatten()

    for ax_i, (rcol, icol), panel_title in [
        (0, ("reversible_loss_true_pct", "irreversible_loss_true_pct"), "TRUE (Measured)"),
        (1, ("reversible_loss_pred_pct", "irreversible_loss_pred_pct"), "LSTM (Predicted)"),
    ]:
        ax = axes[ax_i]
        irr_vals = monthly[icol].fillna(0).values
        rev_vals = monthly[rcol].values
        total_vals = irr_vals + rev_vals
        ax.bar(x_pos, irr_vals, label="Irreversible", color="tomato", alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, label="Reversible",
               color="mediumseagreen", alpha=0.85, edgecolor="white", linewidth=0.5)

        for i, (_, row) in enumerate(monthly.iterrows()):
            n_days = day_counts[row["year_month"]]
            bar_h  = total_vals[i]
            ax.text(i, bar_h + bar_h * 0.05, f"n={n_days}",
                    ha="center", va="bottom", fontsize=14, rotation=90,
                    color="black", alpha=0.7)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        ax.set_title(panel_title, fontsize=20, fontweight="bold", pad=20)
        ax.set_ylabel("Energy Loss (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Month", fontsize=16, fontweight="bold", labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax_i, diff_col, r2_val, diff_title in [
        (2, "reversible_diff_pct",   r2_reversible,   "TRUE - LSTM (Reversible)"),
        (3, "irreversible_diff_pct", r2_irreversible, "TRUE - LSTM (Irreversible)"),
    ]:
        ax = axes[ax_i]
        diff_vals = monthly[diff_col].values
        for pos, val in zip(x_pos, diff_vals):
            color = "#1f77b4" if val > 0 else "#ff7f0e"
            ax.bar(pos, val, color=color, alpha=0.8, width=0.6,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.02, 0.98, f"R² = {r2_val:.3f}", transform=ax.transAxes, fontsize=16,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(diff_title, fontsize=20, fontweight="bold")
        ax.set_ylabel("Energy Loss Diff (%)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Month", fontsize=16, fontweight="bold", labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="x", labelsize=14)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        max_diff = max(abs(diff_vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)

    loss_handles = [
        Patch(facecolor="tomato", alpha=0.85, edgecolor="white", label="Irreversible"),
        Patch(facecolor="mediumseagreen", alpha=0.85, edgecolor="white", label="Reversible"),
    ]
    diff_handles = [
        Patch(facecolor="#1f77b4", alpha=0.8, edgecolor="black", label="TRUE > LSTM"),
        Patch(facecolor="#ff7f0e", alpha=0.8, edgecolor="black", label="TRUE < LSTM"),
    ]
    fig.legend(handles=loss_handles + diff_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=20,
               frameon=True, edgecolor="black")

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0, rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / "monthly_loss_bars_with_diffs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: monthly_loss_bars_with_diffs.png")

    print(f"\nMonthly Loss Summary  [{target}]:")
    print(f"{'Month':<12} {'Days':>5} "
          f"{'Total True':>11} {'Rev True':>10} {'Irr True':>10} "
          f"{'Total LSTM':>11} {'Rev LSTM':>10} {'Irr LSTM':>10} "
          f"{'Rev Diff':>10} {'Irr Diff':>10}")
    print("-" * 100)
    for _, row in monthly.iterrows():
        month = row["year_month"]
        n_days = day_counts[month]
        print(
            f"{month:<12} {n_days:>5} "
            f"{row['total_loss_true_pct']:>10.2f}% "
            f"{row['reversible_loss_true_pct']:>9.2f}% "
            f"{row['irreversible_loss_true_pct']:>9.2f}% "
            f"{row['total_loss_pred_pct']:>10.2f}% "
            f"{row['reversible_loss_pred_pct']:>9.2f}% "
            f"{row['irreversible_loss_pred_pct']:>9.2f}% "
            f"{row['reversible_diff_pct']:>9.2f}% "
            f"{row['irreversible_diff_pct']:>9.2f}%"
        )
    print(f"\nR² (LSTM vs TRUE): Reversible={r2_reversible:.4f}, Irreversible={r2_irreversible:.4f}")

class DegradationAnalyzer:
    def __init__(
        self,
        model_path,
        model_info_path,
        train_data_path,
        test_data_path,
        target_scaler_path,
        feature_scaler_path,
        bad_days_list=None,
        output_dir=None,
        ref_percentile: float = 75,
        panel_temp_path: str = None,
    ):
        self.panel_temp_path = Path(panel_temp_path) if panel_temp_path else None
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.target_scaler_path = Path(target_scaler_path)
        self.feature_scaler_path = Path(feature_scaler_path)
        self.bad_days_list = bad_days_list or []
        self.ref_percentile = ref_percentile
        self.output_dir = (Path(output_dir) if output_dir else Path("pce_analysis_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        with open(self.model_info_path, "rb") as f:
            self.info = pickle.load(f)

        self.df_scaled = merge_train_test_parquets(self.train_data_path, self.test_data_path)

        with open(self.target_scaler_path,  "rb") as f: sd  = pickle.load(f)
        with open(self.feature_scaler_path, "rb") as f: fsd = pickle.load(f)

        self.target_cols = self.info["data_config"]["features"]["target_cols"]
        self.feature_cols = self.info["data_config"]["features"]["feature_cols"]
        self.target_scalers = {t: sd.get(t, sd.get("scaler")) for t in self.target_cols}
        self.feature_scalers = fsd.get("feature_scalers", {})

        from train_lstm_model import MultiOutputLSTMModel, NonOverlappingMultiOutputDataset

        arch = self.info["architecture"]
        horizon = self.info["data_config"]["window_horizon"]["horizon"]
        window = self.info["data_config"]["window_horizon"]["window"]
        use_bad = self.info["data_config"]["features"]["use_bad_day"]
        mask_bad = self.info["data_config"]["features"]["mask_bad_days"]
        bad_map = self.info.get("bad_day_mapping", {})

        self.device = torch.device("cpu")
        self.model  = MultiOutputLSTMModel(
            input_size=len(self.feature_cols),
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            num_outputs=len(self.target_cols),
            dropout=arch["dropout"],
            forecast_steps=horizon,
        )
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        self.model.to(self.device).eval()

        self.full_ds = NonOverlappingMultiOutputDataset(
            self.df_scaled,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            window=window,
            horizon=horizon,
            use_bad_day=use_bad,
            mask_bad_days=mask_bad,
            bad_day_mapping=bad_map,
        )
        print(f"Total windows: {len(self.full_ds)}")

        print("\nRunning inference...")
        self.full_preds = run_inference(self.model, self.full_ds, self.device)
        print(f"Predictions shape: {self.full_preds.shape}")
        self.panel_temp_series = self._load_panel_temp()

    def _load_panel_temp(self) -> pd.Series:
        """
        Load panel temperature from a csv file 
        """
        if self.panel_temp_path is None or not self.panel_temp_path.exists():
            print("panel_temp_path not provided or file not found, skipping panel temp")
            return pd.Series(dtype=float)

        print(f"\n  Loading panel temperature from: {self.panel_temp_path}")
        pt = pd.read_csv(self.panel_temp_path)

        ts_col = None
        for c in ["timestamp", "_time", "time", "datetime"]:
            if c in pt.columns:
                ts_col = c
                break
        if ts_col is None:
            raise ValueError(
                f"Cannot find timestamp column in {self.panel_temp_path}. "
                f"Columns: {list(pt.columns)}"
            )

        pt[ts_col] = pd.to_datetime(pt[ts_col], utc=True, errors="coerce")
        pt = pt.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

        temp_cols = [c for c in pt.columns if c.startswith("Temp_")]
        if not temp_cols:
            raise ValueError(
                f"No column starting with 'Temp_' found in {self.panel_temp_path}. "
                f"Columns: {list(pt.columns)}"
            )

        chosen = temp_cols[0]
        print(f"  Found panel temp column: '{chosen}'  "
              f"({pt[chosen].notna().sum():,} non-NaN rows)")
        if len(temp_cols) > 1:
            print(f"  Multiple Temp_ columns found: {temp_cols}, using first: '{chosen}'")

        return pt[chosen].rename("panel_temp")

    def run(self):
        """Execute the full degradation analysis pipeline."""
        print(f"\nTemperature setting: TEMP_COL = '{TEMP_COL}' ({TEMP_LABEL})")

        print("\nBuilding aligned DataFrame...")
        self.df = build_aligned_df(
            self.full_ds, self.full_preds, self.df_scaled,
            self.info, self.target_scalers, self.feature_scalers,
            bad_days_list=self.bad_days_list,
        )
        print(f"Final DataFrame: {len(self.df):,} rows  |  freq: {FREQ}")

        # Merge panel temperature if available
        if not self.panel_temp_series.empty:
            pt = self.panel_temp_series.copy()
            if pt.index.tz is None:
                pt.index = pt.index.tz_localize("UTC")
            df_idx = self.df.index
            df_idx_utc = df_idx.tz_localize("UTC") if df_idx.tz is None else df_idx
            pt_reindexed = pt.reindex(df_idx_utc)
            pt_reindexed.index = self.df.index
            self.df["panel_temp"] = pt_reindexed
            n_filled = self.df["panel_temp"].notna().sum()
            print(f"panel_temp merged: {n_filled:,} / {len(self.df):,} timestamps matched")
        else:
            self.df["panel_temp"] = np.nan
            print("panel_temp not available, column set to NaN")

        actual_temp_col, actual_temp_label = _resolve_temp_col(self.df)
        print(f"\nTemperature column: '{actual_temp_col}' ({actual_temp_label})")

        print_dataframe_stats(self.df)

        print("\nCalculating PCE")
        self.df = calculate_pce(self.df)
        target = _get_target_name(self.df)

        print("\nPlotting temperature correlation analysis")
        plot_temp_correlation(self.df, self.output_dir, "Perovskite_1")

        print("\nPlotting PCE temperature dependence")
        plot_pce_temperature_dependence(self.df, self.output_dir)

        print("\nPlotting PCE irradiance dependence")
        plot_pce_irradiance_dependence(self.df, self.output_dir)

        print(f"\nBuilding reference surface (p{self.ref_percentile:.0f}, "
              f"observed bins only, T = {actual_temp_label})...")
        self.ref_surface = build_reference_surface(
            self.df, ref_days=120, percentile=self.ref_percentile
        )
        plot_reference_surface(self.ref_surface, self.output_dir, target=target)
        plot_coverage_heatmap(self.ref_surface, self.output_dir, target=target)

        # Assign the same reference surface to both true and predicted branches
        self.df["eta_ref_true"] = get_reference_pce(self.df, self.ref_surface)
        self.df["eta_ref_pred"] = self.df["eta_ref_true"].copy()

        print("\nComputing degradation analysis.")
        self.daily_df = compute_daily_degradation(self.df)

        if len(self.daily_df) > 0:
            print("\nPlotting degradation results.")
            plot_daily_loss_bars_valid_days(self.daily_df, self.output_dir, target=target)

            print("\nPlotting monthly loss bars.")
            plot_monthly_loss_bars(self.daily_df, self.df, self.output_dir, target=target)

            self.daily_df.to_csv(self.output_dir / "daily_degradation.csv")
            print("Saved: daily_degradation.csv")

        self.df.to_parquet(self.output_dir / "aligned_df.parquet")
        print("Saved: aligned_df.parquet")

        print(f"\n{'-'*100}")
        print(f"Analysis complete. Results saved to: {self.output_dir}")

def main():
    BASE_PATH = "path/to/lstm_run_folder"
    bad_days = []
    analyzer = DegradationAnalyzer(
        model_path=f"{BASE_PATH}/lstm_results/<timestamp>_model_multi.pt",
        model_info_path=f"{BASE_PATH}/lstm_results/model_info_multi.pkl",
        train_data_path=f"{BASE_PATH}/training_data/train_scaled.parquet",
        test_data_path=f"{BASE_PATH}/training_data/test_scaled.parquet",
        target_scaler_path=f"{BASE_PATH}/training_data/p_scalers.pkl",
        feature_scaler_path=f"{BASE_PATH}/training_data/scalers.pkl",
        panel_temp_path=f"{BASE_PATH}/merged_pv_dwd_step.csv",
        bad_days_list=bad_days,
        output_dir=f"{BASE_PATH}/lstm_results/pce_analysis_output",
        ref_percentile=95,
    )
    analyzer.run()

if __name__ == "__main__":
    main()