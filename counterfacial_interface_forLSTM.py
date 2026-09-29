from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
import torch
from typing import Optional, List, Dict

from train_lstm_model import MultiOutputLSTMModel, NonOverlappingMultiOutputDataset

DAY_START_HOUR = 4
DAY_END_HOUR = 22
FREQ = "10min"
PANEL_AREA = 0.7              # m^2 -- psc : 0.7, solon : 1.44, sanyo : 1.125
MIN_IRR_FOR_PCE = 300.0
MIN_POWER_FOR_PCE = 20
TEMP_COL = "temp_C"           # panel_temp / temp_C
TEMP_LABEL = "temp_C"

def filter_bad_days(df: pd.DataFrame, bad_day_list: list) -> pd.DataFrame:
    if not bad_day_list or df.empty:
        return df
    bad_dates = pd.to_datetime(bad_day_list).normalize()
    mask = ~df["date"].isin(bad_dates)
    out = df[mask].copy()
    print(f"  Bad day filter: removed {len(df)-len(out)} rows from {len(bad_dates)} days")
    return out

def merge_train_test_parquets(train_path: Path, test_path: Path) -> pd.DataFrame:
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
        print(f"  Removing {dup.sum()} duplicate timestamps")
        merged = merged[~merged.index.duplicated(keep="first")]

    print(f"  Train rows : {(merged['_split']=='train').sum():,}")
    print(f"  Test rows  : {(merged['_split']=='test').sum():,}")
    print(f"  Date range : {merged.index.min().date()} \u2192 {merged.index.max().date()}")
    return merged


def run_inference(model, dataset, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch_x = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
            preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(preds, axis=0)

def _inverse_transform(arr, scaler) -> np.ndarray:
    arr = np.asarray(arr).ravel()
    if scaler is None:
        return arr.copy()
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

def filter_daylight_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    hours = df.index.hour
    mask = (hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR)
    out = df[mask].copy()
    print(f"  Daylight filter ({DAY_START_HOUR}:00-{DAY_END_HOUR}:00): "
          f"kept {len(out)}/{len(df)} rows ({100*len(out)/len(df):.1f}%)")
    return out

def _resolve_temp_col(df: pd.DataFrame) -> tuple[str, str]:
    if TEMP_COL == "panel_temp":
        if ("panel_temp" in df.columns and
                df["panel_temp"].notna().sum() > len(df) * 0.1):
            return "panel_temp", "Panel Temp (\u00b0C)"
        else:
            print("  WARNING: TEMP_COL='panel_temp' but column is missing or "
                  "mostly NaN \u2014 falling back to 'temp_C'.")
            return "temp_C", "Ambient Temperature (\u00b0C)"
    return "temp_C", "Ambient Temperature (\u00b0C)"

def build_aligned_df(
    dataset, preds_scaled, df_scaled, info,
    target_scalers, feature_scalers, bad_days_list=None,
) -> pd.DataFrame:
    target_cols = info["data_config"]["features"]["target_cols"]
    all_feat = info["data_config"]["features"]["feature_cols"]
    horizon = info["data_config"]["window_horizon"]["horizon"]
    window = info["data_config"]["window_horizon"]["window"]

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

    records = []
    for i in range(len(dataset)):
        if i in bad_indices:
            continue
        sample = dataset[i]
        y_true = sample[1]
        y_pred = preds_scaled[i]

        if hasattr(dataset, "get_target_index"):
            ts_index = dataset.get_target_index(i)
        elif hasattr(dataset, "get_timestamps_for_sample"):
            ts_index = dataset.get_timestamps_for_sample(i)
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

    for col, series in feat_real.items():
        df[col] = series.reindex(df.index)

    if "_split" in df_scaled.columns:
        df["_split"] = df_scaled["_split"].reindex(df.index)

    df["date"] = df.index.normalize()
    df = df.resample(FREQ).first()
    df["date"] = df.index.normalize()
    df["dt_hours"] = 10.0 / 60.0

    if bad_days_list:
        df = filter_bad_days(df, bad_days_list)
    df = filter_daylight_hours(df)
    df = df[df["Irr"].notna() & df["temp_C"].notna() & (df["Irr"] > 0)].copy()

    return df

def calculate_pce(df: pd.DataFrame) -> pd.DataFrame:
    true_cols = [c for c in df.columns if c.startswith("true_")]
    target_name = true_cols[0].replace("true_", "")
    print(f"  Using target: {target_name}")

    temp_col, temp_label = _resolve_temp_col(df)
    print(f"  Using temperature column: '{temp_col}' ({temp_label})")

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
    print(f"  Shared valid timestamps (Irr >= {MIN_IRR_FOR_PCE}, "
          f"P >= {MIN_POWER_FOR_PCE} for both): {base_valid.sum()}/{len(df)}")

    df["pce_true"] = np.nan
    df["pce_pred"] = np.nan
    df.loc[base_valid, "pce_true"] = power_true[base_valid] / denom[base_valid]
    df.loc[base_valid, "pce_pred"] = power_pred[base_valid] / denom[base_valid]

    df.loc[df["pce_true"] == 0, "pce_true"] = np.nan
    df.loc[df["pce_pred"] == 0, "pce_pred"] = np.nan

    print("\n  PCE (before outlier filter):")
    print(f"    pce_true: min={df['pce_true'].min():.4f}, "
          f"max={df['pce_true'].max():.4f}, valid={df['pce_true'].notna().sum()}")
    print(f"    pce_pred: min={df['pce_pred'].min():.4f}, "
          f"max={df['pce_pred'].max():.4f}, valid={df['pce_pred'].notna().sum()}")

    count_t_before = df["pce_true"].notna().sum()
    count_p_before = df["pce_pred"].notna().sum()

    p99_true = df.loc[df["pce_true"].notna(), "pce_true"].quantile(0.99)
    p99_pred = df.loc[df["pce_pred"].notna(), "pce_pred"].quantile(0.99)

    outlier_mask = (
        (df["pce_true"] > p99_true) |
        (df["pce_pred"] > p99_pred)
    )

    df.loc[outlier_mask, "pce_true"] = np.nan
    df.loc[outlier_mask, "pce_pred"] = np.nan

    removed_t = count_t_before - df["pce_true"].notna().sum()
    removed_p = count_p_before - df["pce_pred"].notna().sum()
    print(f"\n  After 99th-pct filter (joint): removed {removed_t} true / {removed_p} pred outliers")

    df["pce_true"] *= 100
    df["pce_pred"] *= 100

    print("\n  FINAL PCE (%):")
    print(f"    pce_true: min={df['pce_true'].min():.2f}%, "
          f"max={df['pce_true'].max():.2f}%, valid={df['pce_true'].notna().sum()}")
    print(f"    pce_pred: min={df['pce_pred'].min():.2f}%, "
          f"max={df['pce_pred'].max():.2f}%, valid={df['pce_pred'].notna().sum()}")

    overlap = (df["pce_true"].notna() & df["pce_pred"].notna()).sum()
    print(f"  Final overlap (timestamps with both valid): {overlap}")

    df.attrs["target_name"] = target_name
    return df

def _get_target_name(df: pd.DataFrame) -> str:
    if "target_name" in df.attrs:
        return df.attrs["target_name"]
    true_cols = [c for c in df.columns if c.startswith("true_")]
    if true_cols:
        return true_cols[0].replace("true_", "")
    return "Power"


# ---------------------------------------------------------------------------
# NEW: age=1-based PCE reference surface
# ---------------------------------------------------------------------------
def build_reference_surface_from_age1(
    df_age1_pce: pd.DataFrame,
    n_irr_bins: int = 10,
    n_temp_bins: int = 10,
    min_irr_ref: float = MIN_IRR_FOR_PCE,
    df_full_for_bounds: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Builds the (Irr, Temp) reference PCE lookup table directly from the
    age=1 counterfactual's PREDICTED PCE (pce_pred in df_age1_pce, which was
    computed by calculate_pce() on the age=1 aligned/denormalized frame).

    Every (Irr, Temp) bin gets the MEAN age=1 predicted PCE observed in that
    bin -- no p95, no linear extrapolation, no plane fit. Bins the age=1 run
    never visits are left as NaN ("impossible" or "unvisited" bins), same
    convention as the observed-only method, so downstream code (heatmap
    plotting, get_reference_pce) needs no changes.

    `df_full_for_bounds`, if given, is used only to set the bin-edge range
    (so the age=1 surface spans the same (Irr,Temp) grid as the full
    dataset, making the two reference-surface heatmaps visually comparable).
    Defaults to df_age1_pce's own range if not given.
    """
    temp_col, temp_label = _resolve_temp_col(df_age1_pce)

    print(f"\n{'='*70}")
    print(f"BUILDING REFERENCE SURFACE FROM AGE=1 PREDICTED PCE  [T = {temp_label}]")
    print(f"{'='*70}")

    df_valid = df_age1_pce[
        (df_age1_pce["Irr"] >= min_irr_ref) &
        df_age1_pce["pce_pred"].notna() & (df_age1_pce["pce_pred"] > 0) &
        df_age1_pce[temp_col].notna()
    ].copy()

    print(f"  Age=1 valid points : {len(df_valid)} (Irr >= {min_irr_ref:.0f} W/m\u00b2)")
    if len(df_valid) < 20:
        raise ValueError(
            f"Too few age=1 points above min_irr_ref={min_irr_ref} W/m\u00b2 "
            f"to build a reference surface."
        )

    bounds_df = df_full_for_bounds if df_full_for_bounds is not None else df_valid

    irr_min = MIN_IRR_FOR_PCE
    irr_max = np.ceil(bounds_df["Irr"].max() / 50) * 50
    temp_min = np.floor(bounds_df[temp_col].min() / 2) * 2
    temp_max = np.ceil(bounds_df[temp_col].max() / 2) * 2

    irr_bins = np.linspace(irr_min, irr_max, n_irr_bins + 1)
    temp_bins = np.linspace(temp_min, temp_max, n_temp_bins + 1)
    irr_centres = (irr_bins[:-1] + irr_bins[1:]) / 2
    temp_centres = (temp_bins[:-1] + temp_bins[1:]) / 2

    all_irr_idx = list(range(n_irr_bins))
    all_temp_idx = list(range(n_temp_bins))

    df_valid["irr_bin"] = pd.cut(df_valid["Irr"], bins=irr_bins, labels=False)
    df_valid["temp_bin"] = pd.cut(df_valid[temp_col], bins=temp_bins, labels=False)

    lookup = (
        df_valid
        .groupby(["irr_bin", "temp_bin"])["pce_pred"]
        .quantile(0.95) #.mean()
        .unstack()
        .reindex(index=all_irr_idx, columns=all_temp_idx)
    )
    obs_mask = lookup.notna()
    
    lookup_counts = (
        df_valid
        .groupby(["irr_bin", "temp_bin"])["pce_pred"]
        .count()
        .unstack()
        .reindex(index=all_irr_idx, columns=all_temp_idx)
    )
    

    # "Full dataset occupancy" for the coverage-heatmap convention: since
    # this method has no extrapolation step, "exists" bins are just the
    # observed bins themselves (nothing beyond age=1 data is ever filled).
    exists_mask = obs_mask.copy()
    lookup_counts_full = lookup_counts.copy()

    n_obs = int(obs_mask.sum().sum())
    n_total = int(obs_mask.size)
    print(f"\n  Bin filling summary (age=1 method):")
    print(f"    Observed bins (mean age=1 pred PCE) : {n_obs} / {n_total}")
    print(f"    Unvisited bins (NaN, no extrapolation): {n_total - n_obs} / {n_total}")
    print(f"    Lookup mean PCE (filled bins) : {lookup.mean().mean():.2f}%")
    print(f"    Lookup max PCE  (filled bins) : {lookup.max().max():.2f}%")
    print(f"    Age=1 overall mean PCE        : {df_valid['pce_pred'].mean():.2f}%")

    return {
        "lookup_table": lookup,
        "lookup_obs": lookup,          # identical here -- no extrapolation branch
        "obs_mask": obs_mask,
        "exists_mask": exists_mask,
        "lookup_counts": lookup_counts,
        "lookup_counts_full": lookup_counts_full,
        "irr_bins": irr_bins,
        "temp_bins": temp_bins,
        "irr_centres": irr_centres,
        "temp_centres": temp_centres,
        "temp_col": temp_col,
        "temp_label": temp_label,
        "method": "age1_mean",
    }


def get_reference_pce(df: pd.DataFrame, ref_surface: dict) -> tuple[pd.Series, pd.Series]:
    """
    Look up eta_ref for every row of df from the (age=1-based or observed)
    reference surface. is_extrap is always 0.0 here for filled bins since
    the age=1 method performs no extrapolation -- kept for interface
    compatibility with downstream code from pce_surface_analysis.py.
    """
    lookup = ref_surface["lookup_table"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    temp_col = ref_surface.get("temp_col", TEMP_COL)

    eta_ref = pd.Series(np.nan, index=df.index)
    is_extrap = pd.Series(np.nan, index=df.index)

    irr_binned = pd.cut(df["Irr"], bins=irr_bins, labels=False)
    tmp_binned = pd.cut(df[temp_col], bins=temp_bins, labels=False)

    valid = irr_binned.notna() & tmp_binned.notna()
    bin_i = irr_binned[valid].astype(int)
    bin_j = tmp_binned[valid].astype(int)

    for idx, i, j in zip(df.index[valid], bin_i, bin_j):
        if i in lookup.index and j in lookup.columns:
            val = lookup.loc[i, j]
            if not pd.isna(val):
                eta_ref[idx] = val
                is_extrap[idx] = 0.0

    n_filled = eta_ref.notna().sum()
    print(f"  get_reference_pce (age=1 method): {n_filled:,} filled  |  "
          f"{eta_ref.isna().sum():,} NaN  [T = {temp_col}]")

    return eta_ref, is_extrap


def plot_reference_surface_age1(ref_surface: dict, output_dir: Path, target: str = "Power"):
    """
    Same visual style as plot_reference_surface() in pce_surface_analysis.py,
    but titled for the age=1 method and with no linear-model annotation
    (since none was fit).
    """
    lookup = ref_surface["lookup_table"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    temp_label = ref_surface.get("temp_label", TEMP_LABEL)

    counts_full = ref_surface["lookup_counts_full"]
    temp_has_data = counts_full.sum(axis=0) > 0
    lookup = lookup.loc[:, temp_has_data]

    irr_labels = [f"{int(irr_bins[i])}-{int(irr_bins[i+1])}"
                  for i in range(len(irr_bins)-1)]
    surviving_temp_idx = temp_has_data[temp_has_data].index.tolist()
    temp_labels = [f"{int(temp_bins[i])}-{int(temp_bins[i+1])}"
                   for i in surviving_temp_idx]

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        f"{target} \u2014 Reference PCE Surface from Age=1 Predicted PCE (mean per bin, no extrapolation)",
        fontsize=18, fontweight="bold",
    )

    arr = lookup.values.T
    masked = np.ma.masked_invalid(arr)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    vmin = float(lookup.min().min())
    vmax = float(lookup.max().max())
    im = ax.imshow(masked, origin="lower", aspect="auto",
                    cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean age=1 predicted PCE (%)", fontsize=16)
    tick_values = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_values])
    cbar.ax.tick_params(labelsize=14)

    n_irr, n_temp = lookup.shape[0], lookup.shape[1]
    ax.set_xticks(np.arange(n_irr))
    ax.set_xticklabels(irr_labels, rotation=90, ha="right", fontsize=12)
    ax.set_yticks(np.arange(n_temp))
    ax.set_yticklabels(temp_labels, fontsize=12)
    ax.set_xlabel("Irradiance (W/m\u00b2)", fontsize=14)
    ax.set_ylabel(f"{temp_label}", fontsize=14)

    norm = plt.Normalize(vmin, vmax)
    for i in range(n_irr):
        for j in range(n_temp):
            v = lookup.values[i, j]
            if not np.isnan(v):
                rgba = cmap(norm(v))
                luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                ax.text(i, j, f"{v:.1f}", ha="center", va="center",
                        fontsize=12, color="black" if luminance > 0.5 else "white")

    plt.tight_layout()
    fpath = output_dir / f"{target}_reference_surface_age1_heatmap.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fpath}")


def plot_coverage_heatmap_age1(ref_surface: dict, output_dir: Path, target: str = "Power"):
    """
    Coverage heatmap for the age=1 method: since there is no extrapolation
    branch, this simply shows the count of age=1 timestamps landing in each
    (Irr, Temp) bin.
    """
    counts = ref_surface["lookup_counts"]
    irr_bins = ref_surface["irr_bins"]
    temp_bins = ref_surface["temp_bins"]
    temp_label = ref_surface.get("temp_label", TEMP_LABEL)

    temp_has_data = counts.sum(axis=0) > 0
    counts = counts.loc[:, temp_has_data]
    surviving_temp_idx = temp_has_data[temp_has_data].index.tolist()

    irr_labels = [f"{int(irr_bins[i])}-{int(irr_bins[i+1])}"
                  for i in range(len(irr_bins)-1)]
    temp_labels = [f"{int(temp_bins[i])}-{int(temp_bins[i+1])}"
                   for i in surviving_temp_idx]

    arr = counts.values.T.astype(float)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f"{target} \u2014 Age=1 Reference Surface Coverage (timestamp count per bin)",
                 fontsize=18, fontweight="bold")

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of age=1 timestamps", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    n_irr, n_temp = counts.shape[0], counts.shape[1]
    ax.set_xticks(np.arange(n_irr))
    ax.set_xticklabels(irr_labels, rotation=90, ha="right", fontsize=12)
    ax.set_yticks(np.arange(n_temp))
    ax.set_yticklabels(temp_labels, fontsize=12)
    ax.set_xlabel("Irradiance (W/m\u00b2)", fontsize=14)
    ax.set_ylabel(f"{temp_label}", fontsize=14)

    max_count = np.nanmax(arr) if np.isfinite(arr).any() else 1
    for i in range(n_irr):
        for j in range(n_temp):
            v = counts.values[i, j]
            if not np.isnan(v) and v > 0:
                color = "white" if v > 0.5 * max_count else "black"
                ax.text(i, j, str(int(v)), ha="center", va="center",
                        fontsize=12, color=color)

    plt.tight_layout()
    fpath = output_dir / f"{target}_coverage_heatmap_age1.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fpath}")


# ---------------------------------------------------------------------------
# Per-day reversible / irreversible loss decomposition (reused from
# pce_surface_analysis.py, unchanged) + monthly plotting helpers
# ---------------------------------------------------------------------------
def compute_daily_degradation_notebook_method(df: pd.DataFrame) -> pd.DataFrame:
    temp_col, temp_label = _resolve_temp_col(df)

    print(f"\n{'='*70}")
    print("COMPUTING DEGRADATION ANALYSIS (Energy-Based, Relative-Delta Method)")
    print("  NO CLIPPING — PCE > eta_ref allowed to pass through as-is")
    print(f"  Temperature column: '{temp_col}' ({temp_label})")
    print(f"{'='*70}")

    results = []
    timestamp_records = []
    total_above_ref_true = 0
    total_above_ref_pred = 0
    days_fully_removed = 0
    days_with_above_ref_true = 0
    days_with_above_ref_pred = 0

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

        above_ref_mask_true = day_clean["pce_true"] > day_clean["eta_ref_true"]
        above_ref_mask_pred = day_clean["pce_pred"] > day_clean["eta_ref_pred"]
        n_above_ref_true = int(above_ref_mask_true.sum())
        n_above_ref_pred = int(above_ref_mask_pred.sum())
        total_above_ref_true += n_above_ref_true
        total_above_ref_pred += n_above_ref_pred
        if n_above_ref_true > 0:
            days_with_above_ref_true += 1
        if n_above_ref_pred > 0:
            days_with_above_ref_pred += 1

        day_clean["pce_true_above_ref_flag"] = above_ref_mask_true
        day_clean["pce_pred_above_ref_flag"] = above_ref_mask_pred

        day_clean["pce_true_raw"] = day_clean["pce_true"].copy()
        day_clean["pce_pred_raw"] = day_clean["pce_pred"].copy()

        day_clean["pce_true_smooth"] = (
            day_clean["pce_true"]
            .rolling(window=3, center=True, min_periods=1)
            .mean()
        )
        day_clean["pce_pred_smooth"] = (
            day_clean["pce_pred"]
            .rolling(window=3, center=True, min_periods=1)
            .mean()
        )

        for branch in ("true", "pred"):
            pce_col = f"pce_{branch}_smooth"
            ref_col = f"eta_ref_{branch}"

            day_clean[f"delta_rel_{branch}"] = (
                (day_clean[ref_col] - day_clean[pce_col]) /
                day_clean[ref_col]
            ).replace([np.inf, -np.inf], np.nan)

            if (day_clean[f"delta_rel_{branch}"] < 0).any():
                day_clean.loc[day_clean[f"delta_rel_{branch}"] < 0, f"delta_rel_{branch}"] = 0

        day_clean = day_clean.dropna(subset=["delta_rel_true", "delta_rel_pred"])
        if len(day_clean) == 0:
            days_fully_removed += 1
            continue

        min_delta_true = day_clean["delta_rel_true"].min()
        min_delta_pred = day_clean["delta_rel_pred"].min()
        scale_true = np.clip(1.0 + min_delta_true, 1.0, 1.2)
        scale_pred = np.clip(1.0 + min_delta_pred, 1.0, 1.2)

        day_clean["pce_true_adjusted"] = day_clean["pce_true_smooth"] * scale_true
        day_clean["pce_pred_adjusted"] = day_clean["pce_pred_smooth"] * scale_pred

        dt = day_clean["dt_hours"]
        e_ref_true = (day_clean["eta_ref_true"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_ref_pred = (day_clean["eta_ref_pred"] / 100.0 * day_clean["Irr"] * dt).sum()

        e_meas_true = (day_clean["pce_true_smooth"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_meas_pred = (day_clean["pce_pred_smooth"] / 100.0 * day_clean["Irr"] * dt).sum()

        e_adj_true = (day_clean["pce_true_adjusted"] / 100.0 * day_clean["Irr"] * dt).sum()
        e_adj_pred = (day_clean["pce_pred_adjusted"] / 100.0 * day_clean["Irr"] * dt).sum()

        if e_ref_true < 0.1 or e_ref_pred < 0.1:
            continue

        energy_exceeds_ref_true = e_meas_true > e_ref_true
        energy_exceeds_ref_pred = e_meas_pred > e_ref_pred

        total_loss_true = (e_ref_true - e_meas_true) / e_ref_true
        rev_loss_true = (e_ref_true - e_adj_true) / e_ref_true
        irr_loss_true = (e_adj_true - e_meas_true) / e_ref_true

        total_loss_pred = (e_ref_pred - e_meas_pred) / e_ref_pred
        rev_loss_pred = (e_ref_pred - e_adj_pred) / e_ref_pred
        irr_loss_pred = (e_adj_pred - e_meas_pred) / e_ref_pred

        total_loss_true_report = max(0, total_loss_true)
        total_loss_pred_report = max(0, total_loss_pred)
        rev_loss_true_report = max(0, rev_loss_true)
        rev_loss_pred_report = max(0, rev_loss_pred)
        irr_loss_true_report = max(0, irr_loss_true)
        irr_loss_pred_report = max(0, irr_loss_pred)

        pr_true = e_meas_true / e_ref_true
        pr_pred = e_meas_pred / e_ref_pred

        if pr_true < 0.1 or pr_pred < 0.1:
            continue

        day_clean_out = day_clean.copy()
        day_clean_out["scale_true"] = scale_true
        day_clean_out["scale_pred"] = scale_pred
        day_clean_out["date"] = date
        timestamp_records.append(day_clean_out)

        results.append({
            "date": date,
            "n_points_original": original_size,
            "n_points_after_filter": len(day_clean),
            "n_above_ref_true": n_above_ref_true,
            "n_above_ref_pred": n_above_ref_pred,
            "energy_exceeds_ref_true": energy_exceeds_ref_true,
            "energy_exceeds_ref_pred": energy_exceeds_ref_pred,
            "temp_mean": day_clean[temp_col].mean(),
            "temp_median": day_clean[temp_col].median(),
            "irr_mean": day_clean["Irr"].mean(),
            "irr_median": day_clean["Irr"].median(),
            "meas_pce_true": day_clean["pce_true_smooth"].median(),
            "meas_pce_pred": day_clean["pce_pred_smooth"].median(),
            "pr_true": pr_true,
            "pr_pred": pr_pred,
            "e_ref_true": e_ref_true,
            "e_meas_true": e_meas_true,
            "e_adj_true": e_adj_true,
            "total_loss_true_pct": total_loss_true_report * 100,
            "reversible_loss_true_pct": rev_loss_true_report * 100,
            "irreversible_loss_true_pct": irr_loss_true_report * 100,
            "min_gap_true": min_delta_true,
            "scale_true": scale_true,
            "e_ref_pred": e_ref_pred,
            "e_meas_pred": e_meas_pred,
            "e_adj_pred": e_adj_pred,
            "total_loss_pred_pct": total_loss_pred_report * 100,
            "reversible_loss_pred_pct": rev_loss_pred_report * 100,
            "irreversible_loss_pred_pct": irr_loss_pred_report * 100,
            "min_gap_pred": min_delta_pred,
            "scale_pred": scale_pred,
        })

    daily_df = pd.DataFrame(results).set_index("date").sort_index()

    if timestamp_records:
        timestamp_df = pd.concat(timestamp_records).sort_index()
    else:
        timestamp_df = pd.DataFrame()

    print(f"\n  Filtering Summary:")
    print(f"  {'='*50}")
    print(f"  Days with valid degradation analysis: {len(daily_df)}")
    print(f"  Days completely removed (no valid delta_rel): {days_fully_removed}")
    print(f"\n  PCE > eta_ref occurrence (NOT clipped, NOT removed):")
    print(f"    Total timestamps above ref — TRUE: {total_above_ref_true}   LSTM: {total_above_ref_pred}")
    print(f"    Days with >=1 timestamp above ref — TRUE: {days_with_above_ref_true} / {len(daily_df)}"
          f"   LSTM: {days_with_above_ref_pred} / {len(daily_df)}")

    if len(daily_df) > 0:
        avg_pct_true = (daily_df["n_above_ref_true"] / daily_df["n_points_original"] * 100).mean()
        avg_pct_pred = (daily_df["n_above_ref_pred"] / daily_df["n_points_original"] * 100).mean()
        print(f"    Avg % of points above ref per day: {avg_pct_true:.1f}% (TRUE), {avg_pct_pred:.1f}% (LSTM)")

        n_days_e_exceeds_true = daily_df["energy_exceeds_ref_true"].sum()
        n_days_e_exceeds_pred = daily_df["energy_exceeds_ref_pred"].sum()
        print(f"\n  Days where E_meas > E_ref (daily energy exceeds reference):")
        print(f"    TRUE: {n_days_e_exceeds_true} / {len(daily_df)}"
              f"   LSTM: {n_days_e_exceeds_pred} / {len(daily_df)}")

        print(f"\n  Degradation Summary (after filtering):")
        print(f"  {'='*50}")
        for branch, suffix in [("TRUE", "true"), ("LSTM", "pred")]:
            print(f"\n  {branch} branch:")
            print(f"    Mean total loss       : {daily_df[f'total_loss_{suffix}_pct'].mean():.2f}%")
            print(f"    Mean reversible loss  : {daily_df[f'reversible_loss_{suffix}_pct'].mean():.2f}%")
            print(f"    Mean irreversible loss: {daily_df[f'irreversible_loss_{suffix}_pct'].mean():.2f}%")
            print(f"    Mean performance ratio: {daily_df[f'pr_{suffix}'].mean():.4f}")
            print(f"    PR range              : "
                  f"{daily_df[f'pr_{suffix}'].min():.4f} – {daily_df[f'pr_{suffix}'].max():.4f}")

    if len(daily_df) > 0:
        print("\n")
        print("=" * 110)
        print("MONTHLY DEBUG SUMMARY")
        print("=" * 110)

        dbg = daily_df.copy()
        dbg["month"] = dbg.index.to_period("M")

        monthly = (
            dbg.groupby("month")
            .agg(
                days=("n_points_original", "size"),
                total_timestamps=("n_points_after_filter", "sum"),
                true_above_ref=("n_above_ref_true", "sum"),
                lstm_above_ref=("n_above_ref_pred", "sum"),
                true_days_above=("n_above_ref_true", lambda x: (x > 0).sum()),
                lstm_days_above=("n_above_ref_pred", lambda x: (x > 0).sum()),
                total_loss=("total_loss_true_pct", "mean"),
                reversible=("reversible_loss_true_pct", "mean"),
                irreversible=("irreversible_loss_true_pct", "mean"),
            )
        )

        print(monthly.to_string(float_format=lambda x: f"{x:6.2f}"))
        print("=" * 110)

    return daily_df, timestamp_df


def plot_daily_loss_bars_valid_days(daily_df: pd.DataFrame, output_dir: Path,
                                     start_date: str = None, end_date: str = None,
                                     target: str = "Power"):

    if daily_df is None or len(daily_df) == 0:
        print("  WARNING: No daily data for loss bars plot")
        return

    plot_df = daily_df.copy()
    if plot_df.index.tz is not None:
        plot_df.index = plot_df.index.tz_localize(None)

    if start_date:
        plot_df = plot_df[plot_df.index >= pd.Timestamp(start_date)]
    if end_date:
        plot_df = plot_df[plot_df.index <= pd.Timestamp(end_date)]

    plot_df = plot_df[plot_df['reversible_loss_true_pct'].notna()].copy()

    if len(plot_df) == 0:
        print("  WARNING: No valid days in specified date range")
        return

    plot_df['reversible_diff_pct'] = (plot_df['reversible_loss_true_pct']
                                       - plot_df['reversible_loss_pred_pct'])
    plot_df['irreversible_diff_pct'] = (plot_df['irreversible_loss_true_pct']
                                         - plot_df['irreversible_loss_pred_pct'])

    def _error_metrics(true_vals, pred_vals):
        true_vals = np.asarray(true_vals, dtype=float)
        pred_vals = np.asarray(pred_vals, dtype=float)
        resid = true_vals - pred_vals
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        mbe = np.mean(resid)
        r2 = r2_score(true_vals, pred_vals)
        return dict(mae=mae, rmse=rmse, mbe=mbe, r2=r2)

    rev_metrics = _error_metrics(plot_df['reversible_loss_true_pct'],
                                  plot_df['reversible_loss_pred_pct'])
    irr_metrics = _error_metrics(plot_df['irreversible_loss_true_pct'],
                                  plot_df['irreversible_loss_pred_pct'])

    print(f"  Valid days: {len(plot_df)}")
    print(f"  Reversible   — R²={rev_metrics['r2']:.4f}  MAE={rev_metrics['mae']:.3f}%  "
          f"RMSE={rev_metrics['rmse']:.3f}%  MBE={rev_metrics['mbe']:+.3f}%")
    print(f"  Irreversible — R²={irr_metrics['r2']:.4f}  MAE={irr_metrics['mae']:.3f}%  "
          f"RMSE={irr_metrics['rmse']:.3f}%  MBE={irr_metrics['mbe']:+.3f}%")

    fig, (ax_t, ax_p, ax_rev_diff, ax_irr_diff) = plt.subplots(
        4, 1, figsize=(max(14, len(plot_df)*0.15), 18))
    fig.suptitle(f"Daily Loss Decomposition | {target}", fontsize=24, fontweight="bold", y=0.99)

    x_pos = np.arange(len(plot_df))
    valid_dates = plot_df.index.strftime('%Y-%m-%d')
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
               width=bar_width, edgecolor='white', linewidth=0.5, zorder=3)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, color="mediumseagreen", alpha=0.85,
               label="Reversible", width=bar_width, edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_title(panel_title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        ax.set_ylim(0, max(irr_vals + rev_vals) * 1.25)
        ax.grid(True, alpha=0.3, axis="y", linestyle='--', linewidth=0.5, zorder=2)
        ax.set_xlim(-1.0, len(plot_df) + 0.2)
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        mean_rev = rev_vals.mean()
        mean_irr = irr_vals.mean()

        ax.text(
            0.02, 0.96,
            f'Mean Rev = {mean_rev:.2f}%\nMean Irr = {mean_irr:.2f}%',
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            horizontalalignment='left',
            multialignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

    for ax, diff_col, metrics, diff_title in [
        (ax_rev_diff, 'reversible_diff_pct', rev_metrics, "TRUE - LSTM (Reversible)"),
        (ax_irr_diff, 'irreversible_diff_pct', irr_metrics, "TRUE - LSTM (Irreversible)"),
    ]:
        diff_vals = plot_df[diff_col].values
        for pos, val in zip(x_pos, diff_vals):
            color = '#1f77b4' if val > 0 else '#ff7f0e'
            ax.bar(pos, val, color=color, alpha=0.8, width=bar_width,
                   edgecolor='black', linewidth=0.5, zorder=3)
        stats_text = (f"R²   = {metrics['r2']:.3f}\n"
                      f"MAE  = {metrics['mae']:.3f}%\n"
                      f"RMSE = {metrics['rmse']:.3f}%\n"
                      f"MBE  = {metrics['mbe']:+.3f}%")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=15,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
        ax.set_title(diff_title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        max_diff = max(abs(diff_vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
        ax.grid(True, alpha=0.3, axis="y", linestyle='--', linewidth=0.5, zorder=2)
        ax.set_xlim(-1.0, len(plot_df) + 0.2)
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_irr_diff.set_xlabel("Day", fontsize=16, fontweight="bold", labelpad=20)

    loss_handles = [
        Patch(facecolor='tomato', alpha=0.85, edgecolor='white', label='Irreversible'),
        Patch(facecolor='mediumseagreen', alpha=0.85, edgecolor='white', label='Reversible'),
    ]
    diff_handles = [
        Patch(facecolor='#1f77b4', alpha=0.8, edgecolor='black', label='TRUE > LSTM'),
        Patch(facecolor='#ff7f0e', alpha=0.8, edgecolor='black', label='TRUE < LSTM'),
    ]
    fig.legend(handles=loss_handles + diff_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=20,
               frameon=True, edgecolor='black')

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.96])

    start_str = plot_df.index.min().strftime('%Y%m%d')
    end_str = plot_df.index.max().strftime('%Y%m%d')
    filename = output_dir / f"daily_loss_bars_valid_only_{start_str}_{end_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename.name}")


def plot_monthly_loss_bars_old(daily_df: pd.DataFrame, df_full: pd.DataFrame,
                            output_dir: Path, target: str = "Power"):
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

    monthly['total_loss_true_pct'] = (monthly['reversible_loss_true_pct']
                                       + monthly['irreversible_loss_true_pct'])
    monthly['total_loss_pred_pct'] = (monthly['reversible_loss_pred_pct']
                                       + monthly['irreversible_loss_pred_pct'])
    monthly['reversible_diff_pct'] = (monthly['reversible_loss_true_pct']
                                       - monthly['reversible_loss_pred_pct'])
    monthly['irreversible_diff_pct'] = (monthly['irreversible_loss_true_pct']
                                         - monthly['irreversible_loss_pred_pct'])

    def _error_metrics(true_vals, pred_vals):
        true_vals = np.asarray(true_vals, dtype=float)
        pred_vals = np.asarray(pred_vals, dtype=float)
        resid = true_vals - pred_vals
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        mbe = np.mean(resid)
        r2 = r2_score(true_vals, pred_vals)
        return dict(mae=mae, rmse=rmse, mbe=mbe, r2=r2)

    rev_metrics = _error_metrics(monthly['reversible_loss_true_pct'],
                                  monthly['reversible_loss_pred_pct'])
    irr_metrics = _error_metrics(monthly['irreversible_loss_true_pct'],
                                  monthly['irreversible_loss_pred_pct'])

    x_pos = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(f'Monthly Mean Loss Decomposition | {target}',
                 fontsize=24, fontweight='bold', y=1.02)
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
               edgecolor='white', linewidth=0.5)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, label="Reversible",
               color="mediumseagreen", alpha=0.85, edgecolor='white', linewidth=0.5)
        for i, (_, row) in enumerate(monthly.iterrows()):
            n_days = day_counts[row['year_month']]
            bar_h = total_vals[i]
            ax.text(i, bar_h + bar_h * 0.05, f'n={n_days}',
                    ha='center', va='bottom', fontsize=14, rotation=90,
                    color='black', alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        ax.set_title(panel_title, fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel("Energy Loss (%)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=16, fontweight='bold', labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax_i, diff_col, metrics, diff_title in [
        (2, 'reversible_diff_pct', rev_metrics, "TRUE - LSTM (Reversible)"),
        (3, 'irreversible_diff_pct', irr_metrics, "TRUE - LSTM (Irreversible)"),
    ]:
        ax = axes[ax_i]
        diff_vals = monthly[diff_col].values
        for pos, val in zip(x_pos, diff_vals):
            color = '#1f77b4' if val > 0 else '#ff7f0e'
            ax.bar(pos, val, color=color, alpha=0.8, width=0.6,
                   edgecolor='black', linewidth=0.5, zorder=3)
        stats_text = (f"R²   = {metrics['r2']:.3f}\n"
                      f"MAE  = {metrics['mae']:.3f}%")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(diff_title, fontsize=20, fontweight='bold')
        ax.set_ylabel("Energy Loss Diff (%)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=16, fontweight='bold', labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        max_diff = max(abs(diff_vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)

    loss_handles = [
        Patch(facecolor='tomato', alpha=0.85, edgecolor='white', label='Irreversible'),
        Patch(facecolor='mediumseagreen', alpha=0.85, edgecolor='white', label='Reversible'),
    ]
    diff_handles = [
        Patch(facecolor='#1f77b4', alpha=0.8, edgecolor='black', label='TRUE > LSTM'),
        Patch(facecolor='#ff7f0e', alpha=0.8, edgecolor='black', label='TRUE < LSTM'),
    ]
    fig.legend(handles=loss_handles + diff_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=20,
               frameon=True, edgecolor='black')

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0, rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / "monthly_loss_bars_with_diffs.png", dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Saved: monthly_loss_bars_with_diffs.png")

    print(f"\n  Monthly Loss Summary  [{target}]:")
    print(f"  {'Month':<12} {'Days':>5} "
          f"{'Total True':>11} {'Rev True':>10} {'Irr True':>10} "
          f"{'Total LSTM':>11} {'Rev LSTM':>10} {'Irr LSTM':>10} "
          f"{'Rev Diff':>10} {'Irr Diff':>10}")
    print("-" * 115)
    for _, row in monthly.iterrows():
        month = row['year_month']
        n_days = day_counts[month]
        print(
            f"  {month:<12} {n_days:>5} "
            f"{row['total_loss_true_pct']:>10.2f}% "
            f"{row['reversible_loss_true_pct']:>9.2f}% "
            f"{row['irreversible_loss_true_pct']:>9.2f}% "
            f"{row['total_loss_pred_pct']:>10.2f}% "
            f"{row['reversible_loss_pred_pct']:>9.2f}% "
            f"{row['irreversible_loss_pred_pct']:>9.2f}% "
            f"{row['reversible_diff_pct']:>9.2f}% "
            f"{row['irreversible_diff_pct']:>9.2f}%"
        )
    print(f"\n  Monthly Error Metrics (LSTM vs TRUE)  [{target}]:")
    print(f"  {'Component':<14} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MBE':>8}")
    print("  " + "-" * 50)
    print(f"  {'Reversible':<14} {rev_metrics['r2']:>8.4f} {rev_metrics['mae']:>7.3f}% "
          f"{rev_metrics['rmse']:>7.3f}% {rev_metrics['mbe']:>+7.3f}%")
    print(f"  {'Irreversible':<14} {irr_metrics['r2']:>8.4f} {irr_metrics['mae']:>7.3f}% "
          f"{irr_metrics['rmse']:>7.3f}% {irr_metrics['mbe']:>+7.3f}%")

def plot_monthly_loss_bars(daily_df: pd.DataFrame, df_full: pd.DataFrame,
                            output_dir: Path, target: str = "Power"):
    dfc = daily_df.copy()
    dfc["year_month"] = dfc.index.strftime("%Y-%m")
    day_counts = dfc.groupby("year_month").size()

    # ── Energy-weighted monthly aggregation (ratio of sums, not mean of ratios) ──
    # total_loss_pct   = (sum(E_ref) - sum(E_meas)) / sum(E_ref) * 100
    # reversible_pct   = (sum(E_ref) - sum(E_adj))  / sum(E_ref) * 100
    # irreversible_pct = (sum(E_adj) - sum(E_meas)) / sum(E_ref) * 100
    # This weights each day by its own E_ref (proportional to that day's valid
    # timestamp count), instead of averaging already-computed daily percentages
    # unweighted, so sparse-data days no longer count equally with data-rich days.
    energy_sums = dfc.groupby("year_month").agg(
        e_ref_true_sum=("e_ref_true", "sum"),
        e_meas_true_sum=("e_meas_true", "sum"),
        e_adj_true_sum=("e_adj_true", "sum"),
        e_ref_pred_sum=("e_ref_pred", "sum"),
        e_meas_pred_sum=("e_meas_pred", "sum"),
        e_adj_pred_sum=("e_adj_pred", "sum"),
    ).reset_index()

    def _safe_ratio(numer, denom):
        return np.where(denom > 0, numer / denom * 100.0, np.nan)

    monthly = energy_sums.copy()
    monthly['total_loss_true_pct'] = _safe_ratio(
        monthly['e_ref_true_sum'] - monthly['e_meas_true_sum'], monthly['e_ref_true_sum'])
    monthly['reversible_loss_true_pct'] = _safe_ratio(
        monthly['e_ref_true_sum'] - monthly['e_adj_true_sum'], monthly['e_ref_true_sum'])
    monthly['irreversible_loss_true_pct'] = _safe_ratio(
        monthly['e_adj_true_sum'] - monthly['e_meas_true_sum'], monthly['e_ref_true_sum'])

    monthly['total_loss_pred_pct'] = _safe_ratio(
        monthly['e_ref_pred_sum'] - monthly['e_meas_pred_sum'], monthly['e_ref_pred_sum'])
    monthly['reversible_loss_pred_pct'] = _safe_ratio(
        monthly['e_ref_pred_sum'] - monthly['e_adj_pred_sum'], monthly['e_ref_pred_sum'])
    monthly['irreversible_loss_pred_pct'] = _safe_ratio(
        monthly['e_adj_pred_sum'] - monthly['e_meas_pred_sum'], monthly['e_ref_pred_sum'])

    # Clip negative loss components to 0 for reporting, same convention as
    # the daily-level computation in compute_daily_degradation_notebook_method.
    for col in ['total_loss_true_pct', 'reversible_loss_true_pct', 'irreversible_loss_true_pct',
                'total_loss_pred_pct', 'reversible_loss_pred_pct', 'irreversible_loss_pred_pct']:
        monthly[col] = monthly[col].clip(lower=0)

    monthly['reversible_diff_pct'] = (monthly['reversible_loss_true_pct']
                                       - monthly['reversible_loss_pred_pct'])
    monthly['irreversible_diff_pct'] = (monthly['irreversible_loss_true_pct']
                                         - monthly['irreversible_loss_pred_pct'])

    def _error_metrics(true_vals, pred_vals):
        true_vals = np.asarray(true_vals, dtype=float)
        pred_vals = np.asarray(pred_vals, dtype=float)
        resid = true_vals - pred_vals
        mae = np.mean(np.abs(resid))
        rmse = np.sqrt(np.mean(resid ** 2))
        mbe = np.mean(resid)
        r2 = r2_score(true_vals, pred_vals)
        return dict(mae=mae, rmse=rmse, mbe=mbe, r2=r2)

    rev_metrics = _error_metrics(monthly['reversible_loss_true_pct'],
                                  monthly['reversible_loss_pred_pct'])
    irr_metrics = _error_metrics(monthly['irreversible_loss_true_pct'],
                                  monthly['irreversible_loss_pred_pct'])

    x_pos = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(f'Monthly Energy-Weighted Loss Decomposition | {target}',
                 fontsize=24, fontweight='bold', y=1.02)
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
               edgecolor='white', linewidth=0.5)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, label="Reversible",
               color="mediumseagreen", alpha=0.85, edgecolor='white', linewidth=0.5)
        for i, (_, row) in enumerate(monthly.iterrows()):
            n_days = day_counts[row['year_month']]
            bar_h = total_vals[i]
            ax.text(i, bar_h + bar_h * 0.05, f'n={n_days}',
                    ha='center', va='bottom', fontsize=14, rotation=90,
                    color='black', alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        ax.set_title(panel_title, fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel("Energy Loss (%, energy-weighted)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=16, fontweight='bold', labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax_i, diff_col, metrics, diff_title in [
        (2, 'reversible_diff_pct', rev_metrics, "TRUE - LSTM (Reversible)"),
        (3, 'irreversible_diff_pct', irr_metrics, "TRUE - LSTM (Irreversible)"),
    ]:
        ax = axes[ax_i]
        diff_vals = monthly[diff_col].values
        for pos, val in zip(x_pos, diff_vals):
            color = '#1f77b4' if val > 0 else '#ff7f0e'
            ax.bar(pos, val, color=color, alpha=0.8, width=0.6,
                   edgecolor='black', linewidth=0.5, zorder=3)
        stats_text = (f"R²   = {metrics['r2']:.3f}\n"
                      f"MAE  = {metrics['mae']:.3f}%")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(diff_title, fontsize=20, fontweight='bold')
        ax.set_ylabel("Energy Loss Diff (%)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=16, fontweight='bold', labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        max_diff = max(abs(diff_vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)

    loss_handles = [
        Patch(facecolor='tomato', alpha=0.85, edgecolor='white', label='Irreversible'),
        Patch(facecolor='mediumseagreen', alpha=0.85, edgecolor='white', label='Reversible'),
    ]
    diff_handles = [
        Patch(facecolor='#1f77b4', alpha=0.8, edgecolor='black', label='TRUE > LSTM'),
        Patch(facecolor='#ff7f0e', alpha=0.8, edgecolor='black', label='TRUE < LSTM'),
    ]
    fig.legend(handles=loss_handles + diff_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=20,
               frameon=True, edgecolor='black')

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0, rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / "monthly_loss_bars_with_diffs.png", dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  Saved: monthly_loss_bars_with_diffs.png")

    print(f"\n  Monthly Loss Summary (energy-weighted)  [{target}]:")
    print(f"  {'Month':<12} {'Days':>5} "
          f"{'Total True':>11} {'Rev True':>10} {'Irr True':>10} "
          f"{'Total LSTM':>11} {'Rev LSTM':>10} {'Irr LSTM':>10} "
          f"{'Rev Diff':>10} {'Irr Diff':>10}")
    print("-" * 115)
    for _, row in monthly.iterrows():
        month = row['year_month']
        n_days = day_counts[month]
        print(
            f"  {month:<12} {n_days:>5} "
            f"{row['total_loss_true_pct']:>10.2f}% "
            f"{row['reversible_loss_true_pct']:>9.2f}% "
            f"{row['irreversible_loss_true_pct']:>9.2f}% "
            f"{row['total_loss_pred_pct']:>10.2f}% "
            f"{row['reversible_loss_pred_pct']:>9.2f}% "
            f"{row['irreversible_loss_pred_pct']:>9.2f}% "
            f"{row['reversible_diff_pct']:>9.2f}% "
            f"{row['irreversible_diff_pct']:>9.2f}%"
        )
    print(f"\n  Monthly Error Metrics (LSTM vs TRUE)  [{target}]:")
    print(f"  {'Component':<14} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'MBE':>8}")
    print("  " + "-" * 50)
    print(f"  {'Reversible':<14} {rev_metrics['r2']:>8.4f} {rev_metrics['mae']:>7.3f}% "
          f"{rev_metrics['rmse']:>7.3f}% {rev_metrics['mbe']:>+7.3f}%")
    print(f"  {'Irreversible':<14} {irr_metrics['r2']:>8.4f} {irr_metrics['mae']:>7.3f}% "
          f"{irr_metrics['rmse']:>7.3f}% {irr_metrics['mbe']:>+7.3f}%")
# ---------------------------------------------------------------------------
# Counterfactual analyzer
# ---------------------------------------------------------------------------
class CounterfactualAgeAnalyzer:
    """
    Loads a trained multi-output LSTM, runs inference over the FULL merged
    train+test dataset, and compares normal predictions against a
    counterfactual scenario where days_since_install_<family> is clamped
    to a constant value for every timestep of every sequence.

    Also builds a PCE reference surface directly from the age=1
    counterfactual's predicted PCE, as an alternative to the
    p95-of-reference-period + linear-extrapolation method.
    """

    def __init__(
        self,
        model_path: str,
        model_info_path: str,
        train_data_path: str,
        test_data_path: str,
        target_scaler_path: str,
        feature_scaler_path: str,
        panel_temp_path: Optional[str] = None,
        bad_days_list: Optional[List[str]] = None,
        output_dir: str = "./counterfactual_age_results",
        age_col: str = "days_since_install_perovskite",
    ):
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.target_scaler_path = Path(target_scaler_path)
        self.feature_scaler_path = Path(feature_scaler_path)
        self.panel_temp_path = Path(panel_temp_path) if panel_temp_path else None
        self.bad_days_list = bad_days_list or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.age_col = age_col

        self._cf_cache: Dict[float, Dict] = {}

        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        with open(self.model_info_path, "rb") as f:
            self.info = pickle.load(f)

        self.df_scaled = merge_train_test_parquets(
            self.train_data_path, self.test_data_path
        )

        with open(self.target_scaler_path, "rb") as f:
            sd = pickle.load(f)
        with open(self.feature_scaler_path, "rb") as f:
            fsd = pickle.load(f)

        self.target_cols = self.info["data_config"]["features"]["target_cols"]
        self.feature_cols = self.info["data_config"]["features"]["feature_cols"]
        self.target_scalers = {t: sd.get(t, sd.get("scaler")) for t in self.target_cols}
        self.feature_scalers = fsd.get("feature_scalers", {})

        if self.age_col not in self.feature_cols:
            raise ValueError(
                f"'{self.age_col}' not in model's feature_cols: {self.feature_cols}. "
                f"Confirm this trained model actually includes the age feature."
            )
        if self.age_col not in self.feature_scalers:
            raise ValueError(f"'{self.age_col}' not found in feature_scalers.")

        arch = self.info["architecture"]
        self.horizon = self.info["data_config"]["window_horizon"]["horizon"]
        self.window = self.info["data_config"]["window_horizon"]["window"]
        use_bad = self.info["data_config"]["features"]["use_bad_day"]
        mask_bad = self.info["data_config"]["features"]["mask_bad_days"]
        bad_map = self.info.get("bad_day_mapping", {})

        print(f"\n[CONFIG] use_bad_day={use_bad}, mask_bad_days={mask_bad}")
        if use_bad:
            print("  WARNING: use_bad_day=True \u2014 bad_day_* columns are part of feature_cols. "
                  "This analyzer does not currently override them for the counterfactual pass.")

        self.device = torch.device("cpu")
        self.model = MultiOutputLSTMModel(
            input_size=len(self.feature_cols),
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            num_outputs=len(self.target_cols),
            dropout=arch["dropout"],
            forecast_steps=self.horizon,
        )
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )
        self.model.to(self.device).eval()

        self.full_ds = NonOverlappingMultiOutputDataset(
            self.df_scaled,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            window=self.window,
            horizon=self.horizon,
            use_bad_day=use_bad,
            mask_bad_days=mask_bad,
            bad_day_mapping=bad_map,
        )
        print(f"  Total windows: {len(self.full_ds)}")

        print("\nRunning inference (normal)...")
        self.full_preds = run_inference(self.model, self.full_ds, self.device)
        print(f"  Predictions shape: {self.full_preds.shape}")

        age_scaler = self.feature_scalers[self.age_col]
        self.real_age_series = pd.Series(
            _inverse_transform(self.df_scaled[self.age_col].values, age_scaler),
            index=self.df_scaled.index,
        )

    # ------------------------------------------------------------------
    def _build_counterfactual_dataset(self, constant_age_days: float) -> pd.DataFrame:
        age_scaler = self.feature_scalers[self.age_col]
        scaled_age = float(age_scaler.transform([[constant_age_days]])[0, 0])

        df_cf = self.df_scaled.copy()
        df_cf[self.age_col] = scaled_age
        return df_cf

    def run_counterfactual(self, constant_age_days: float = 1.0) -> Dict:
        print(f"\n{'='*70}")
        print(f"COUNTERFACTUAL RUN: age={constant_age_days} days (full merged train+test)")
        print(f"{'='*70}")

        use_bad = self.info["data_config"]["features"]["use_bad_day"]
        mask_bad = self.info["data_config"]["features"]["mask_bad_days"]
        bad_map = self.info.get("bad_day_mapping", {})

        df_cf = self._build_counterfactual_dataset(constant_age_days)

        cf_ds = NonOverlappingMultiOutputDataset(
            df_cf,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            window=self.window,
            horizon=self.horizon,
            use_bad_day=use_bad,
            mask_bad_days=mask_bad,
            bad_day_mapping=bad_map,
        )

        print("\nRunning inference (counterfactual)...")
        cf_preds = run_inference(self.model, cf_ds, self.device)
        print(f"  Predictions shape: {cf_preds.shape}")

        df_normal_aligned = build_aligned_df(
            self.full_ds, self.full_preds, self.df_scaled, self.info,
            self.target_scalers, self.feature_scalers, bad_days_list=self.bad_days_list,
        )
        df_cf_aligned = build_aligned_df(
            cf_ds, cf_preds, df_cf, self.info,
            self.target_scalers, self.feature_scalers, bad_days_list=self.bad_days_list,
        )

        print("\n  Computing PCE (normal)...")
        df_normal_pce = calculate_pce(df_normal_aligned.copy())
        print("\n  Computing PCE (counterfactual)...")
        df_cf_pce = calculate_pce(df_cf_aligned.copy())

        comparison_rows = []
        for tname in self.target_cols:
            pred_col = f"pred_{tname}"
            common_idx = df_normal_aligned.index.intersection(df_cf_aligned.index)

            normal_vals = df_normal_aligned.loc[common_idx, pred_col]
            cf_vals = df_cf_aligned.loc[common_idx, pred_col]

            for ts in common_idx:
                comparison_rows.append({
                    "timestamp": ts,
                    "target": tname,
                    "pred_normal": normal_vals.loc[ts],
                    "pred_counterfactual": cf_vals.loc[ts],
                    "delta": cf_vals.loc[ts] - normal_vals.loc[ts],
                })

        comp_df = pd.DataFrame(comparison_rows)
        csv_path = self.output_dir / f"counterfactual_age{constant_age_days}.csv"
        comp_df.to_csv(csv_path, index=False)
        print(f"[SAVED] {csv_path}")

        summary = (
            comp_df.groupby("target")["delta"]
            .agg(["mean", "std", "min", "max"])
            .rename(columns={"mean": "mean_delta", "std": "std_delta",
                              "min": "min_delta", "max": "max_delta"})
        )
        summary_path = self.output_dir / f"counterfactual_age{constant_age_days}_summary.csv"
        summary.to_csv(summary_path)
        print(f"[SAVED] {summary_path}")
        print("\n[SUMMARY] Mean (counterfactual - normal) power delta per target:")
        print(summary)

        self._save_plots(comp_df, constant_age_days)
        self._save_pce_plot(df_normal_pce, df_cf_pce, constant_age_days)

        result = {
            "comparison_df": comp_df,
            "summary": summary,
            "df_normal_aligned": df_normal_aligned,
            "df_counterfactual_aligned": df_cf_aligned,
            "df_normal_pce": df_normal_pce,
            "df_cf_pce": df_cf_pce,
        }
        self._cf_cache[constant_age_days] = result
        return result

    # ------------------------------------------------------------------
    def _save_plots(self, comp_df: pd.DataFrame, constant_age_days: float):
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for tname in self.target_cols:
            sub = comp_df[comp_df["target"] == tname]
            if sub.empty:
                continue

            normal = sub["pred_normal"].values
            cf = sub["pred_counterfactual"].values
            delta = sub["delta"].values

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            axes[0].scatter(normal, cf, s=3, alpha=0.4)
            lims = [min(normal.min(), cf.min()), max(normal.max(), cf.max())]
            axes[0].plot(lims, lims, "r--", linewidth=1, label="y = x")
            axes[0].set_xlabel("Predicted power (normal age)")
            axes[0].set_ylabel(f"Predicted power (age={constant_age_days}d)")
            axes[0].set_title(f"{tname}: Normal vs Counterfactual")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].hist(delta, bins=60, alpha=0.75)
            axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
            axes[1].axvline(delta.mean(), color="green", linewidth=1,
                             label=f"mean={delta.mean():.3f}")
            axes[1].set_xlabel("Delta (counterfactual - normal)")
            axes[1].set_title(f"{tname}: Prediction Shift Distribution")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.suptitle(f"{tname} | age set to {constant_age_days} days | full merged train+test")
            plt.tight_layout()
            fpath = plots_dir / f"{tname}_age{constant_age_days}.png"
            plt.savefig(fpath, dpi=150)
            plt.close()
            print(f"[SAVED] {fpath}")

    # ------------------------------------------------------------------
    # Plot 2: PCE over time -- true vs predicted (this age scenario)
    # ------------------------------------------------------------------
    def _save_pce_plot(self, df_normal_pce: pd.DataFrame, df_cf_pce: pd.DataFrame, constant_age_days: float):
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        target_name = _get_target_name(df_normal_pce)

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(df_normal_pce.index, df_normal_pce["pce_true"],
                label="PCE true", color="black", linewidth=0.8, alpha=0.6)
        ax.plot(df_normal_pce.index, df_normal_pce["pce_pred"],
                label="PCE pred (normal age)", color="tab:blue", linewidth=0.8, alpha=0.8)
        ax.plot(df_cf_pce.index, df_cf_pce["pce_pred"],
                label=f"PCE pred (age={constant_age_days}d)", color="tab:orange",
                linewidth=0.8, alpha=0.8)

        ax.set_xlabel("Time")
        ax.set_ylabel("PCE (%)")
        ax.set_title(f"{target_name}: PCE over time \u2014 true vs normal vs age={constant_age_days}d counterfactual")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fpath = plots_dir / f"{target_name}_pce_over_time_age{constant_age_days}.png"
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"[SAVED] {fpath}")

    # ------------------------------------------------------------------
    # Plot 3: D(t) = (P_normal - P_age_ref) / P_normal * 100, vs. real age (days)
    # ------------------------------------------------------------------
    def compute_and_plot_degradation_curve(
        self,
        reference_age_days: float = 1.0,
        min_power_for_ratio: float = 20.0,
    ) -> pd.DataFrame:
        """
        D(t) = (P_normal(t) - P_reference_age(t)) / P_normal(t) * 100

        i.e. percentage power LOST relative to a same-conditions panel that
        was installed `reference_age_days` days ago (default: 1 day, "as if
        installed today"). Positive D(t) => real (aged) panel underperforms
        the fresh-install counterfactual on that timestep; negative D(t) =>
        it outperforms it.

        Requires run_counterfactual(reference_age_days) to have been called
        already (or will call it now if not cached).

        Produces a single figure with two stacked panels:
          Top panel: for each month, four adjacent box plots (IQR ranges) of
                     true value, normal prediction, age=reference_age_days
                     prediction, and Irr for that month.
          Bottom panel: monthly box plot of D(t) itself (one box per month),
                     replacing the previous scatter + weekly-mean line.
        """
        if reference_age_days not in self._cf_cache:
            print(f"[INFO] No cached run for reference_age_days={reference_age_days}, running it now...")
            self.run_counterfactual(reference_age_days)

        cf_result = self._cf_cache[reference_age_days]
        df_normal = cf_result["df_normal_aligned"]
        df_ref = cf_result["df_counterfactual_aligned"]

        tname = self.target_cols[0] if len(self.target_cols) == 1 else None
        if tname is None:
            raise ValueError(
                "compute_and_plot_degradation_curve currently supports a single "
                f"target; got target_cols={self.target_cols}."
            )

        pred_col = f"pred_{tname}"
        common_idx = df_normal.index.intersection(df_ref.index)

        p_normal = df_normal.loc[common_idx, pred_col]
        p_ref = df_ref.loc[common_idx, pred_col]

        valid = (p_normal.abs() >= min_power_for_ratio) & p_normal.notna() & p_ref.notna()

        d_t = pd.Series(np.nan, index=common_idx)
        d_t.loc[valid] = (p_normal.loc[valid] - p_ref.loc[valid]) / p_normal.loc[valid] * 100.0

        real_age_aligned = self.real_age_series.reindex(common_idx)

        d_df = pd.DataFrame({
            "timestamp": common_idx,
            "real_age_days": real_age_aligned.values,
            "P_normal": p_normal.values,
            f"P_age{reference_age_days}": p_ref.values,
            "D_t_pct": d_t.values,
            "Irr": df_normal.loc[common_idx, "Irr"].values if "Irr" in df_normal.columns else np.nan,
            "temp_C": df_normal.loc[common_idx, "temp_C"].values if "temp_C" in df_normal.columns else np.nan,
        }).set_index("timestamp")

        csv_path = self.output_dir / f"degradation_Dt_ref_age{reference_age_days}.csv"
        d_df.to_csv(csv_path)
        print(f"[SAVED] {csv_path}")

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        d_valid = d_df.dropna(subset=["real_age_days", "D_t_pct"])

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.scatter(d_valid["real_age_days"], d_valid["D_t_pct"], s=3, alpha=0.25, color="tab:blue")

        d_valid_sorted = d_valid.sort_values("real_age_days")
        bin_edges = np.arange(0, d_valid_sorted["real_age_days"].max() + 7, 7)
        d_valid_sorted["age_bin"] = pd.cut(d_valid_sorted["real_age_days"], bins=bin_edges)
        binned_mean = d_valid_sorted.groupby("age_bin", observed=True)["D_t_pct"].mean()
        bin_centers = [interval.mid for interval in binned_mean.index]

        ax.plot(bin_centers, binned_mean.values, color="red", linewidth=2,
                label="Weekly mean D(t)")
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xlabel("Real days since install")
        ax.set_ylabel("D(t) = (P_normal \u2212 P_ref) / P_normal \u00d7 100  [%]")
        ax.set_title(
            f"{tname}: Estimated power loss vs. as-if-installed-{reference_age_days:.0f}d-ago counterfactual"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fpath = plots_dir / f"{tname}_Dt_vs_real_age_ref{reference_age_days}.png"
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"[SAVED] {fpath}")

        self._save_dt_heatmap(d_df, tname, reference_age_days)

        return d_df

    # ------------------------------------------------------------------
    def _save_dt_heatmap(
        self,
        d_df: pd.DataFrame,
        target_name: str,
        reference_age_days: float,
        n_irr_bins: int = 15,
        n_temp_bins: int = 15,
    ):
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        sub = d_df.dropna(subset=["Irr", "temp_C", "D_t_pct"]).copy()
        if sub.empty:
            print("[WARNING] No valid rows for D(t) Irr/temp heatmap \u2014 skipping.")
            return

        irr_bins = pd.cut(sub["Irr"], bins=n_irr_bins)
        temp_bins = pd.cut(sub["temp_C"], bins=n_temp_bins)

        pivot = (
            sub.assign(irr_bin=irr_bins, temp_bin=temp_bins)
            .groupby(["irr_bin", "temp_bin"], observed=True)["D_t_pct"]
            .mean()
            .unstack("temp_bin")
        )

        pivot = pivot.reindex(sorted(pivot.index, key=lambda iv: iv.left))
        pivot = pivot[sorted(pivot.columns, key=lambda iv: iv.left)]

        irr_labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in pivot.index]
        temp_labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(10, n_temp_bins * 0.6), max(8, n_irr_bins * 0.5)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", origin="lower")

        ax.set_xticks(range(len(temp_labels)))
        ax.set_xticklabels(temp_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(irr_labels)))
        ax.set_yticklabels(irr_labels, fontsize=7)

        ax.set_xlabel("Temperature bin (\u00b0C)")
        ax.set_ylabel("Irradiance bin (W/m\u00b2)")
        ax.set_title(
            f"{target_name}: Mean D(t) [%] by Irradiance \u00d7 Temperature "
            f"(ref age={reference_age_days}d)"
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean D(t) (%)")

        plt.tight_layout()
        fpath = plots_dir / f"{target_name}_Dt_Irr_Temp_heatmap_ref{reference_age_days}.png"
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"[SAVED] {fpath}")

    # ------------------------------------------------------------------
    # NEW: age=1-based PCE reference surface
    # ------------------------------------------------------------------
    def build_and_plot_age1_reference_surface(
        self,
        reference_age_days: float = 1.0,
        n_irr_bins: int = 10,
        n_temp_bins: int = 10,
    ) -> dict:
        """
        Builds the eta_ref lookup table directly from the age=1
        counterfactual's predicted PCE (mean per Irr x Temp bin, no
        extrapolation), plots the reference-surface heatmap and its
        coverage heatmap, computes eta_ref for every row of the age=1
        aligned frame via get_reference_pce, and saves the result.

        This is the direct alternative to build_reference_surface() in
        pce_surface_analysis.py (which used p95-of-first-120-days +
        linear extrapolation on the TRUE measured signal).
        """
        if reference_age_days not in self._cf_cache:
            print(f"[INFO] No cached run for reference_age_days={reference_age_days}, running it now...")
            self.run_counterfactual(reference_age_days)

        cf_result = self._cf_cache[reference_age_days]
        df_age1_pce = cf_result["df_cf_pce"]
        df_normal = cf_result["df_normal_aligned"]

        tname = _get_target_name(df_age1_pce)

        ref_surface = build_reference_surface_from_age1(
            df_age1_pce,
            n_irr_bins=n_irr_bins,
            n_temp_bins=n_temp_bins,
            min_irr_ref=MIN_IRR_FOR_PCE,
            df_full_for_bounds=df_normal,
        )

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_reference_surface_age1(ref_surface, plots_dir, target=tname)
        plot_coverage_heatmap_age1(ref_surface, plots_dir, target=tname)

        eta_ref, is_extrap = get_reference_pce(df_age1_pce, ref_surface)
        df_age1_pce = df_age1_pce.copy()
        df_age1_pce["eta_ref_age1"] = eta_ref
        df_age1_pce["is_extrap_age1"] = is_extrap

        csv_path = self.output_dir / f"age1_reference_surface_lookup_age{reference_age_days}.csv"
        ref_surface["lookup_table"].to_csv(csv_path)
        print(f"[SAVED] {csv_path}")

        aligned_csv_path = self.output_dir / f"age1_pce_with_eta_ref_age{reference_age_days}.csv"
        df_age1_pce.to_csv(aligned_csv_path)
        print(f"[SAVED] {aligned_csv_path}")

        return {
            "ref_surface": ref_surface,
            "df_age1_pce_with_eta_ref": df_age1_pce,
        }

    # ------------------------------------------------------------------
    # NEW: per-day reversible/irreversible loss decomposition, using the
    # age=1 reference surface as eta_ref, plus daily and monthly plots.
    # ------------------------------------------------------------------
    def compute_and_plot_loss_decomposition(
        self,
        reference_age_days: float = 1.0,
        n_irr_bins: int = 10,
        n_temp_bins: int = 10,
    ) -> Dict:
        """
        Computes per-day reversible/irreversible energy loss (TRUE vs LSTM
        branches, both measured against the SAME age=1-derived eta_ref),
        using compute_daily_degradation_notebook_method, then produces:
          - the daily bar-chart plot (plot_daily_loss_bars_valid_days)
          - the monthly bar-chart plot + printed summary (plot_monthly_loss_bars)

        Requires build_and_plot_age1_reference_surface(reference_age_days)
        to have been run already (or runs it now if not cached).
        """
        age1_result = self.build_and_plot_age1_reference_surface(
            reference_age_days=reference_age_days,
            n_irr_bins=n_irr_bins,
            n_temp_bins=n_temp_bins,
        )
        ref_surface = age1_result["ref_surface"]

        cf_result = self._cf_cache[reference_age_days]
        df_normal = cf_result["df_normal_aligned"]
        tname = _get_target_name(df_normal)

        # Recompute PCE fresh on df_normal (calculate_pce mutates/adds columns
        # in place, so work on a copy to avoid clobbering cached results).
        df_normal_pce = calculate_pce(df_normal.copy())

        # eta_ref for the TRUE/LSTM branch both come from the SAME age=1
        # reference surface — matching SimplePCEAnalyzer's convention of
        # eta_ref_pred = eta_ref_true.copy() for the observed-only method.
        eta_ref, is_extrap = get_reference_pce(df_normal_pce, ref_surface)
        df_normal_pce["eta_ref_true"] = eta_ref
        df_normal_pce["eta_ref_pred"] = eta_ref
        df_normal_pce["is_extrap"] = is_extrap

        print("\nComputing per-day reversible/irreversible loss decomposition "
              "(eta_ref from age=1 reference surface)...")
        daily_df, timestamp_df = compute_daily_degradation_notebook_method(df_normal_pce)

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if len(daily_df) > 0:
            print("\nPlotting daily loss bars (valid days only)...")
            plot_daily_loss_bars_valid_days(daily_df, plots_dir, target=tname)

            print("\nPlotting monthly loss bars...")
            plot_monthly_loss_bars(daily_df, df_normal_pce, plots_dir, target=tname)

            daily_csv = self.output_dir / f"daily_degradation_age1ref_{reference_age_days}.csv"
            daily_df.to_csv(daily_csv)
            print(f"[SAVED] {daily_csv}")
        else:
            print("[WARNING] No valid days after filtering — skipping loss plots.")

        return {
            "daily_df": daily_df,
            "timestamp_df": timestamp_df,
            "df_normal_pce_with_eta_ref": df_normal_pce,
        }


def main():
    BASE_PATH = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_26_133601"
    analyzer = CounterfactualAgeAnalyzer(
        model_path          = f"{BASE_PATH}/lstm_results/26.09.2026.133728_model_multi.pt",
        model_info_path     = f"{BASE_PATH}/lstm_results/model_info_multi.pkl",
        train_data_path     = f"{BASE_PATH}/training_data/train_scaled.parquet",
        test_data_path      = f"{BASE_PATH}/training_data/test_scaled.parquet",
        target_scaler_path  = f"{BASE_PATH}/training_data/p_scalers.pkl",
        feature_scaler_path = f"{BASE_PATH}/training_data/scalers.pkl",
        panel_temp_path     = f"{BASE_PATH}/merged_pv_dwd_step.csv",
        bad_days_list       = [],
        output_dir          = f"{BASE_PATH}/lstm_results/counterfactua_inference",
        age_col             = "days_since_install_perovskite",
    )

    for age_days in [1.0]:
        analyzer.run_counterfactual(constant_age_days=age_days)

    analyzer.compute_and_plot_degradation_curve(reference_age_days=1.0)

    analyzer.build_and_plot_age1_reference_surface(reference_age_days=1.0)

    analyzer.compute_and_plot_loss_decomposition(reference_age_days=1.0)


if __name__ == "__main__":
    main()