import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
import traceback

from train_lstm_model import (
    MultiOutputLSTMModel,
    NonOverlappingMultiOutputDataset,
    build_continuous_series_multi_output,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Physical bounds used to clamp perturbed feature values back to realistic ranges.
PHYSICAL_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "Irr": (0.0,   None),
    "temp_C": (-50.0, 100.0),
    "humidity": (0.0,   None),
    "precip_mm": (0.0,   None),
    "precip_indicator": (0.0,   1.0),
    "cloud_cover": (0.0,   8.0),
}

# Cyclical features (sin/cos) and binary indicators are excluded from perturbation, because perturbing them independently breaks their mathematical relationship.
SKIP_PERTURBATION_SUFFIXES = ("_sin", "_cos")
SKIP_PERTURBATION_EXACT = ("precip_indicator",)

def _save(fig: plt.Figure, path: Path, dpi: int = 150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def is_skipped_feature(name: str) -> bool:
    """
    Return True if the feature should be excluded from perturbation.
    """
    n = name.lower()
    if n in SKIP_PERTURBATION_EXACT:
        return True
    return any(n.endswith(s) for s in SKIP_PERTURBATION_SUFFIXES)

def get_physical_bounds(name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (lo, hi) physical bounds for a feature name.
    """
    key = name.lower()
    if key in PHYSICAL_BOUNDS:
        return PHYSICAL_BOUNDS[key]
    if "days_since_install" in key:
        return (0.0, None)
    return (None, None)

def inverse_transform_feature(feature_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Invert MinMax scaling for a feature array.
    """
    if scaler is None:
        return feature_scaled.copy()
    flat = feature_scaled.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(feature_scaled.shape)

def transform_feature(feature_real: np.ndarray, scaler) -> np.ndarray:
    """
    Apply MinMax scaling to a real-valued feature array and clip to [0, 1].
    """
    if scaler is None:
        return np.clip(feature_real, 0.0, 1.0)
    flat = feature_real.reshape(-1, 1)
    return np.clip(scaler.transform(flat).reshape(feature_real.shape), 0.0, 1.0)

def apply_real_space_perturbation(
    X_scaled: np.ndarray,
    feature_idx: int,
    feature_name: str,
    multiplicative_factor: float,
    feature_scaler,
    additive_shift_days: Optional[float] = None,
) -> np.ndarray:
    """
    Perturb a single feature in real (unscaled) space, then re-scale back.

    For age features (days_since_install_), an additive shift in days is applied instead of a multiplicative factor, 
    so that the perturbation reflects a realistic change in module age rather than a percentage change.

    Physical bounds are enforced after perturbation to keep values realistic.
    """
    X_out = X_scaled.copy()

    # Invert scaling to get real-space values
    real = inverse_transform_feature(X_scaled[:, :, feature_idx], feature_scaler)

    is_age_feature = "days_since_install" in feature_name.lower()
    if is_age_feature and additive_shift_days is not None:
        real = real + additive_shift_days
    else:
        real = real * multiplicative_factor

    # Clamp to physical bounds
    lo, hi = get_physical_bounds(feature_name)
    if lo is not None:
        real = np.maximum(real, lo)
    if hi is not None:
        real = np.minimum(real, hi)

    # Re-apply scaling
    X_out[:, :, feature_idx] = transform_feature(real, feature_scaler)
    return X_out

def extract_X_array(dataset) -> np.ndarray:
    """
    Stack all input sequences from a dataset into a single numpy array.
    Shape: (n_samples, window+horizon, n_features)
    """
    return np.stack([dataset[i][0].numpy() for i in range(len(dataset))])

def run_inference(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run batched inference on a numpy input array.
    Returns predictions of shape (n_samples, horizon, n_outputs).
    """
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[s: s + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)

def denormalise(preds_scaled: np.ndarray, scaler) -> np.ndarray:
    """Invert target scaling; returns predictions in original power units (W)."""
    flat = preds_scaled.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(preds_scaled.shape)

def collect_arrays(
    model: nn.Module,
    X: np.ndarray,
    dataset,
    n_targets: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and collect predictions, ground-truth targets, and bad-day masks into aligned numpy arrays.

    Returns:
        preds_raw : (n, horizon, n_targets)   normalised predictions
        trues     : (n, horizon, n_targets)   normalised ground truth
        bad_days  : (n, horizon, n_targets)   bad-day mask (1 = excluded)
    """
    preds_raw = run_inference(model, X, device)
    n, H      = len(dataset), preds_raw.shape[1]
    trues     = np.zeros((n, H, n_targets), dtype=np.float32)
    bad_days  = np.zeros((n, H, n_targets), dtype=np.float32)

    for i in range(n):
        sample = dataset[i]
        if len(sample) == 3:
            _, y, bd = sample
            bad_days[i] = bd.numpy()
        else:
            _, y = sample
        trues[i] = y.numpy()

    return preds_raw, trues, bad_days

def build_continuous(
    preds_raw: np.ndarray,
    trues_raw: np.ndarray,
    bad_days: np.ndarray,
    dataset,
    target_scalers: Dict,
    target_cols: List[str],
    expected_freq: str = "10min",
) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    Reconstruct continuous denormalised time series from non-overlapping predictions.
    """
    _, res = build_continuous_series_multi_output(
        dataset       = dataset,
        preds         = preds_raw,
        trues         = trues_raw,
        bad_days      = bad_days,
        scaler_dict   = target_scalers,
        target_cols   = target_cols,
        mask_bad_days = True,
        expected_freq = expected_freq,
    )
    return {t: (res[t][0], res[t][1]) for t in target_cols}

def feature_perturbation(
    model: nn.Module,
    X_original: np.ndarray,
    target_idx: int,
    target_scaler,
    feature_idx: int,
    feature_name: str,
    feature_scaler,
    delta_pct: float = 10.0,
    additive_shift_days: float = 30.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Compute sensitivity scores for a single feature-target pair by perturbing the feature up and down and measuring the relative change in daytime power.
    For standard features: perturbs by ±delta_pct% in real space. 
    For age features (days_since_install_*): shifts by ±additive_shift_days days.

    Sensitivity Score (SS) = mean(ΔP_daytime) / mean(|P_baseline_daytime|)
    Only daytime sequences (baseline mean power > 1 W) are included.

    Returns dict with SS_up, SS_down, and baseline_power_mean_W.
    """
    # Baseline predictions in real space (W)
    baseline = run_inference(model, X_original, device)
    bp = denormalise(baseline[:, :, target_idx], target_scaler).mean(axis=1)
    day_mask = bp > 1.0  # filter nighttime / near-zero sequences
 
    is_age_feature = "days_since_install" in feature_name.lower()
    results: Dict[str, float] = {}

    for direction, factor, shift in [
        ("up",   1.0 + delta_pct / 100.0,  +additive_shift_days),
        ("down", 1.0 - delta_pct / 100.0,  -additive_shift_days),
    ]:
        Xp = apply_real_space_perturbation(
            X_original, feature_idx, feature_name,
            multiplicative_factor = factor if not is_age_feature else 1.0,
            feature_scaler = feature_scaler,
            additive_shift_days = shift if is_age_feature else None,
        )
        pp = denormalise(run_inference(model, Xp, device)[:, :, target_idx], target_scaler).mean(axis=1)
        dp = pp - bp

        ss = (
            float(dp[day_mask].mean()) / (float(np.abs(bp[day_mask]).mean()) + 1e-12)
            if day_mask.any() else 0.0
        )
        results[f"SS_{direction}"] = ss
        if is_age_feature:
            results[f"shift_days_{direction}"] = shift

    results["baseline_power_mean_W"] = float(bp[day_mask].mean()) if day_mask.any() else 0.0
    return results

def _build_ratio_df(
    dataset, preds_raw, trues_raw, bad_days, df_raw,
    target_name, target_scalers, target_cols,
    feature_scalers, panel_area_m2, irr_min_w_m2, min_power_w=1.0,
    expected_freq="10min",
) -> pd.DataFrame:
    """
    Build a df of (true_power, pred_power, irr, true_ratio, pred_ratio, date) aligned on the continuous time index.
    PCE ratio = Power / (Irradiance * Panel_area_m^2)
    """
    IRR_COL = "Irr"
    _, res = build_continuous_series_multi_output(
        dataset=dataset, preds=preds_raw, trues=trues_raw,
        bad_days=bad_days, scaler_dict=target_scalers,
        target_cols=target_cols, mask_bad_days=True, expected_freq=expected_freq,
    )
    true_s, pred_s, _ = res[target_name]
    idx = true_s.index

    irr_sc   = feature_scalers.get(IRR_COL)
    irr_raw  = df_raw[IRR_COL].reindex(idx)
    irr_vals = (
        irr_sc.inverse_transform(irr_raw.values.reshape(-1, 1)).flatten()
        if irr_sc is not None else irr_raw.values.copy()
    )

    df = pd.DataFrame({
        "true_power": true_s,
        "pred_power": pred_s,
        "irr":        pd.Series(irr_vals, index=idx),
    }).dropna()

    df = df[df["irr"] > irr_min_w_m2]
    df = df[(df["true_power"] >= min_power_w) & (df["pred_power"] >= min_power_w)]

    df["true_ratio"] = df["true_power"] / (df["irr"] * panel_area_m2)
    df["pred_ratio"] = df["pred_power"] / (df["irr"] * panel_area_m2)
    df["date"]       = df.index.date
    return df

def _daily_median(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10 min ratio rows to one median value per day."""
    return (
        df.groupby("date")
        .agg(
            true_ratio=("true_ratio", "median"),
            pred_ratio=("pred_ratio", "median"),
        )
        .reset_index()
        .sort_values("date")
    )

def _ols(idx: np.ndarray, vals: np.ndarray):
    """
    Ordinary least-squares linear regression.
    Returns (slope, intercept, fitted_values).
    """
    s, i, *_ = stats.linregress(idx, vals)
    return s, i, s * idx + i

def _slope_metrics(
    true_slope: float,
    pred_slope: float,
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
) -> Dict:
    """
    Compute degradation slope comparison metrics.
    slope_delta_abs: absolute difference between true and predicted slopes (ratio/day)
    slope_delta_pct: relative difference as a percentage of the true slope
    r2: R² of predicted vs true daily values
    """
    delta_abs = abs(true_slope - pred_slope)
    delta_pct = (
        abs(true_slope - pred_slope) / abs(true_slope) * 100.0
        if abs(true_slope) > 1e-12 else np.nan
    )

    if len(true_vals) > 1 and len(pred_vals) > 1:
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    return {"slope_delta_abs": delta_abs, "slope_delta_pct": delta_pct, "r2": r2}

def _metrics(true: np.ndarray, pred: np.ndarray) -> Dict:
    """
    Compute MAE, MAPE, Pearson correlation, and R² between two arrays.
    MAPE is computed only over elements where |true| > 1e-6 to avoid division by zero.
    """
    mae  = float(np.mean(np.abs(pred - true)))
    vm   = np.abs(true) > 1e-6
    mape = float(np.mean(np.abs((pred[vm] - true[vm]) / true[vm])) * 100.0) if vm.any() else np.nan
    corr = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else np.nan
    if len(true) > 1:
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    return {"mae": mae, "mape": mape, "correlation": corr, "r2": r2}

def _quantile_filter(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    q_low: float = 0.25,
    q_high: float = 0.95,
) -> np.ndarray:
    """
    Return a boolean mask keeping rows where both true and pred fall within [q_low, q_high] quantile range of their respective distributions.
    """
    t_lo = np.nanpercentile(true_vals, q_low * 100)
    t_hi = np.nanpercentile(true_vals, q_high * 100)
    p_lo = np.nanpercentile(pred_vals, q_low * 100)
    p_hi = np.nanpercentile(pred_vals, q_high * 100)
    return (
        (true_vals >= t_lo) & (true_vals <= t_hi) &
        (pred_vals >= p_lo) & (pred_vals <= p_hi)
    )

def compute_p_irr_ratio(
    dataset_test, preds_test, trues_test, bad_days_test, df_test,
    dataset_train, preds_train, trues_train, bad_days_train, df_train,
    target_name, target_scalers, target_cols, feature_scalers,
    panel_area_m2=1.0, irr_min_w_m2=100.0, min_power_w=1.0,
    expected_freq="10min",
    quantile_filtering: bool = True,
    q_low: float = 0.25,
    q_high: float = 0.95,
) -> Dict:
    """
    Compute PCE degradation trends for test and merged (train+test) windows.

    For each window:
    1. Build daily median PCE ratios (P / (Irr * Area))
    2. Optionally apply quantile filtering to remove outlier days
    3. Fit OLS linear trends to true and predicted ratios
    4. Compute slope comparison and regression metrics

    Returns a result dict with slopes, intercepts, fitted values, and metrics for both the test-only and merged train+test windows.
    """
    kwargs = dict(
        target_name=target_name, target_scalers=target_scalers,
        target_cols=target_cols, feature_scalers=feature_scalers,
        panel_area_m2=panel_area_m2, irr_min_w_m2=irr_min_w_m2,
        min_power_w=min_power_w, expected_freq=expected_freq,
    )

    # Test
    df_t = _build_ratio_df(
        dataset_test, preds_test, trues_test, bad_days_test, df_test, **kwargs
    )
    if df_t.empty:
        raise ValueError(
            f"No valid rows for {target_name} after "
            f"Irr>{irr_min_w_m2} and Power>={min_power_w}W filters."
        )

    dt = _daily_median(df_t)
    valid_mask  = (dt["true_ratio"] > 0) & (dt["pred_ratio"] > 0)
    dt_filtered = dt[valid_mask].copy()

    if quantile_filtering and len(dt_filtered) > 1:
        med_mask = _quantile_filter(dt_filtered["true_ratio"].values, dt_filtered["pred_ratio"].values, q_low=q_low, q_high=q_high)
        dt_filtered = dt_filtered[med_mask].copy()
        print(f"    Test daily median: {len(dt)} → {len(dt_filtered)} days after quantile filter")

    t_true = dt_filtered["true_ratio"].values
    t_pred = dt_filtered["pred_ratio"].values
    t_idx  = np.arange(len(dt_filtered), dtype=float)

    if len(t_true) > 1:
        ts, ti, tf = _ols(t_idx, t_true)
        ps, pi, pf = _ols(t_idx, t_pred)
        ts_pct = ts * 100.0
        ps_pct = ps * 100.0
        t_slope_metrics = _slope_metrics(ts, ps, t_true, t_pred)
    else:
        ts = ti = tf = ps = pi = pf = ts_pct = ps_pct = np.nan
        t_slope_metrics = {"slope_delta_abs": np.nan, "slope_delta_pct": np.nan, "r2": np.nan}

    result = {
        "test_dates": dt_filtered["date"].astype(str).tolist(),
        "test_day_indices": t_idx,
        "test_true_ratio": t_true,
        "test_pred_ratio": t_pred,
        "test_true_slope": ts,
        "test_pred_slope": ps,
        "test_true_slope_pct": ts_pct,
        "test_pred_slope_pct": ps_pct,
        "test_true_intercept": ti,
        "test_pred_intercept": pi,
        "test_true_fit": tf,
        "test_pred_fit": pf,
        **{f"test_{k}": v for k, v in t_slope_metrics.items()},
        **{f"test_{k}": v for k, v in _metrics(t_true, t_pred).items()},
        "target_name": target_name,
        "panel_area_m2": panel_area_m2,
        "irr_min_w_m2": irr_min_w_m2,
        "min_power_w": min_power_w,
    }

    # Merged (train + test)
    df_tr = _build_ratio_df(dataset_train, preds_train, trues_train, bad_days_train, df_train, **kwargs)
    dtr = _daily_median(df_tr)
    train_end = str(dtr["date"].max())

    dall = (
        pd.concat([dtr, dt_filtered], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    merged_valid  = (dall["true_ratio"] > 0) & (dall["pred_ratio"] > 0)
    dall_filtered = dall[merged_valid].copy()

    if quantile_filtering and len(dall_filtered) > 1:
        merged_mask = _quantile_filter(dall_filtered["true_ratio"].values, dall_filtered["pred_ratio"].values, q_low=q_low, q_high=q_high)
        dall_filtered = dall_filtered[merged_mask].copy()
        print(f"    Merged daily median: {len(dall)} → {len(dall_filtered)} days after quantile filter")

    m_true = dall_filtered["true_ratio"].values
    m_pred = dall_filtered["pred_ratio"].values
    m_idx  = np.arange(len(dall_filtered), dtype=float)

    if len(m_true) > 1:
        mts, mti, mtf = _ols(m_idx, m_true)
        mps, mpi, mpf = _ols(m_idx, m_pred)
        mts_pct = mts * 100.0
        mps_pct = mps * 100.0
        m_slope_metrics = _slope_metrics(mts, mps, m_true, m_pred)
    else:
        mts = mti = mtf = mps = mpi = mpf = mts_pct = mps_pct = np.nan
        m_slope_metrics = {"slope_delta_abs": np.nan, "slope_delta_pct": np.nan, "r2": np.nan}

    result.update({
        "merged_dates": dall_filtered["date"].astype(str).tolist(),
        "merged_day_indices": m_idx,
        "merged_true_ratio": m_true,
        "merged_pred_ratio": m_pred,
        "merged_true_slope": mts,
        "merged_pred_slope": mps,
        "merged_true_slope_pct": mts_pct,
        "merged_pred_slope_pct": mps_pct,
        "merged_true_fit": mtf,
        "merged_pred_fit": mpf,
        **{f"merged_{k}": v for k, v in m_slope_metrics.items()},
        **{f"merged_{k}": v for k, v in _metrics(m_true, m_pred).items()},
        "train_end_date":train_end,
        "quantile_filtering_applied": quantile_filtering,
        "q_low": q_low,
        "q_high": q_high,
    })
    return result

def _format_yaxis(ax, y_min: float = 0.0, y_max: float = None, n_ticks: int = 6, fontsize: int = 14):
    if y_max is None:
        y_max = 1.0
    padding = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 1.0
    y_max_plot = y_max + padding
    ticks = np.linspace(y_min, y_max_plot, n_ticks)
    ax.set_ylim(y_min, y_max_plot)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" for v in ticks], fontsize=fontsize)

def _style_ax(ax, title, xlabel, ylabel, fontsize=14):
    ax.set_title(title, fontsize=24, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5, zorder=2)
    ax.grid(True, alpha=0.1, axis="x", linestyle="--", linewidth=0.5, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _make_ticks(idx: np.ndarray, dates, fig_width: float = 14, n_max: int = 10):
    n = len(idx)
    if n <= 5:
        return list(idx), list(dates)
    n_ticks = min(n_max, max(5, int(fig_width / 2)))
    ti = np.unique(np.concatenate([[0, n - 1], np.linspace(0, n - 1, n_ticks, dtype=int)]))
    return [idx[i] for i in ti], [dates[i] for i in ti]

def _fit_and_plot_ols(
    ax, idx, true_pct, pred_pct, dates,
    fig_width=18, fontsize=14, y_min=0.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Scatter true and predicted PCE values (in %) and overlay OLS linear trend lines.
    """
    ax.scatter(idx, true_pct, color="steelblue", alpha=0.85, s=30, zorder=3,
               label="True", edgecolor="white", linewidth=0.4)
    ax.scatter(idx, pred_pct, color="crimson",   alpha=0.85, s=30, zorder=3,
               label="Predicted", marker="^", edgecolor="white", linewidth=0.4)

    ts_pct = ps_pct = np.nan
    tf = pf = np.full_like(idx, np.nan)
    if len(idx) > 1:
        ts_pct, _, tf = _ols(idx, true_pct)
        ps_pct, _, pf = _ols(idx, pred_pct)
        ax.plot(idx, tf, color="steelblue", lw=2, ls="--", zorder=4,
                label=f"True: {ts_pct:+.8f}%/day | {ts_pct * 365:+.2f}%/year")
        ax.plot(idx, pf, color="crimson",   lw=2, ls="--", zorder=4,
                label=f"Pred: {ps_pct:+.8f}%/day | {ps_pct * 365:+.2f}%/year")

    tick_pos, tick_labels = _make_ticks(idx, dates, fig_width=fig_width)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=fontsize)

    all_values = np.concatenate([true_pct, pred_pct])
    if len(idx) > 1:
        all_values = np.concatenate([all_values, tf, pf])
    y_max = np.nanmax(all_values)
    _format_yaxis(ax, y_min=y_min, y_max=y_max, n_ticks=6, fontsize=fontsize)

    ax.legend(fontsize=fontsize, loc="lower left",
              frameon=True, framealpha=0.95, edgecolor="black", facecolor="white")
    return ts_pct, ps_pct, tf, pf

def _fit_and_plot_mean_sd(
    ax, idx, mean_true, mean_pred, sd_true, sd_pred,
    dates, fig_width=18, fontsize=14, y_min=0.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Plot daily mean PCE ± 1 standard deviation bands with OLS linear trend lines.
    """
    ax.fill_between(idx, mean_true - sd_true, mean_true + sd_true,
                    color="steelblue", alpha=0.18, zorder=1, label="True ±1 SD")
    ax.fill_between(idx, mean_pred - sd_pred, mean_pred + sd_pred,
                    color="crimson", alpha=0.18, zorder=1, label="Pred ±1 SD")

    ax.scatter(idx, mean_true, color="steelblue", alpha=0.9, s=35, zorder=3,
               label="True mean", edgecolor="white", linewidth=0.4)
    ax.scatter(idx, mean_pred, color="crimson",   alpha=0.9, s=35, zorder=3,
               label="Pred mean", marker="^", edgecolor="white", linewidth=0.4)

    ts_pct = ps_pct = np.nan
    tf = pf = np.full_like(idx, np.nan)
    if len(idx) > 1:
        ts_pct, _, tf = _ols(idx, mean_true)
        ps_pct, _, pf = _ols(idx, mean_pred)
        ax.plot(idx, tf, color="steelblue", lw=2.5, ls="--", zorder=4,
                label=f"True: {ts_pct:+.8f}%/day | {ts_pct * 365:+.2f}%/year")
        ax.plot(idx, pf, color="crimson",   lw=2.5, ls="--", zorder=4,
                label=f"Pred: {ps_pct:+.8f}%/day | {ps_pct * 365:+.2f}%/year")

    tick_pos, tick_labels = _make_ticks(idx, dates, fig_width=fig_width)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=fontsize)

    all_values = np.concatenate([
        mean_true - sd_true, mean_true + sd_true,
        mean_pred - sd_pred, mean_pred + sd_pred,
    ])
    if len(idx) > 1:
        all_values = np.concatenate([all_values, tf, pf])
    y_max = np.nanmax(all_values)
    _format_yaxis(ax, y_min=y_min, y_max=y_max, n_ticks=6, fontsize=fontsize)

    ax.legend(fontsize=fontsize - 1, loc="lower left",
              frameon=True, framealpha=0.95, edgecolor="black",
              facecolor="white", ncol=2)
    return ts_pct, ps_pct, tf, pf

def plot_p_irr(
    ratio_result: Dict,
    out_dir: Path,
    target_name: str,
    df_train_10min: pd.DataFrame,
    df_test_10min: pd.DataFrame,
    quantile_filtering: bool = True,
    q_low: float = 0.25,
    q_high: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Generate and save four PCE degradation trend plots for a single target module.
    Each plot fits and displays linear OLS trend line:
      1. Merged raw 10-min PCE values
      2. Merged daily mean PCE ± 1 SD
      3. Merged daily median PCE
      4. Merged daily 90th-percentile PCE
    Quantile filtering is applied before fitting when enabled to remove outlier days.
    """
    plot_slopes: Dict[str, Dict[str, float]] = {}

    # Build and filter merged raw 10-min frame
    df_merged_raw = (
        pd.concat([df_train_10min, df_test_10min], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_merged_raw = df_merged_raw[
        (df_merged_raw["true_ratio"] > 0) &
        (df_merged_raw["pred_ratio"] > 0)
    ].copy()

    if quantile_filtering:
        raw_mask = _quantile_filter(
            df_merged_raw["true_ratio"].values, df_merged_raw["pred_ratio"].values,
            q_low=q_low, q_high=q_high,
        )
        df_raw_filt = df_merged_raw[raw_mask].copy()
        print(f"  Raw 10-min merged: {len(df_merged_raw):,} → {len(df_raw_filt):,} "
              f"after [{q_low:.1%}-{q_high:.1%}] quantile filter")
    else:
        df_raw_filt = df_merged_raw.copy()
        print(f"  Raw 10-min merged: {len(df_merged_raw):,} rows (no quantile filtering)")

    #1. Merged raw 10 min
    if len(df_raw_filt) > 1:
        raw_idx = np.arange(len(df_raw_filt), dtype=float)
        raw_dates = df_raw_filt["date"].astype(str).tolist()
        true_raw = df_raw_filt["true_ratio"].values * 100
        pred_raw = df_raw_filt["pred_ratio"].values * 100

        fig, ax = plt.subplots(figsize=(20, 6))
        ts_pct, ps_pct, _, _ = _fit_and_plot_ols(ax, raw_idx, true_raw, pred_raw, raw_dates, fig_width=20, y_min=0.0)
        _style_ax(ax, f"Raw PCE Values (10-min, Merged) | {target_name}", "Date", "PCE (%)")
        _save(fig, out_dir / f"p_irr_{target_name}_merged_raw.png", dpi=300)

        m_raw = _metrics(df_raw_filt["true_ratio"].values, df_raw_filt["pred_ratio"].values)
        plot_slopes["merged_raw"] = dict(
            true_slope_pct_per_day = ts_pct,
            pred_slope_pct_per_day = ps_pct,
            true_slope_pct_per_year = ts_pct * 365 if not np.isnan(ts_pct) else np.nan,
            pred_slope_pct_per_year = ps_pct * 365 if not np.isnan(ps_pct) else np.nan,
            n=len(df_raw_filt), **m_raw,
        )
        print(f"  Raw: True={ts_pct:+.6f}%/day  Pred={ps_pct:+.6f}%/day")

    # ── Daily aggregates ──────────────────────────────────────────────────────
    daily = (
        df_raw_filt.groupby("date")
        .agg(
            true_mean =("true_ratio", "mean"),
            pred_mean =("pred_ratio", "mean"),
            true_std =("true_ratio", "std"),
            pred_std =("pred_ratio", "std"),
            true_median =("true_ratio", "median"),
            pred_median =("pred_ratio", "median"),
            true_p90 =("true_ratio", lambda x: np.percentile(x, 90)),
            pred_p90 =("pred_ratio", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["true_std"] = daily["true_std"].fillna(0.0)
    daily["pred_std"] = daily["pred_std"].fillna(0.0)
    daily["date_str"] = daily["date"].astype(str)

    #2. Merged daily mean +/- 1 SD 
    if len(daily) >= 2:
        if quantile_filtering:
            mean_mask = _quantile_filter(daily["true_mean"].values, daily["pred_mean"].values, q_low=q_low, q_high=q_high,)
            d_m = daily[mean_mask].copy()
            print(f"  Daily Mean: {len(daily)} → {mean_mask.sum()} days "
                  f"after [{q_low:.1%}-{q_high:.1%}] quantile filter")
        else:
            d_m = daily.copy()
            print(f"  Daily Mean: {len(daily)} days (no quantile filtering)")

        if len(d_m) >= 2:
            idx_m = np.arange(len(d_m), dtype=float)
            mt_pct = d_m["true_mean"].values * 100
            mp_pct = d_m["pred_mean"].values * 100
            st_pct = d_m["true_std"].values  * 100
            sp_pct = d_m["pred_std"].values  * 100

            fig, ax = plt.subplots(figsize=(18, 6))
            ts_pct, ps_pct, _, _ = _fit_and_plot_mean_sd(ax, idx_m, mt_pct, mp_pct, st_pct, sp_pct, d_m["date_str"].tolist(), fig_width=18, y_min=0.0)
            _style_ax(ax, f"Daily Mean PCE ± 1 SD (Merged) | {target_name}", "Date", "Daily Mean PCE (%)")
            _save(fig, out_dir / f"p_irr_{target_name}_merged_mean.png", dpi=300)

            m_mean = _metrics(d_m["true_mean"].values, d_m["pred_mean"].values)
            plot_slopes["merged_mean"] = dict(
                true_slope_pct_per_day = ts_pct,
                pred_slope_pct_per_day = ps_pct,
                true_slope_pct_per_year = ts_pct * 365 if not np.isnan(ts_pct) else np.nan,
                pred_slope_pct_per_year = ps_pct * 365 if not np.isnan(ps_pct) else np.nan,
                n=len(d_m), **m_mean,
            )
            print(f"  Daily Mean: True={ts_pct:+.6f}%/day  Pred={ps_pct:+.6f}%/day")

    # 3. Merged daily median 
    if len(daily) >= 2:
        if quantile_filtering:
            med_mask = _quantile_filter(daily["true_median"].values, daily["pred_median"].values, q_low=q_low, q_high=q_high,)
            d_med = daily[med_mask].copy()
            print(f"  Daily Median: {len(daily)} → {med_mask.sum()} days "
                  f"after [{q_low:.1%}-{q_high:.1%}] quantile filter")
        else:
            d_med = daily.copy()
            print(f"  Daily Median: {len(daily)} days (no quantile filtering)")

        if len(d_med) >= 2:
            idx_med      = np.arange(len(d_med), dtype=float)
            true_med_pct = d_med["true_median"].values * 100
            pred_med_pct = d_med["pred_median"].values * 100

            fig, ax = plt.subplots(figsize=(18, 6))
            ts_pct, ps_pct, _, _ = _fit_and_plot_ols(ax, idx_med, true_med_pct, pred_med_pct, d_med["date_str"].tolist(), fig_width=18, y_min=0.0)
            _style_ax(ax, f"Daily PCE Trend | {target_name}", "Date", "Daily Median PCE (%)")
            _save(fig, out_dir / f"p_irr_{target_name}_merged_median.png", dpi=300)

            m_med = _metrics(d_med["true_median"].values, d_med["pred_median"].values)
            plot_slopes["merged_median"] = dict(
                true_slope_pct_per_day = ts_pct,
                pred_slope_pct_per_day = ps_pct,
                true_slope_pct_per_year = ts_pct * 365 if not np.isnan(ts_pct) else np.nan,
                pred_slope_pct_per_year = ps_pct * 365 if not np.isnan(ps_pct) else np.nan,
                n=len(d_med), **m_med,
            )
            print(f"  Daily Median: True={ts_pct:+.6f}%/day  Pred={ps_pct:+.6f}%/day")

    #4. Merged daily 90th percentile 
    if len(daily) >= 2:
        if quantile_filtering:
            p90_mask = _quantile_filter(daily["true_p90"].values, daily["pred_p90"].values, q_low=q_low, q_high=q_high,)
            d_p90 = daily[p90_mask].copy()
            print(f"  Daily P90: {len(daily)} → {p90_mask.sum()} days "
                  f"after [{q_low:.1%}-{q_high:.1%}] quantile filter")
        else:
            d_p90 = daily.copy()
            print(f"  Daily P90: {len(daily)} days (no quantile filtering)")

        if len(d_p90) >= 2:
            idx_p90      = np.arange(len(d_p90), dtype=float)
            true_p90_pct = d_p90["true_p90"].values * 100
            pred_p90_pct = d_p90["pred_p90"].values * 100

            fig, ax = plt.subplots(figsize=(18, 6))
            ts_pct, ps_pct, _, _ = _fit_and_plot_ols(ax, idx_p90, true_p90_pct, pred_p90_pct, d_p90["date_str"].tolist(), fig_width=18, y_min=0.0)
            _style_ax(ax, f"Daily 90th Percentile PCE Trend (Merged) | {target_name}", "Date", "Daily P90 PCE (%)")
            _save(fig, out_dir / f"p_irr_{target_name}_merged_p90.png", dpi=300)

            m_p90 = _metrics(d_p90["true_p90"].values, d_p90["pred_p90"].values)
            plot_slopes["merged_p90"] = dict(
                true_slope_pct_per_day = ts_pct,
                pred_slope_pct_per_day = ps_pct,
                true_slope_pct_per_year = ts_pct * 365 if not np.isnan(ts_pct) else np.nan,
                pred_slope_pct_per_year = ps_pct * 365 if not np.isnan(ps_pct) else np.nan,
                n=len(d_p90), **m_p90,
            )
            print(f"  Daily P90:  True={ts_pct:+.6f}%/day  Pred={ps_pct:+.6f}%/day")

    return plot_slopes

def write_report(
    fp_test: Dict,
    fp_merged: Dict,
    pirr: Dict,
    feat_names: List[str],
    target_names: List[str],
    delta_pct: float,
    out_path: Path,
):
    """
    Full sensitivity analysis report to a text file.
    """
    sep  = "=" * 78
    sep2 = "-" * 50

    with open(out_path, "w") as f:
        f.write(f"{sep}\nSENSITIVITY ANALYSIS REPORT\n{sep}\n\n")

        #Feature perturbation 
        for label, fp in [("TEST ONLY", fp_test), ("MERGED TRAIN+TEST", fp_merged)]:
            f.write(
                f"PART 1 FEATURE PERTURBATION SENSITIVITY  [{label}]  "
                f"(±{delta_pct:.0f}%)\n{sep2}\n"
            )
            f.write("SS = mean(ΔP_daytime) / mean(|P_baseline_daytime|)\n\n")

            for target in target_names:
                if target not in fp:
                    continue
                f.write(f"{target}:\n")
                f.write(f"{'Feature':<32s}  {'SS_up':>10s}  {'SS_down':>10s}\n")
                f.write(f"{'-'*32}  {'-'*10}  {'-'*10}\n")

                items = [
                    (feat, fp[target][feat])
                    for feat in feat_names
                    if feat in fp[target]
                ]
                # Sort by maximum absolute sensitivity score
                items.sort(
                    key=lambda x: max(abs(x[1].get("SS_up", 0)), abs(x[1].get("SS_down", 0))),
                    reverse=True,
                )
                for feat, res in items:
                    f.write(
                        f"    {feat:<32s}  "
                        f"{res.get('SS_up',   float('nan')):>+10.5f}  "
                        f"{res.get('SS_down', float('nan')):>+10.5f}\n"
                    )
                f.write("\n")

        #PCE ratio 
        f.write(f"\nPART 2 DAILY PCE RATIO (P / (Irr * Area))\n{sep2}\n")
        f.write("Slopes are fitted on the exact filtered dataset used in each plot.\n")
        f.write("All Δslope values are derived from those same plot slopes.\n")
        f.write(
            "Δslope% is reported as N/A when |true slope| < 1e-6 %/day "
            "(near-zero denominator).\n\n"
        )

        for target in target_names:
            if target not in pirr:
                continue
            r      = pirr[target]
            slopes = r.get("plot_slopes", {})

            f.write(f"{target}:\n")
            f.write(f"Panel area: {r.get('panel_area_m2', 'N/A')} m²\n")
            f.write(f"Min Irradiance: {r.get('irr_min_w_m2', 'N/A')} W/m²\n")
            f.write(f"Min Power: {r.get('min_power_w', 'N/A')} W\n")

            quantile_status = "Applied" if r.get("quantile_filtering_applied", True) else "Not applied"
            f.write(f"    Quantile filtering  : {quantile_status}\n")
            if quantile_status == "Applied":
                f.write(
                    f"q_low: {r.get('q_low', 0.25):.4f},  "
                    f"q_high: {r.get('q_high', 0.95):.4f}\n"
                )
            f.write(f"Train end date: {r.get('train_end_date', 'N/A')}\n\n")

            plot_label_map = {
                "merged_raw": "Merged raw 10-min",
                "merged_mean": "Merged daily mean",
                "merged_median": "Merged daily median",
                "merged_p90": "Merged daily P90",
            }

            col_w = 24
            f.write(
                f"{'Plot':<{col_w}}  "
                f"{'N':>6s}  "
                f"{'True(%/day)':>13s} {'True(%/yr)':>11s} "
                f"{'Pred(%/day)':>13s} {'Pred(%/yr)':>11s} "
                f"{'|Δ|(%/day)':>12s} {'|Δ|(%/yr)':>11s} {'Δ%':>7s} "
                f"{'MAE':>9s}  {'MAPE':>7s}  {'R²':>7s} {'Corr':>7s}\n"
            )
            f.write(
                f"    {'-'*col_w}  {'-'*6}  {'-'*13}  {'-'*11}  "
                f"{'-'*13}  {'-'*11}  {'-'*12}  {'-'*11}  {'-'*7}  "
                f"{'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}\n"
            )

            for key, plabel in plot_label_map.items():
                s = slopes.get(key, {})
                td = s.get("true_slope_pct_per_day", np.nan)
                ty = s.get("true_slope_pct_per_year", np.nan)
                pd_ = s.get("pred_slope_pct_per_day", np.nan)
                py = s.get("pred_slope_pct_per_year", np.nan)
                n  = s.get("n", np.nan)
                mae = s.get("mae", np.nan)
                mape = s.get("mape", np.nan)
                r2 = s.get("r2", np.nan)
                corr = s.get("correlation", np.nan)

                if not (np.isnan(td) or np.isnan(pd_)):
                    d_day = abs(td - pd_)
                    d_yr  = abs(ty - py) if not (np.isnan(ty) or np.isnan(py)) else np.nan
                    d_pct = d_day / abs(td) * 100.0 if abs(td) >= 1e-6 else np.nan
                else:
                    d_day = d_yr = d_pct = np.nan

                def _fv(v, fmt="+.8f"):
                    return ("N/A" 
                            if (v is None or (isinstance(v, float) and np.isnan(v))) 
                            else format(v, fmt)
                            )

                n_str = "N/A" if (isinstance(n, float) and np.isnan(n)) else str(int(n))
                mape_str = (_fv(mape, ".2f") + "%" if not (isinstance(mape, float) and np.isnan(mape)) else "N/A")
                r2_str = (_fv(r2,   ".4f") if not (isinstance(r2,   float) and np.isnan(r2))   else "N/A")
                corr_str = (_fv(corr, ".4f") if not (isinstance(corr, float) and np.isnan(corr)) else "N/A")
                mae_str = (_fv(mae,  ".6f") if not (isinstance(mae,  float) and np.isnan(mae))  else "N/A")
                dpct_str = (_fv(d_pct, ".2f") + "%" if not (isinstance(d_pct, float) and np.isnan(d_pct)) else "N/A")

                f.write(
                    f"    {plabel:<{col_w}}  "
                    f"{n_str:>6s}  "
                    f"{_fv(td):>13s}  {_fv(ty, '+.4f'):>11s}  "
                    f"{_fv(pd_):>13s}  {_fv(py, '+.4f'):>11s}  "
                    f"{_fv(d_day):>12s}  {_fv(d_yr, '+.4f'):>11s}  {dpct_str:>7s}  "
                    f"{mae_str:>9s}  {mape_str:>7s}  {r2_str:>7s}  {corr_str:>7s}\n"
                )
            f.write("\n")

        f.write(f"\nGenerated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")

    print(f"saved: {out_path.name}")

class LSTMSensitivityAnalyser:
    def __init__(
        self,
        model_path: str,
        model_info_path: str,
        train_data_path: str,
        test_data_path: str,
        target_scaler_path: str,
        feature_scaler_path: str,
        output_dir: str = None,
        delta_pct: float = 10.0,
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.train_data_path = Path(train_data_path)
        self.test_data_path = Path(test_data_path)
        self.target_scaler_path = Path(target_scaler_path)
        self.feature_scaler_path = Path(feature_scaler_path)
        self.delta_pct = delta_pct

        self.output_dir = (Path(output_dir) 
                           if output_dir
                           else self.model_path.parent / "sensitivity_analysis"
                           )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._load()

        self._test_preds = self._test_trues  = self._test_bad  = None
        self._train_preds = self._train_trues = self._train_bad = None
        self._test_baseline: Dict[str, pd.Series] = {}
        self._train_baseline: Dict[str, pd.Series] = {}

    def _load(self):
        with open(self.model_info_path, "rb") as f:
            info = pickle.load(f)

        self.target_cols = info["data_config"]["features"]["target_cols"]
        all_feat = info["data_config"]["features"]["feature_cols"]
        self.feature_cols  = [c for c in all_feat if c not in self.target_cols]
        self.window = info["data_config"]["window_horizon"]["window"]
        self.horizon = info["data_config"]["window_horizon"]["horizon"]
        use_bad = info["data_config"]["features"]["use_bad_day"]
        mask_bad = info["data_config"]["features"]["mask_bad_days"]
        self.bad_day_mapping = info.get("bad_day_mapping", {})
        arch = info["architecture"]

        print(f"Targets  : {self.target_cols}")
        print(f"Features : {len(self.feature_cols)}")
        print(f"Window   : {self.window} Horizon : {self.horizon}")

        self.train_df = pd.read_parquet(self.train_data_path)
        self.test_df  = pd.read_parquet(self.test_data_path)
        print(f"Train rows: {len(self.train_df):,}   Test rows: {len(self.test_df):,}")

        with open(self.target_scaler_path, "rb") as f:
            sd = pickle.load(f)
        self.target_scalers = {t: sd.get(t, sd.get("scaler")) for t in self.target_cols}

        with open(self.feature_scaler_path, "rb") as f:
            fsd = pickle.load(f)
        self.feature_scalers = fsd.get("feature_scalers", {})
        print("Scalers loaded")

        self.device = torch.device("cpu")
        self.model  = MultiOutputLSTMModel(
            input_size = len(all_feat),
            hidden_size = arch["hidden_size"],
            num_layers = arch["num_layers"],
            num_outputs = len(self.target_cols),
            dropout = arch["dropout"],
            forecast_steps = self.horizon,
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()
        print("Model loaded")

        ds_kw = dict(
            feature_cols = self.feature_cols,
            target_cols = self.target_cols,
            window = self.window,
            horizon = self.horizon,
            use_bad_day = use_bad,
            mask_bad_days = mask_bad,
            bad_day_mapping = self.bad_day_mapping,
        )
        self.train_ds = NonOverlappingMultiOutputDataset(self.train_df, **ds_kw)
        self.test_ds  = NonOverlappingMultiOutputDataset(self.test_df,  **ds_kw)
        print(f"Dataset: Train: {len(self.train_ds)}  Test: {len(self.test_ds)}\n")

        self.X_test = extract_X_array(self.test_ds)
        self.X_train = extract_X_array(self.train_ds)
        self.perturb_feats = [f for f in self.feature_cols if not is_skipped_feature(f)]
        print(f"Perturb features: {len(self.perturb_feats)} / {len(self.feature_cols)}")

    def _ensure_test(self):
        if self._test_preds is None:
            self._test_preds, self._test_trues, self._test_bad = collect_arrays(
                self.model, self.X_test, self.test_ds, len(self.target_cols), self.device,
            )

    def _ensure_train(self):
        if self._train_preds is None:
            self._train_preds, self._train_trues, self._train_bad = collect_arrays(
                self.model, self.X_train, self.train_ds, len(self.target_cols), self.device,
            )

    def run_feature_perturbation(self) -> Tuple[Dict, Dict]:
        print("=" * 70)
        print(f"FEATURE PERTURBATION SENSITIVITY  (±{self.delta_pct:.0f}%)")
        print("=" * 70)
        self._ensure_test()
        self._ensure_train()
        X_merged = np.concatenate([self.X_train, self.X_test], axis=0)
        fp_test = {t: {} for t in self.target_cols}
        fp_merged = {t: {} for t in self.target_cols}

        for t_idx, target in enumerate(self.target_cols):
            print(f"  Target: {target}")
            for feat in self.perturb_feats:
                f_idx   = self.feature_cols.index(feat)
                fscaler = self.feature_scalers.get(feat)

                rt = feature_perturbation(
                    model=self.model, X_original=self.X_test,
                    target_idx=t_idx, target_scaler=self.target_scalers[target],
                    feature_idx=f_idx, feature_name=feat, feature_scaler=fscaler,
                    delta_pct=self.delta_pct, additive_shift_days=30.0, device=self.device,
                )
                rm = feature_perturbation(
                    model=self.model, X_original=X_merged,
                    target_idx=t_idx, target_scaler=self.target_scalers[target],
                    feature_idx=f_idx, feature_name=feat, feature_scaler=fscaler,
                    delta_pct=self.delta_pct, additive_shift_days=30.0, device=self.device,
                )
                fp_test[target][feat]   = rt
                fp_merged[target][feat] = rm

                print(
                    f"{feat:<32s}  test  {rt['SS_up']:+.5f}/{rt['SS_down']:+.5f}"
                    f"merged  {rm['SS_up']:+.5f}/{rm['SS_down']:+.5f}"
                )
        return fp_test, fp_merged

    def run_p_irr(
        self,
        panel_area_m2: float = 1.0,
        irr_min_w_m2: float = 100.0,
        min_power_w: float = 1.0,
        quantile_filtering: bool = True,
        q_low: float = 0.25,
        q_high: float = 0.95,
    ) -> Dict:
        print("=" * 70)
        print(f"P/Irr RATIO  "
            f"(area={panel_area_m2} m²  Irr>{irr_min_w_m2} W/m²  Power>={min_power_w} W)")
        if quantile_filtering:
            print(f"  quantile filter [{q_low:.1%}-{q_high:.1%}]")
        else:
            print("  quantile filtering: DISABLED")
        print("=" * 70)
        self._ensure_test()
        self._ensure_train()

        pirr = {}
        _build_kw = dict(
            target_scalers = self.target_scalers,
            target_cols = self.target_cols,
            feature_scalers= self.feature_scalers,
            panel_area_m2 = panel_area_m2,
            irr_min_w_m2 = irr_min_w_m2,
            min_power_w = min_power_w,
        )

        for target in self.target_cols:
            print(f"  Target: {target}")
            try:
                res = compute_p_irr_ratio(
                    dataset_test=self.test_ds, preds_test=self._test_preds,
                    trues_test=self._test_trues, bad_days_test=self._test_bad,
                    df_test=self.test_df,
                    dataset_train=self.train_ds, preds_train=self._train_preds,
                    trues_train=self._train_trues, bad_days_train=self._train_bad,
                    df_train=self.train_df,
                    target_name=target,
                    panel_area_m2=panel_area_m2,
                    irr_min_w_m2=irr_min_w_m2,
                    min_power_w=min_power_w,
                    quantile_filtering=quantile_filtering,
                    q_low=q_low,
                    q_high=q_high,
                    **{k: v for k, v in _build_kw.items()
                    if k not in ("panel_area_m2", "irr_min_w_m2", "min_power_w")},
                )

                df_train_10min = _build_ratio_df(
                    self.train_ds, self._train_preds, self._train_trues,
                    self._train_bad, self.train_df, target_name=target, **_build_kw,
                )
                df_test_10min  = _build_ratio_df(
                    self.test_ds,  self._test_preds,  self._test_trues,
                    self._test_bad, self.test_df, target_name=target, **_build_kw,
                )

                plot_slopes = plot_p_irr(
                    ratio_result=res,
                    out_dir=self.plots_dir,
                    target_name=target,
                    df_train_10min=df_train_10min,
                    df_test_10min=df_test_10min,
                    quantile_filtering=quantile_filtering,
                    q_low=q_low,
                    q_high=q_high,
                )
                res["plot_slopes"] = plot_slopes
                pirr[target] = res

                for mode in ("test", "merged"):
                    sd_pct = res.get(f"{mode}_slope_delta_pct", np.nan)
                    print(
                        f"    [{mode}]  "
                        f"true={res[f'{mode}_true_slope']:+.8f}  "
                        f"pred={res[f'{mode}_pred_slope']:+.8f}  "
                        f"|Δ|={res[f'{mode}_slope_delta_abs']:.8f}  "
                        f"({'N/A' if np.isnan(sd_pct) else f'{sd_pct:.2f}%'})  "
                        f"R²={res[f'{mode}_r2']:.4f}"
                    )

            except Exception as exc:
                print(f"WARNING: P/Irr failed for {target}: {exc}")
                traceback.print_exc()

        print()
        return pirr

    def run_all(
        self,
        panel_area_m2: float = 1.0,
        irr_min_w_m2: float = 100.0,
        min_power_w: float = 1.0,
        quantile_filtering: bool = True,
        q_low: float = 0.25,
        q_high: float = 0.95,
        run_feature_pert: bool = True,
        run_p_irr: bool = True,
    ):
        fp_test = fp_merged = {}
        pirr = {}

        if run_feature_pert:
            fp_test, fp_merged = self.run_feature_perturbation()
        else:
            print("Feature Perturbation Sensitivity, SKIPPED")

        if run_p_irr:
            pirr = self.run_p_irr(panel_area_m2, irr_min_w_m2, min_power_w, quantile_filtering=quantile_filtering, q_low=q_low, q_high=q_high)
        else:
            print("P/Irr Ratio, SKIPPED")

        write_report(
            fp_test, fp_merged,
            pirr,
            self.perturb_feats, self.target_cols,
            self.delta_pct,
            self.output_dir / "sensitivity_report.txt",
        )
        print(f"\n{'='*70}")
        print(f"ALL ANALYSES COMPLETE outputs in: {self.output_dir}")
        print(f"{'='*70}")

def main():
    RUN_FEATURE_PERT = True
    RUN_P_IRR        = True

    BASE_PATH = "path/to/lstm_run_folder"

    analyser = LSTMSensitivityAnalyser(
        model_path = f"{BASE_PATH}/lstm_results/<timestamp>_model_multi.pt",
        model_info_path = f"{BASE_PATH}/lstm_results/model_info_multi.pkl",
        train_data_path = f"{BASE_PATH}/training_data/train_scaled.parquet",
        test_data_path = f"{BASE_PATH}/training_data/test_scaled.parquet",
        target_scaler_path = f"{BASE_PATH}/training_data/p_scalers.pkl",
        feature_scaler_path = f"{BASE_PATH}/training_data/scalers.pkl",
        output_dir  = f"{BASE_PATH}/lstm_results/sensitivity_analysis",
        delta_pct = 10.0,
        seed = 42,
    )

    analyser.run_all(
        panel_area_m2=0.7,      # psc: 0.7  |  solon: 1.44  |  sanyo: 1.125  (m^2)
        irr_min_w_m2=300.0,
        min_power_w=0.0,
        quantile_filtering=True,
        q_low=0.001,
        q_high=0.9999,
        run_feature_pert=RUN_FEATURE_PERT,
        run_p_irr=RUN_P_IRR,
    )

if __name__ == "__main__":
    main()