"""
PCE + reversible / irreversible loss decomposition from residual_export.csv.

No model / inference needed: everything is computed from the columns already
saved by residual_LSTM.py / residual_MLPLSTM.py:
    true_<target>, pred_<target>, pred_<target>_age1, Irr, temp_C,
    doy_sin, doy_cos, hour_sin, hour_cos, bad_day_<target>, _split, ...

Pipeline (mirrors the counterfactual-age analyzer script):
  1. Load CSV, per target: drop bad days, keep daylight hours, Irr > 0.
  2. PCE [%] for TRUE, PRED (normal age) and AGE1 (counterfactual age=1 pred):
        PCE = P / (Irr * PANEL_AREA) * 100
     with a shared validity mask (Irr >= 300, P >= 20 W for all three) and a
     joint 99th-percentile outlier filter.
  3. eta_ref (reference efficiency) = age=1 PCE at the same timestamp (direct).
  4. Per-day energy-based, relative-delta loss decomposition for the TRUE and
     PRED branches (both against the same eta_ref):
        total / reversible / irreversible loss [%], performance ratio, ...
  5. Plots: PCE over time, daily loss bars (+ TRUE-PRED diffs),
     monthly loss bars (+ diffs).

Usage: edit the variables in main() at the bottom and run the file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Config (same meaning as in the counterfactual script)
# ---------------------------------------------------------------------------
DAY_START_HOUR = 4
DAY_END_HOUR = 22
DEFAULT_DT_HOURS = 10.0 / 60.0
PANEL_AREA = 0.7               # m^2 -- psc : 0.7, solon : 1.44, sanyo : 1.125
MIN_IRR_FOR_PCE = 300.0
MIN_POWER_FOR_PCE = 20.0
TEMP_COL = "temp_C"
BRANCHES = ("true", "pred")


# ---------------------------------------------------------------------------
# Loading / preparation
# ---------------------------------------------------------------------------
def load_residual_csv(csv_path: str, tz: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=["timestamp"])

    # Mixed UTC offsets (DST) can leave an object index -> normalise
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True)
        if tz:
            idx = idx.tz_convert(tz)
        df.index = idx
    # Drop tz but keep local wall-clock time (needed for the daylight filter)
    if df.index.tz is not None:
        if tz:
            df.index = df.index.tz_convert(tz)
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    dup = df.index.duplicated(keep="first")
    if dup.any():
        print(f"  Removing {dup.sum()} duplicate timestamps")
        df = df[~dup]

    print(f"Loaded {csv_path}")
    print(f"  rows: {len(df):,}")
    print(f"  date range: {df.index.min()} -> {df.index.max()}")
    return df


def detect_targets(df: pd.DataFrame) -> list:
    targets = []
    for c in df.columns:
        if not c.startswith("residual_"):
            continue
        t = c.replace("residual_", "")
        needed = [f"true_{t}", f"pred_{t}", f"pred_{t}_age1"]
        missing = [n for n in needed if n not in df.columns]
        if missing:
            print(f"  WARNING: target '{t}' skipped, missing columns {missing}")
        else:
            targets.append(t)
    print(f"  Usable targets: {targets}")
    return targets


def prepare_target_df(df: pd.DataFrame, target: str, exclude_bad_days: bool) -> pd.DataFrame:
    """Bad-day filter, daylight filter, Irr/temp validity, date + dt_hours cols."""
    valid = pd.Series(True, index=df.index)

    if exclude_bad_days:
        bad_col = f"bad_day_{target}"
        if bad_col not in df.columns:
            cands = [c for c in df.columns if c.startswith("bad_day_")]
            bad_col = cands[0] if len(cands) == 1 else None
        if bad_col is not None:
            valid &= df[bad_col] < 0.5
            print(f"  Excluding bad days using column: {bad_col}")
        else:
            print("  WARNING: no bad_day column resolved, not excluding bad days.")

    hours = df.index.hour
    valid &= (hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR)
    valid &= df["Irr"].notna() & df[TEMP_COL].notna() & (df["Irr"] > 0)

    out = df.loc[valid].copy()
    print(f"  {target}: {len(out):,} / {len(df):,} rows kept "
          f"(bad-day + daylight {DAY_START_HOUR}-{DAY_END_HOUR}h + Irr>0)")

    out["date"] = out.index.normalize()
    step = out.index.to_series().diff().median()
    dt_hours = step.total_seconds() / 3600.0 if pd.notna(step) and step > pd.Timedelta(0) \
        else DEFAULT_DT_HOURS
    out["dt_hours"] = dt_hours
    print(f"  Time step: {dt_hours * 60:.1f} min")
    return out


# ---------------------------------------------------------------------------
# PCE for true / pred / age=1 pred
# ---------------------------------------------------------------------------
def calculate_pce(df: pd.DataFrame, target: str, panel_area: float = PANEL_AREA) -> pd.DataFrame:
    power = {
        "true": df[f"true_{target}"],
        "pred": df[f"pred_{target}"],
        "age1": df[f"pred_{target}_age1"],
    }
    denom = df["Irr"] * panel_area

    valid = df["Irr"].notna() & (df["Irr"] >= MIN_IRR_FOR_PCE) & df[TEMP_COL].notna()
    for s in power.values():
        valid &= s.notna() & (s >= MIN_POWER_FOR_PCE)
    print(f"  Shared valid timestamps (Irr >= {MIN_IRR_FOR_PCE}, "
          f"P >= {MIN_POWER_FOR_PCE} W for true/pred/age1): {valid.sum():,}/{len(df):,}")

    for k, s in power.items():
        df[f"pce_{k}"] = np.nan
        df.loc[valid, f"pce_{k}"] = s[valid] / denom[valid]
        df.loc[df[f"pce_{k}"] == 0, f"pce_{k}"] = np.nan

    # Joint 99th-percentile outlier filter
    outlier = pd.Series(False, index=df.index)
    for k in power:
        col = df[f"pce_{k}"]
        if col.notna().any():
            outlier |= col > col.quantile(0.99)
    before = {k: int(df[f"pce_{k}"].notna().sum()) for k in power}
    for k in power:
        df.loc[outlier, f"pce_{k}"] = np.nan
        df[f"pce_{k}"] *= 100.0
    print("  After joint 99th-pct filter, removed: " +
          ", ".join(f"{k}={before[k] - int(df[f'pce_{k}'].notna().sum())}" for k in power))

    print("  FINAL PCE (%):")
    for k in power:
        c = df[f"pce_{k}"]
        if c.notna().any():
            print(f"    pce_{k:<5}: min={c.min():.2f}  max={c.max():.2f}  valid={c.notna().sum():,}")
        else:
            print(f"    pce_{k:<5}: no valid values")
    return df


# ---------------------------------------------------------------------------
# eta_ref = age=1 PCE signal
# ---------------------------------------------------------------------------
def build_eta_ref_direct(df: pd.DataFrame) -> pd.Series:
    """eta_ref(t) = age=1 PCE at the same timestamp."""
    eta = df["pce_age1"].copy()
    print(f"  eta_ref (direct): {eta.notna().sum():,} filled | {eta.isna().sum():,} NaN")
    return eta


# ---------------------------------------------------------------------------
# Per-day reversible / irreversible loss decomposition
# ---------------------------------------------------------------------------
def compute_daily_degradation(df: pd.DataFrame):
    """
    Energy-based, relative-delta method. For each day and branch (true / pred):
      delta_rel = (eta_ref - PCE) / eta_ref (smoothed, negatives -> 0)
      scale     = clip(1 + min(delta_rel), 1, 1.2)
      PCE_adj   = PCE * scale
      total loss  = (E_ref - E_meas) / E_ref
      reversible  = (E_ref - E_adj)  / E_ref
      irreversible= (E_adj - E_meas) / E_ref
    """
    print(f"\n{'=' * 70}")
    print("COMPUTING PER-DAY REV / IRREV LOSS (energy-based, relative-delta)")
    print(f"{'=' * 70}")

    results, ts_records = [], []
    days_fully_removed = 0

    for date, g in df.groupby("date"):
        day = g[
            (g["Irr"] >= MIN_IRR_FOR_PCE)
            & g["pce_true"].notna() & g["pce_pred"].notna()
            & g["eta_ref"].notna() & g["dt_hours"].notna() & g[TEMP_COL].notna()
        ].copy()
        if len(day) < 5:
            continue
        n_orig = len(day)

        for c in ("pce_true", "pce_pred", "eta_ref"):
            day[f"{c}_smooth"] = day[c].rolling(window=3, center=True, min_periods=1).mean()

        n_above = {b: int((day[f"pce_{b}_smooth"] > day["eta_ref_smooth"]).sum()) for b in BRANCHES}

        for b in BRANCHES:
            d = ((day["eta_ref_smooth"] - day[f"pce_{b}_smooth"]) / day["eta_ref_smooth"]
                 ).replace([np.inf, -np.inf], np.nan)
            day[f"delta_rel_{b}"] = d.clip(lower=0)

        day = day.dropna(subset=[f"delta_rel_{b}" for b in BRANCHES])
        if day.empty:
            days_fully_removed += 1
            continue

        dt, irr = day["dt_hours"], day["Irr"]
        e_ref = (day["eta_ref_smooth"] / 100.0 * irr * dt).sum()
        if e_ref < 0.1:
            continue

        rec = {
            "date": date,
            "n_points_original": n_orig,
            "n_points_after_filter": len(day),
            "temp_mean": day[TEMP_COL].mean(),
            "temp_median": day[TEMP_COL].median(),
            "irr_mean": day["Irr"].mean(),
            "irr_median": day["Irr"].median(),
        }
        skip = False
        for b in BRANCHES:
            min_delta = day[f"delta_rel_{b}"].min()
            scale = float(np.clip(1.0 + min_delta, 1.0, 1.2))
            pce_adj = day[f"pce_{b}_smooth"] * scale

            e_meas = (day[f"pce_{b}_smooth"] / 100.0 * irr * dt).sum()
            e_adj = (pce_adj / 100.0 * irr * dt).sum()

            pr = e_meas / e_ref
            if pr < 0.1:
                skip = True
                break

            day[f"pce_{b}_adjusted"] = pce_adj
            day[f"scale_{b}"] = scale
            rec.update({
                f"n_above_ref_{b}": n_above[b],
                f"energy_exceeds_ref_{b}": bool(e_meas > e_ref),
                f"meas_pce_{b}": day[f"pce_{b}_smooth"].median(),
                f"pr_{b}": pr,
                f"e_ref_{b}": e_ref,
                f"e_meas_{b}": e_meas,
                f"e_adj_{b}": e_adj,
                f"total_loss_{b}_pct": max(0.0, (e_ref - e_meas) / e_ref) * 100,
                f"reversible_loss_{b}_pct": max(0.0, (e_ref - e_adj) / e_ref) * 100,
                f"irreversible_loss_{b}_pct": max(0.0, (e_adj - e_meas) / e_ref) * 100,
                f"min_gap_{b}": min_delta,
                f"scale_{b}": scale,
            })
        if skip:
            continue

        day["date"] = date
        ts_records.append(day)
        results.append(rec)

    if not results:
        print("  WARNING: no valid days after filtering.")
        return pd.DataFrame(), pd.DataFrame()

    daily_df = pd.DataFrame(results).set_index("date").sort_index()
    timestamp_df = pd.concat(ts_records).sort_index()

    print(f"  Days with valid decomposition : {len(daily_df)}")
    print(f"  Days fully removed (no delta) : {days_fully_removed}")
    for b, name in (("true", "TRUE"), ("pred", "PRED")):
        print(f"\n  {name} branch:")
        print(f"    Mean total loss        : {daily_df[f'total_loss_{b}_pct'].mean():.2f}%")
        print(f"    Mean reversible loss   : {daily_df[f'reversible_loss_{b}_pct'].mean():.2f}%")
        print(f"    Mean irreversible loss : {daily_df[f'irreversible_loss_{b}_pct'].mean():.2f}%")
        print(f"    Mean performance ratio : {daily_df[f'pr_{b}'].mean():.4f}")
        print(f"    Days with PCE > eta_ref: {int((daily_df[f'n_above_ref_{b}'] > 0).sum())}"
              f" / {len(daily_df)}")
    return daily_df, timestamp_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _error_metrics(true_vals, pred_vals) -> dict:
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    resid = t - p
    return dict(
        mae=np.mean(np.abs(resid)),
        rmse=np.sqrt(np.mean(resid ** 2)),
        mbe=np.mean(resid),
        r2=r2_score(t, p) if len(t) > 1 else np.nan,
    )


def plot_pce_over_time(df: pd.DataFrame, output_dir: Path, target: str):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.scatter(df.index, df["pce_true"], s=2, alpha=0.4, color="black", label="PCE true")
    ax.scatter(df.index, df["pce_pred"], s=2, alpha=0.4, color="tab:blue", label="PCE pred")
    ax.scatter(df.index, df["pce_age1"], s=2, alpha=0.4, color="tab:orange", label="PCE pred (age=1)")
    ax.scatter(df.index, df["eta_ref"], s=2, alpha=0.4, color="tab:green", label="eta_ref")
    ax.set_xlabel("Time")
    ax.set_ylabel("PCE (%)")
    ax.set_title(f"{target}: PCE over time  (true / pred / age=1 / eta_ref)")
    leg = ax.legend(loc="upper right", fontsize=10, markerscale=5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = output_dir / f"{target}_pce_over_time.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[SAVED] {fpath}")


def plot_daily_loss_bars(daily_df: pd.DataFrame, output_dir: Path,
                         start_date: str = None, end_date: str = None,
                         target: str = "Power"):
    if daily_df is None or len(daily_df) == 0:
        print("  WARNING: No daily data for loss bars plot")
        return

    plot_df = daily_df.copy()
    if start_date:
        plot_df = plot_df[plot_df.index >= pd.Timestamp(start_date)]
    if end_date:
        plot_df = plot_df[plot_df.index <= pd.Timestamp(end_date)]
    plot_df = plot_df[plot_df["reversible_loss_true_pct"].notna()
                      & plot_df["reversible_loss_pred_pct"].notna()].copy()
    if len(plot_df) == 0:
        print("  WARNING: No valid days in specified date range")
        return

    plot_df["reversible_diff_pct"] = (plot_df["reversible_loss_true_pct"]
                                      - plot_df["reversible_loss_pred_pct"])
    plot_df["irreversible_diff_pct"] = (plot_df["irreversible_loss_true_pct"]
                                        - plot_df["irreversible_loss_pred_pct"])

    rev_m = _error_metrics(plot_df["reversible_loss_true_pct"], plot_df["reversible_loss_pred_pct"])
    irr_m = _error_metrics(plot_df["irreversible_loss_true_pct"], plot_df["irreversible_loss_pred_pct"])
    print(f"  Valid days: {len(plot_df)}")
    print(f"  Reversible   -- R²={rev_m['r2']:.4f}  MAE={rev_m['mae']:.3f}%  "
          f"RMSE={rev_m['rmse']:.3f}%  MBE={rev_m['mbe']:+.3f}%")
    print(f"  Irreversible -- R²={irr_m['r2']:.4f}  MAE={irr_m['mae']:.3f}%  "
          f"RMSE={irr_m['rmse']:.3f}%  MBE={irr_m['mbe']:+.3f}%")

    fig, (ax_t, ax_p, ax_rev_diff, ax_irr_diff) = plt.subplots(
        4, 1, figsize=(min(max(14, len(plot_df) * 0.15), 60), 18))
    fig.suptitle(f"Daily Loss Decomposition | {target}", fontsize=24, fontweight="bold", y=0.99)

    x_pos = np.arange(len(plot_df))
    dates = plot_df.index.strftime("%Y-%m-%d")
    tick_positions = x_pos[::7]
    tick_labels = [dates[i] for i in range(0, len(dates), 7)]
    bar_width = 0.7

    def _style_axis(ax):
        ax.set_xlim(-1.0, len(plot_df) + 0.2)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.5, zorder=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax, (rcol, icol), title in [
        (ax_t, ("reversible_loss_true_pct", "irreversible_loss_true_pct"), "TRUE (Measured)"),
        (ax_p, ("reversible_loss_pred_pct", "irreversible_loss_pred_pct"), "Model (Predicted)"),
    ]:
        irr_vals = plot_df[icol].fillna(0).values
        rev_vals = plot_df[rcol].values
        ax.bar(x_pos, irr_vals, color="tomato", alpha=0.85, width=bar_width,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, color="mediumseagreen", alpha=0.85,
               width=bar_width, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        ax.set_ylim(0, max((irr_vals + rev_vals).max(), 1e-6) * 1.25)
        _style_axis(ax)
        ax.text(0.02, 0.96,
                f"Mean Rev = {rev_vals.mean():.2f}%\nMean Irr = {irr_vals.mean():.2f}%",
                transform=ax.transAxes, fontsize=20, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    for ax, dcol, m, title in [
        (ax_rev_diff, "reversible_diff_pct", rev_m, "TRUE - Model (Reversible)"),
        (ax_irr_diff, "irreversible_diff_pct", irr_m, "TRUE - Model (Irreversible)"),
    ]:
        vals = plot_df[dcol].values
        colors = np.where(vals > 0, "#1f77b4", "#ff7f0e")
        ax.bar(x_pos, vals, color=colors, alpha=0.8, width=bar_width,
               edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.02, 0.98,
                f"R²   = {m['r2']:.3f}\nMAE  = {m['mae']:.3f}%\n"
                f"RMSE = {m['rmse']:.3f}%\nMBE  = {m['mbe']:+.3f}%",
                transform=ax.transAxes, fontsize=15, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_title(title, fontweight="bold", fontsize=20)
        ax.set_ylabel("Energy loss (%)", fontsize=16, fontweight="bold")
        max_diff = max(np.abs(vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, zorder=2)
        _style_axis(ax)

    ax_irr_diff.set_xlabel("Day", fontsize=16, fontweight="bold", labelpad=20)

    handles = [
        Patch(facecolor="tomato", alpha=0.85, edgecolor="white", label="Irreversible"),
        Patch(facecolor="mediumseagreen", alpha=0.85, edgecolor="white", label="Reversible"),
        Patch(facecolor="#1f77b4", alpha=0.8, edgecolor="black", label="TRUE > Model"),
        Patch(facecolor="#ff7f0e", alpha=0.8, edgecolor="black", label="TRUE < Model"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4,
               fontsize=20, frameon=True, edgecolor="black")
    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0, rect=[0, 0, 1, 0.96])

    fname = output_dir / (f"{target}_daily_loss_bars_"
                          f"{plot_df.index.min():%Y%m%d}_{plot_df.index.max():%Y%m%d}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fname}")


def plot_monthly_loss_bars(daily_df: pd.DataFrame, output_dir: Path, target: str = "Power"):
    if daily_df is None or len(daily_df) == 0:
        print("  WARNING: No daily data for monthly plot")
        return

    dfc = daily_df.copy()
    dfc["year_month"] = dfc.index.strftime("%Y-%m")
    day_counts = dfc.groupby("year_month").size()

    # Simple mean of daily loss percentages (equal weight per day)
    monthly = dfc.groupby("year_month").agg(
        total_loss_true_pct=("total_loss_true_pct", "mean"),
        reversible_loss_true_pct=("reversible_loss_true_pct", "mean"),
        irreversible_loss_true_pct=("irreversible_loss_true_pct", "mean"),
        total_loss_pred_pct=("total_loss_pred_pct", "mean"),
        reversible_loss_pred_pct=("reversible_loss_pred_pct", "mean"),
        irreversible_loss_pred_pct=("irreversible_loss_pred_pct", "mean"),
    ).reset_index()

    monthly["reversible_diff_pct"] = monthly["reversible_loss_true_pct"] - monthly["reversible_loss_pred_pct"]
    monthly["irreversible_diff_pct"] = monthly["irreversible_loss_true_pct"] - monthly["irreversible_loss_pred_pct"]

    ok = monthly[["reversible_loss_true_pct", "reversible_loss_pred_pct",
                  "irreversible_loss_true_pct", "irreversible_loss_pred_pct"]].notna().all(axis=1)
    monthly = monthly[ok].reset_index(drop=True)
    if monthly.empty:
        print("  WARNING: no valid months for monthly plot")
        return

    rev_m = _error_metrics(monthly["reversible_loss_true_pct"], monthly["reversible_loss_pred_pct"])
    irr_m = _error_metrics(monthly["irreversible_loss_true_pct"], monthly["irreversible_loss_pred_pct"])

    x_pos = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Monthly Mean Loss Decomposition | {target}",
                 fontsize=24, fontweight="bold", y=1.02)
    axes = axes.flatten()

    def _month_axis(ax):
        ax.set_xlabel("Month", fontsize=16, fontweight="bold", labelpad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax_i, (rcol, icol), title in [
        (0, ("reversible_loss_true_pct", "irreversible_loss_true_pct"), "TRUE (Measured)"),
        (1, ("reversible_loss_pred_pct", "irreversible_loss_pred_pct"), "Model (Predicted)"),
    ]:
        ax = axes[ax_i]
        irr_vals = monthly[icol].fillna(0).values
        rev_vals = monthly[rcol].values
        total = irr_vals + rev_vals
        ax.bar(x_pos, irr_vals, color="tomato", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x_pos, rev_vals, bottom=irr_vals, color="mediumseagreen", alpha=0.85,
               edgecolor="white", linewidth=0.5)
        for i, row in monthly.iterrows():
            ax.text(i, total[i] * 1.05, f"n={day_counts[row['year_month']]}",
                    ha="center", va="bottom", fontsize=14, rotation=90, alpha=0.7)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
        ax.set_ylabel("Energy Loss (%)", fontsize=16, fontweight="bold")
        _month_axis(ax)

    for ax_i, dcol, m, title in [
        (2, "reversible_diff_pct", rev_m, "TRUE - Model (Reversible)"),
        (3, "irreversible_diff_pct", irr_m, "TRUE - Model (Irreversible)"),
    ]:
        ax = axes[ax_i]
        vals = monthly[dcol].values
        colors = np.where(vals > 0, "#1f77b4", "#ff7f0e")
        ax.bar(x_pos, vals, color=colors, alpha=0.8, width=0.6,
               edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(0.02, 0.98, f"R²   = {m['r2']:.3f}\nMAE  = {m['mae']:.3f}%",
                transform=ax.transAxes, fontsize=13, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylabel("Energy Loss Diff (%)", fontsize=16, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        max_diff = max(np.abs(vals).max(), 5)
        ax.set_ylim(-max_diff * 1.3, max_diff * 1.3)
        _month_axis(ax)

    handles = [
        Patch(facecolor="tomato", alpha=0.85, edgecolor="white", label="Irreversible"),
        Patch(facecolor="mediumseagreen", alpha=0.85, edgecolor="white", label="Reversible"),
        Patch(facecolor="#1f77b4", alpha=0.8, edgecolor="black", label="TRUE > Model"),
        Patch(facecolor="#ff7f0e", alpha=0.8, edgecolor="black", label="TRUE < Model"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4,
               fontsize=20, frameon=True, edgecolor="black")
    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0, rect=[0, 0, 1, 0.98])

    fpath = output_dir / f"{target}_monthly_loss_bars_with_diffs.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fpath}")

    monthly.to_csv(output_dir / f"{target}_monthly_loss_summary.csv", index=False)
    print(f"\n  Monthly Loss Summary [{target}]:")
    print(f"  {'Month':<9}{'Days':>5}{'TotTrue':>9}{'RevTrue':>9}{'IrrTrue':>9}"
          f"{'TotPred':>9}{'RevPred':>9}{'IrrPred':>9}{'RevDiff':>9}{'IrrDiff':>9}")
    for _, r in monthly.iterrows():
        print(f"  {r['year_month']:<9}{day_counts[r['year_month']]:>5}"
              f"{r['total_loss_true_pct']:>9.2f}{r['reversible_loss_true_pct']:>9.2f}"
              f"{r['irreversible_loss_true_pct']:>9.2f}{r['total_loss_pred_pct']:>9.2f}"
              f"{r['reversible_loss_pred_pct']:>9.2f}{r['irreversible_loss_pred_pct']:>9.2f}"
              f"{r['reversible_diff_pct']:>9.2f}{r['irreversible_diff_pct']:>9.2f}")
    print(f"\n  Monthly error metrics (Model vs TRUE) [{target}]:")
    print(f"  {'Component':<14}{'R²':>8}{'MAE':>8}{'RMSE':>8}{'MBE':>8}")
    for name, m in (("Reversible", rev_m), ("Irreversible", irr_m)):
        print(f"  {name:<14}{m['r2']:>8.4f}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['mbe']:>+8.3f}")



def plot_monthly_above_ref_pct(daily_df: pd.DataFrame, output_dir: Path, target: str = "Power"):
    """
    Bar chart: % of (post-filter) timestamps per month where PCE was above
    eta_ref (the age=1 signal) -- one bar for TRUE, one for the model's PRED,
    grouped sequentially by month.
    """
    if daily_df is None or len(daily_df) == 0:
        print("  WARNING: No daily data for above-ref plot")
        return

    dfc = daily_df.copy()
    
    dfc["period"] = dfc.index.to_period("M")
    dfc["year_month"] = dfc.index.strftime("%b")    
    day_counts = dfc.groupby("period").size()

    monthly = dfc.groupby(["period", "year_month"]).agg(
        n_points=("n_points_after_filter", "sum"),
        n_above_true=("n_above_ref_true", "sum"),
        n_above_pred=("n_above_ref_pred", "sum"),
    ).reset_index()
    
    monthly = monthly.sort_values("period").reset_index(drop=True)
    monthly = monthly[monthly["n_points"] > 0].reset_index(drop=True)
    if monthly.empty:
        print("  WARNING: no valid months for above-ref plot")
        return

    monthly["pct_above_true"] = monthly["n_above_true"] / monthly["n_points"] * 100.0
    monthly["pct_above_pred"] = monthly["n_above_pred"] / monthly["n_points"] * 100.0

    x_pos = np.arange(len(monthly))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(monthly) * 1.0), 6))
    ax.bar(x_pos - width / 2, monthly["pct_above_true"], width, 
           color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5, 
           label="TRUE (Measured)", zorder=3)
    ax.bar(x_pos + width / 2, monthly["pct_above_pred"], width, 
           color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.5, 
           label="LSTM (Predicted)", zorder=3)

    for i, row in monthly.iterrows():
        n_days = day_counts[row["period"]]
        y = max(row["pct_above_true"], row["pct_above_pred"])
        #ax.text(i, y + 1, f"n={n_days}d", ha="center", va="bottom", fontsize=10, alpha=0.7)

    ax.set_xlabel("Month", fontsize=16, fontweight="bold")
    ax.set_ylabel("Timestamps with PCE > \u03b7_ref (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"{target}: Monthly % of timestamps above age=1 reference (eta_ref)",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(monthly["year_month"], rotation=90, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, max(monthly["pct_above_true"].max(), monthly["pct_above_pred"].max(), 1) * 1.2)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fpath = output_dir / f"{target}_monthly_above_ref_pct.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fpath}")

    print(f"\n  Monthly % timestamps above eta_ref [{target}]:")
    print(f"  {'Month':<9}{'Days':>5}{'N pts':>8}{'%TRUE':>8}{'%Pred':>8}")
    for _, r in monthly.iterrows():
        print(f"  {r['year_month']:<9}{day_counts[r['period']]:>5}{r['n_points']:>8.0f}"
              f"{r['pct_above_true']:>8.2f}{r['pct_above_pred']:>8.2f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_loss_analysis(
    csv_path: str,
    output_dir: str = "./loss_decomposition",
    panel_area: float = PANEL_AREA,
    split: str = "all",
    exclude_bad_days: bool = True,
    tz: str = None,
    start_date: str = None,
    end_date: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_residual_csv(csv_path, tz=tz)
    targets = detect_targets(df)

    if split != "all":
        if "_split" not in df.columns:
            print(f"  WARNING: split='{split}' requested but no '_split' column, using all rows.")
        else:
            df = df[df["_split"] == split].copy()
            print(f"  Filtered to split='{split}': {len(df):,} rows")

    for target in targets:
        print(f"\n{'=' * 70}\nTARGET: {target}  (split={split}, panel_area={panel_area} m²)\n{'=' * 70}")
        tdir = output_dir / target
        tdir.mkdir(exist_ok=True)

        sub = prepare_target_df(df, target, exclude_bad_days)
        if len(sub) == 0:
            print("  WARNING: no rows left, skipping.")
            continue

        sub = calculate_pce(sub, target, panel_area=panel_area)
        if sub["pce_true"].notna().sum() == 0:
            print("  WARNING: no valid PCE rows, skipping.")
            continue

        sub["eta_ref"] = build_eta_ref_direct(sub)

        sub.to_csv(tdir / f"{target}_pce_signals.csv")
        plot_pce_over_time(sub, tdir, target)

        daily_df, timestamp_df = compute_daily_degradation(sub)
        if len(daily_df) == 0:
            continue

        daily_df.to_csv(tdir / f"{target}_daily_loss.csv")
        timestamp_df.to_csv(tdir / f"{target}_timestamp_loss_inputs.csv")
        print(f"[SAVED] {tdir / f'{target}_daily_loss.csv'}")

        print("\nPlotting daily loss bars...")
        plot_daily_loss_bars(daily_df, tdir, start_date=start_date, end_date=end_date, target=target)
        print("\nPlotting monthly loss bars...")
        plot_monthly_loss_bars(daily_df, tdir, target=target)
        print("\nPlotting monthly % above eta_ref...")
        plot_monthly_above_ref_pct(daily_df, tdir, target=target)


def main():
    CSV_PATH = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_25_150211_MLP_LSTM/lstm_results/residual_export/residual_export.csv"
    OUTPUT_DIR = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_25_150211_MLP_LSTM/lstm_results/loss_decomposition"

    run_loss_analysis(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        panel_area=0.7,          # m^2 -- psc: 0.7, solon: 1.44, sanyo: 1.125
        split="all",             # "all", "train", or "test"
        exclude_bad_days=True,
        tz=None,                 # e.g. "Europe/Berlin" if timestamps are tz-aware / mixed offset
        start_date=None,         # optional, for the daily bar plot, e.g. "2025-06-01"
        end_date=None,
    )


if __name__ == "__main__":
    main()