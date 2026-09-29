"""
1. Loads the trained MultiOutputLSTMModel, scalers and train and test datasets. 
2. Merges the train and test datasets into a continuous time sorted frame
3. Builds a non-overlapping sequesnces and passes throught he trained model
4. Denormalizes the predictions 
5. Adds cols: residual = pred - true to the trye, pred, env cols
6. saves the df as a csv file
"""

from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

from train_lstm_model import (
    MultiOutputLSTMModel,
    NonOverlappingMultiOutputDataset,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def inverse_transform(arr: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform a 1D array with a fitted MinMaxScaler (or pass through if None)."""
    arr = np.asarray(arr).ravel()
    if scaler is None:
        return arr.copy()
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def merge_train_test(train_path: Path, test_path: Path) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print("LOADING DATA")
    print(f"{'=' * 70}")

    train_df = pd.read_parquet(train_path).copy()
    test_df = pd.read_parquet(test_path).copy()
    train_df["_split"] = "train"
    test_df["_split"] = "test"

    merged = pd.concat([train_df, test_df]).sort_index()
    dup = merged.index.duplicated(keep=False)
    if dup.any():
        print(f"  Removing {dup.sum()} duplicate timestamps")
        merged = merged[~merged.index.duplicated(keep="first")]

    print(f"  Train rows : {(merged['_split'] == 'train').sum():,}")
    print(f"  Test rows  : {(merged['_split'] == 'test').sum():,}")
    print(f"  Date range : {merged.index.min()} -> {merged.index.max()}")
    return merged


def run_model_inference(model, dataset, device, batch_size: int = 256):
    """
    Runs the model over every sample in `dataset` and returns
    preds, trues, bad_day_masks -- each shape (n_samples, horizon, num_outputs),
    all still in SCALED units.
    """
    model.eval()
    all_preds, all_trues, all_bad = [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch = [dataset[i] for i in range(start, end)]
            xb = torch.stack([b[0] for b in batch]).to(device)
            yb = torch.stack([b[1] for b in batch])
            bmb = torch.stack([b[2] for b in batch])

            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(yb.numpy())
            all_bad.append(bmb.numpy())

    if not all_preds:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_trues, axis=0),
        np.concatenate(all_bad, axis=0),
    )


def build_full_export_frame(
    dataset: NonOverlappingMultiOutputDataset,
    preds_scaled: np.ndarray,
    trues_scaled: np.ndarray,
    bad_masks_scaled: np.ndarray,
    df_scaled: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str],
    target_scalers: dict,
    feature_scalers: dict,
    bad_day_mapping: dict,
    dataset_age1=None, 
    preds_age1_scaled=None,
) -> pd.DataFrame:
    """
    Builds one wide, timestamp-indexed dataframe with:
      - true_<target>, pred_<target>, residual_<target> (= pred - true), physical units
      - every feature_col in physical (denormalized) units, taken directly
        from df_scaled (not from the model's window input, so every
        timestamp in df_scaled gets a value, not just ones inside a
        prediction window)
      - bad_day_<target> columns, taken from df_scaled directly (0/1, not
        scaled)
      - _split column, if present in df_scaled
    """
    horizon = preds_scaled.shape[1]

    # ---- 1. Denormalize predictions + true targets, aligned by timestamp ----
    records: dict[pd.Timestamp, dict] = {}
    for i in range(len(dataset)):
        ts_index = dataset.get_timestamps_for_sample(i)
        if len(ts_index) != horizon:
            continue

        for h in range(horizon):
            ts = ts_index[h]
            if ts in records:
                continue  # non-overlapping dataset -> no dup expected, but be safe

            row = {}
            for t_idx, tname in enumerate(target_cols):
                sc = target_scalers.get(tname)
                pred_phys = inverse_transform(np.array([preds_scaled[i, h, t_idx]]), sc)[0]
                true_phys = inverse_transform(np.array([trues_scaled[i, h, t_idx]]), sc)[0]
                row[f"true_{tname}"] = float(true_phys)
                row[f"pred_{tname}"] = float(pred_phys)
                row[f"residual_{tname}"] = float(pred_phys - true_phys)  # pred - true
            records[ts] = row

    pred_df = pd.DataFrame.from_dict(records, orient="index")
    pred_df.index.name = "timestamp"
    pred_df = pred_df.sort_index()

    print(f"\nBuilt prediction/residual frame: {len(pred_df):,} timestamps")

    # ---- 1b. Denormalized age=1 counterfactual predictions (NEW) ----
    if dataset_age1 is not None and preds_age1_scaled is not None:
        age1_records: dict[pd.Timestamp, dict] = {}
        for i in range(len(dataset_age1)):
            ts_index = dataset_age1.get_timestamps_for_sample(i)
            if len(ts_index) != horizon:
                continue
            for h in range(horizon):
                ts = ts_index[h]
                if ts in age1_records:
                    continue
                row = {}
                for t_idx, tname in enumerate(target_cols):
                    sc = target_scalers.get(tname)
                    pred_age1_phys = inverse_transform(np.array([preds_age1_scaled[i, h, t_idx]]), sc)[0]
                    row[f"pred_{tname}_age1"] = float(pred_age1_phys)
                age1_records[ts] = row
        age1_df = pd.DataFrame.from_dict(age1_records, orient="index")
        age1_df.index.name = "timestamp"
    else:
        age1_df = pd.DataFrame(index=pd.Index([], name="timestamp"))

    # ---- 2. Denormalize every feature column directly from df_scaled ----
    # (covers the full df_scaled range, not just prediction-window timestamps,
    # so downstream you can see env conditions even where no prediction exists)
    feat_phys = {}
    for col in feature_cols:
        if col not in df_scaled.columns:
            continue
        scaler = feature_scalers.get(col)
        feat_phys[col] = pd.Series(
            inverse_transform(df_scaled[col].values, scaler),
            index=df_scaled.index,
        )
    feat_df = pd.DataFrame(feat_phys)

    # ---- 3. bad_day columns (already 0/1, not scaled -- pull as-is) ----
    bad_day_cols = sorted(set(c for c in bad_day_mapping.values() if c and c in df_scaled.columns))
    bad_df = df_scaled[bad_day_cols].copy() if bad_day_cols else pd.DataFrame(index=df_scaled.index)

    # ---- 4. _split column, if present ----
    split_df = df_scaled[["_split"]].copy() if "_split" in df_scaled.columns else pd.DataFrame(index=df_scaled.index)

    # ---- 5. Combine everything, aligned on the prediction frame's timestamps ----
    full_df = (pred_df
               .join(feat_df, how="left")
               .join(bad_df, how="left")
               .join(split_df, how="left")
               .join(age1_df, how="left")
               )
    full_df = full_df.sort_index()
    full_df.index.name = "timestamp"

    return full_df


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_export(
    train_data_path: str,
    test_data_path: str,
    model_path: str,
    model_info_path: str,
    feature_scaler_path: str,
    target_scaler_path: str,
    output_dir: str = "./residual_export",
    batch_size: int = 256,
) -> pd.DataFrame:
    train_data_path = Path(train_data_path)
    test_data_path = Path(test_data_path)
    model_path = Path(model_path)
    model_info_path = Path(model_info_path)
    feature_scaler_path = Path(feature_scaler_path)
    target_scaler_path = Path(target_scaler_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RESIDUAL EXPORT")
    print("=" * 70)

    # ---- 1. Load model info ----
    info = load_pickle(model_info_path)
    arch = info["architecture"]
    data_cfg = info["data_config"]

    feature_cols = data_cfg["features"]["feature_cols"]   # includes age col(s) -- plain model has no separate age head
    target_cols = data_cfg["features"]["target_cols"]
    use_bad_day = data_cfg["features"]["use_bad_day"]
    mask_bad_days = data_cfg["features"]["mask_bad_days"]
    bad_day_mapping = info.get("bad_day_mapping", data_cfg.get("bad_day_mapping", {}))
    window = data_cfg["window_horizon"]["window"]
    horizon = data_cfg["window_horizon"]["horizon"]

    print(f"\nLoaded model_info from {model_info_path}")
    print(f"  feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"  target_cols  : {target_cols}")
    print(f"  window/horizon: {window}/{horizon}")
    print(f"  bad_day_mapping: {bad_day_mapping}")

    # ---- 2. Load + merge data ----
    df_scaled = merge_train_test(train_data_path, test_data_path)

    missing_feats = [c for c in feature_cols if c not in df_scaled.columns]
    if missing_feats:
        raise ValueError(f"Merged data is missing feature column(s): {missing_feats}")
    missing_targets = [c for c in target_cols if c not in df_scaled.columns]
    if missing_targets:
        raise ValueError(f"Merged data is missing target column(s): {missing_targets}")

    # ---- 3. Load scalers ----
    feature_scaler_payload = load_pickle(feature_scaler_path)
    feature_scalers = feature_scaler_payload.get("feature_scalers", feature_scaler_payload)

    target_scaler_payload = load_pickle(target_scaler_path)
    if all(hasattr(v, "inverse_transform") for v in target_scaler_payload.values()):
        target_scalers = target_scaler_payload
    else:
        target_scalers = target_scaler_payload.get("target_scalers", target_scaler_payload)

    print(f"\nLoaded feature scalers from {feature_scaler_path} ({len(feature_scalers)} cols)")
    print(f"Loaded target scalers from {target_scaler_path}: {list(target_scalers.keys())}")

    # ---- 4. Build model ----
    device = torch.device("cpu")
    model = MultiOutputLSTMModel(
        input_size=len(feature_cols),
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=len(target_cols),
        dropout=arch["dropout"],
        forecast_steps=horizon,
    )
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"\nLoaded model weights from {model_path}")
    print(f"  total params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 5. Build dataset ----
    dataset = NonOverlappingMultiOutputDataset(
        df_scaled,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=window,
        horizon=horizon,
        use_bad_day=use_bad_day,
        mask_bad_days=mask_bad_days,
        bad_day_mapping=bad_day_mapping,
    )
    print(f"\nBuilt dataset: {len(dataset)} non-overlapping windows")

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset has 0 samples. Merged data likely has fewer rows than "
            f"window+horizon = {window + horizon}."
        )

    # ---- 6. Run inference ----
    print("\nRunning inference...")
    preds_scaled, trues_scaled, bad_masks_scaled = run_model_inference(
        model, dataset, device, batch_size=batch_size
    )
    print(f"  predictions shape (scaled): {preds_scaled.shape}")

    # ---- 6b. Counterfactual: age clamped to 1.0 day ----
    age_col = "days_since_install_perovskite"  # adjust to your actual column name
    age_scaler = feature_scalers.get(age_col)
    if age_scaler is not None and age_col in feature_cols:
        scaled_age_1 = float(age_scaler.transform([[1.0]])[0, 0])
        print(f"\n[AGE=1 COUNTERFACTUAL] scaled age=1.0 -> {scaled_age_1:.4f}")

        df_age1 = df_scaled.copy()
        df_age1[age_col] = scaled_age_1

        dataset_age1 = NonOverlappingMultiOutputDataset(
            df_age1,
            feature_cols=feature_cols,
            target_cols=target_cols,
            window=window,
            horizon=horizon,
            use_bad_day=use_bad_day,
            mask_bad_days=mask_bad_days,
            bad_day_mapping=bad_day_mapping,
        )
        preds_age1_scaled, _, _ = run_model_inference(model, dataset_age1, device, batch_size=batch_size)
        print(f"  age=1 predictions shape: {preds_age1_scaled.shape}")
    else:
        dataset_age1 = None
        preds_age1_scaled = None
        print(f"\n[AGE=1 COUNTERFACTUAL] skipped: '{age_col}' not found in feature_cols/scalers")

    # ---- 7. Build the full export frame ----
    print("\nBuilding full export dataframe (denormalized, timestamp-sorted)...")
    full_df = build_full_export_frame(
        dataset, preds_scaled, trues_scaled, bad_masks_scaled,
        df_scaled, target_cols, feature_cols,
        target_scalers, feature_scalers, bad_day_mapping,
        dataset_age1=dataset_age1, preds_age1_scaled=preds_age1_scaled,
    )

    print(f"\nFinal export frame: {len(full_df):,} rows, {len(full_df.columns)} columns")
    print(f"  columns: {list(full_df.columns)}")

    for t in target_cols:
        resid_col = f"residual_{t}"
        if resid_col in full_df.columns:
            resid = full_df[resid_col].dropna()
            print(f"\n  {t} residual (pred - true) stats:")
            print(f"    mean={resid.mean():+.3f}  std={resid.std():.3f}  "
                  f"min={resid.min():+.3f}  max={resid.max():+.3f}")

    # ---- 8. Save ----
    csv_path = output_dir / "residual_export.csv"
    parquet_path = output_dir / "residual_export.parquet"
    full_df.to_csv(csv_path)
    full_df.to_parquet(parquet_path)
    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {parquet_path}")

    return full_df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    BASE_PATH = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/PhD/Code/lstm_run_2026_09_26_133601"

    run_export(
        train_data_path=f"{BASE_PATH}/training_data/train_scaled.parquet",
        test_data_path=f"{BASE_PATH}/training_data/test_scaled.parquet",
        model_path=f"{BASE_PATH}/lstm_results/26.09.2026.133728_model_multi.pt",
        model_info_path=f"{BASE_PATH}/lstm_results/model_info_multi.pkl",
        feature_scaler_path=f"{BASE_PATH}/training_data/scalers.pkl",
        target_scaler_path=f"{BASE_PATH}/training_data/p_scalers.pkl",
        output_dir=f"{BASE_PATH}/lstm_results/residual_export",
        batch_size=256,
    )


if __name__ == "__main__":
    main()