#Version: 1.0 
#Handles only 1 output
from pathlib import Path
import pickle
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.dates as mdates
import warnings
import random
from typing import Optional, Tuple, Dict
from datetime import datetime

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, forecast_steps=1, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_steps = forecast_steps
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, forecast_steps * output_size)
        
        # ReLU activation to prevent negative predictions 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length= window + horizon, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use last timestep's hidden state for prediction
        # This captures information from the entire input sequence
        out = self.fc(out[:, -1, :])
        
        out = self.relu(out)
        
        # Reshape to (batch_size, forecast_steps, output_size)
        out = out.view(batch_size, self.forecast_steps, self.output_size)
        
        return out.squeeze(-1)  # (batch_size, horizon)
    
class MultiStepDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col, window: int, horizon: int, use_bad_day: bool = True, mask_bad_days: bool = True):
        """
        Creates sliding window samples for time series | stride = 1
        window: Lookback period (e.g., 48 timesteps)
        horizon: Forecast horizon (e.g., 36 timesteps)
        bad_day: Binary mask for low-quality data
        
        Input X includes: [window + horizon] timesteps of features
        Target y includes: only [horizon] timesteps of target
        """
        self.window = window
        self.horizon = horizon
        self.use_bad_day = use_bad_day
        self.mask_bad_days = mask_bad_days
        
        self.original_feature_cols = feature_cols.copy()
        # Create actual feature columns based on use_bad_day flag
        if use_bad_day:
            # Include 'bad_day' as a feature if it exists
            if 'bad_day' in df.columns and 'bad_day' not in feature_cols:
                self.feature_cols = feature_cols + ['bad_day']
            elif 'bad_day' in df.columns and 'bad_day' in feature_cols:
                self.feature_cols = feature_cols 
            else:
                self.feature_cols = feature_cols
                print(f"WARNING: use_bad_day=True but 'bad_day' column not found in DataFrame.")
        else:
            # Exclude 'bad_day' from features
            self.feature_cols = [col for col in feature_cols if col != 'bad_day']
        
        # Get feature data
        self.X = df[self.feature_cols].to_numpy(np.float32)
        self.y = df[target_col].to_numpy(np.float32)
        self.ts = df.index

        # Get bad_day data for masking (even if not used as feature)
        if 'bad_day' in df.columns:
            self.bad_day = df['bad_day'].to_numpy(np.float32)
        else:
            print(f"'bad_day' column not found. Assuming all days are valid (bad_day=0).")
            self.bad_day = np.zeros_like(self.y, dtype=np.float32)

        max_idx = len(df) - (window + horizon) + 1  # Input_size: window+horizon, Target_size: horizon
        self.valid_indices = list(range(max_idx))
        self.n = len(self.valid_indices)

        print(f"Dataset Configuration:")
        print(f"  use_bad_day={use_bad_day}, mask_bad_days={mask_bad_days}")
        print(f"  Features used: {self.feature_cols}")
        print(f"  Input shape per sample: ({window + horizon}, {len(self.feature_cols)})")
        print(f"  Target shape per sample: ({horizon},)")
        print(f"  Total samples: {self.n}")
        
        bad_day_counts = []
        for idx in self.valid_indices:
            bad_days_in_horizon = self.bad_day[idx + window: idx + window + horizon]
            bad_day_counts.append(np.sum(bad_days_in_horizon > 0.5))

        total_bad_days = sum(bad_day_counts)
        print(f"Bad day statistics: {total_bad_days} horizon periods with bad_day=1 "
              f"({total_bad_days/(self.n * horizon)*100:.1f}% of horizon steps)")
        
        i = self.valid_indices[0]
        print("-" * 100)
        print("[DEBUG DATASET]")
        print(f"X covers: {self.ts[i]} → {self.ts[i + self.window + self.horizon - 1]}")
        print(f"y covers: {self.ts[i + self.window]} → {self.ts[i + self.window + self.horizon - 1]}")

        print("First 5 y values:", self.y[i + self.window : i + self.window + 5])
        print("First 5 bad_day:", self.bad_day[i + self.window : i + self.window + 5])
        print("-" * 100)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        
        # Input includes window + horizon timesteps
        x = self.X[i : i + self.window + self.horizon].copy()  # (window+horizon, features)
        
        # Target starts after the entire input sequence
        y = self.y[i + self.window : i + self.window + self.horizon].copy()
        bad_day_mask = self.bad_day[i + self.window : i + self.window + self.horizon].copy()

        # Replace NaN targets with 0.0, loss will be masked for bad days
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        
        if self.mask_bad_days:
            bad_day_mask = bad_day_mask.astype(np.float32)
        else:
            bad_day_mask = np.zeros_like(y, dtype=np.float32)

        return (
            torch.from_numpy(x),            # (window+horizon, features)
            torch.from_numpy(y),            # (horizon,) 
            torch.from_numpy(bad_day_mask), # (horizon,) 
        )

    def get_timestamps_for_sample(self, idx):
        i = self.valid_indices[idx]
        # Return timestamps for the target horizon
        return self.ts[i + self.window : i + self.window + self.horizon]
    
    def get_feature_columns(self):
        """Return the actual feature columns used in the dataset"""
        return self.feature_cols

#NEW 
class NonOverlappingMultiStepDataset(Dataset):
    def __init__(self, df: pd.DataFrame,feature_cols, target_col, 
                 window: int, horizon: int, use_bad_day: bool = True, mask_bad_days: bool = True, debug: bool = True):
        """
        Creates NON-OVERLAPPING sliding window samples for time series (stride=horizon)        
        Input X includes: [window + horizon] timesteps of features
        Target y includes: [horizon] timesteps of target
        
        Non-overlapping sequences: stride=horizon, used for evaluation
        """
        self.window = window
        self.horizon = horizon
        self.use_bad_day = use_bad_day
        self.mask_bad_days = mask_bad_days
        self.debug = debug
        self.original_feature_cols = feature_cols.copy()
        # Create actual feature columns based on use_bad_day flag
        if use_bad_day:
            if 'bad_day' in df.columns and 'bad_day' not in feature_cols:
                self.feature_cols = feature_cols + ['bad_day']
            else:
                self.feature_cols = feature_cols
        else:
            # Exclude 'bad_day' from features
            self.feature_cols = [c for c in feature_cols if c != 'bad_day']
        self.X = df[self.feature_cols].to_numpy(np.float32)
        self.y = df[target_col].to_numpy(np.float32)
        self.ts = df.index

        # Get bad_day data for masking (even if not used as feature)
        if 'bad_day' in df.columns:
            self.bad_day = df['bad_day'].to_numpy(np.float32)
        else:
            print("WARNING: 'bad_day' column missing → assuming all good days")
            self.bad_day = np.zeros_like(self.y, dtype=np.float32)

        # Calculate non-overlapping indices with stride = horizon
        total_length = len(df)

        # stride = horizon : ensures no missing forecast steps
        stride = horizon
        max_idx = total_length - (window + horizon) + 1

        self.valid_indices = list(range(0, max_idx, stride))
        self.n = len(self.valid_indices)

        print(f"stride={stride}")
        print(f"window={window}, horizon={horizon}")
        print(f"Features: {self.feature_cols}")
        print(f"Input shape: ({window + horizon}, {len(self.feature_cols)})")
        print(f"Target shape: ({horizon},)")
        print(f"Total samples: {self.n}")
        
        # Calculate statistics for bad days in horizon
        bad_day_counts = []
        for i in self.valid_indices:
            bad_day_counts.append(np.sum(self.bad_day[i + window : i + window + horizon] > 0.5))

        total_bad = int(np.sum(bad_day_counts))
        print(
            f"Bad-day steps in horizon: {total_bad} / {self.n*horizon} "
            f"({100 * total_bad/(self.n*horizon):.2f}%)"
        )

        if debug:
            self._debug_sequence_alignment()
            self._debug_target_continuity()

    def _debug_sequence_alignment(self, n_show=3):
        print("\n[DEBUG] Sequence alignment check:")
        for k in range(min(n_show, self.n)):
            i = self.valid_indices[k]
            xs = i
            xe = i + self.window + self.horizon - 1
            ys = i + self.window
            ye = i + self.window + self.horizon - 1

            print(f"Sample {k}")
            print(f"  X: [{xs:6d} → {xe:6d}] | {self.ts[xs]} → {self.ts[xe]}")
            print(f"  y: [{ys:6d} → {ye:6d}] | {self.ts[ys]} → {self.ts[ye]}")
            print("")

    def _debug_target_continuity(self, n_check=8):
        print("\n[DEBUG] Target continuity check:")
        prev_end = None
        for k in range(min(n_check, self.n)):
            i = self.valid_indices[k]
            ys = i + self.window
            ye = i + self.window + self.horizon - 1

            if prev_end is not None:
                gap = ys - prev_end - 1
                status = "OK" if gap == 0 else f"GAP={gap}"
            else:
                status = "START"

            print(
                f"Sample {k}: y [{ys:6d} → {ye:6d}] | "
                f"{self.ts[ys]} → {self.ts[ye]} | {status}"
            )

            prev_end = ye

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.X[i : i + self.window + self.horizon].copy()
        y = self.y[i + self.window : i + self.window + self.horizon].copy()
        bad_day_mask = self.bad_day[i + self.window : i + self.window + self.horizon].copy()

        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        
        if self.mask_bad_days:
            bad_day_mask = bad_day_mask.astype(np.float32)
        else:
            bad_day_mask = np.zeros_like(y, dtype=np.float32)

        return (
            torch.from_numpy(x),            # (window+horizon, features)
            torch.from_numpy(y),            # (horizon,) 
            torch.from_numpy(bad_day_mask), # (horizon,) 
        )

    def get_timestamps_for_sample(self, idx):
        i = self.valid_indices[idx]
        # Return timestamps for the target horizon
        return self.ts[i + self.window : i + self.window + self.horizon]
    
    def get_feature_columns(self):
        """Return the actual feature columns used in the dataset"""
        return self.feature_cols
    
    def get_valid_indices(self):
        """Return the actual indices used in the dataset"""
        return self.valid_indices

#NEW
def split_train_val_from_sequences(
    X_seq,
    y_seq,
    bad_seq,
    ts_seq=None,
    validation_split: float = 0.1,
    use_validation_set: bool = True,
):
    """
    Split TRAIN sequences into train / val using modulo logic.
    """
    if not use_validation_set or validation_split <= 0:
        if ts_seq is not None:
            return X_seq, None, y_seq, None, bad_seq, None, ts_seq, None
        else:
            return X_seq, None, y_seq, None, bad_seq, None

    n = round(1 / validation_split)
    idx = np.arange(len(X_seq))

    val_mask = (idx % n) == 0
    train_mask = ~val_mask

    X_train = X_seq[train_mask]
    y_train = y_seq[train_mask]
    bad_train = bad_seq[train_mask]
    X_val = X_seq[val_mask]
    y_val = y_seq[val_mask]
    bad_val = bad_seq[val_mask]

    if ts_seq is not None:
        ts_train = [ts for i, ts in enumerate(ts_seq) if train_mask[i]]
        ts_val = [ts for i, ts in enumerate(ts_seq) if val_mask[i]]
    else:
        ts_train, ts_val = None, None

    print("\n[SEQUENCE SPLIT]")
    print(f"Total sequences : {len(X_seq)}")
    print(f"Train sequences : {len(X_train)}")
    print(f"Val sequences   : {len(X_val)}")

    return X_train, X_val, y_train, y_val, bad_train, bad_val, ts_train, ts_val

def masked_mae_with_bad_day(pred, true, bad_day_mask):
    """
    Compute MAE only on good days (bad_day == 0).
    Bad days contribute zero loss and zero gradient.
    Args:
        pred: (B, H) predictions
        true: (B, H) ground truth
        bad_day_mask: (B, H) bad_day mask (1=bad_day, 0=good_day)
    Returns:
        Scalar MAE loss computed only on good days
    """
    # mask for good days
    good_day_mask = (1.0 - bad_day_mask)
    
    diff = torch.abs(pred - true)    
    masked_diff = diff * good_day_mask
    # Sum over all elements, divide by number of good days
    total_good_days = torch.sum(good_day_mask) + 1e-8  
    
    return torch.sum(masked_diff) / total_good_days

def compute_masked_metrics_with_bad_day(
    true_series: pd.Series,
    pred_series: pd.Series,
    bad_series: pd.Series,
):
    """
    Calculate metrics for model evaluation using continuous series.
    Filters out NaN values and bad days.
    Args:
        true_series: Continuous time series of true values
        pred_series: Continuous time series of predicted values
        bad_series: Continuous time series of bad day mask (1=bad_day, 0=good_day)
    Returns:
        Dictionary of metrics computed only on good days
    """
    common_index = true_series.index.intersection(pred_series.index).intersection(bad_series.index)
    y_true = true_series.loc[common_index].values
    y_pred = pred_series.loc[common_index].values
    bad_flat = bad_series.loc[common_index].values
    
    # Filter NaN values and bad days
    mask = (~(np.isnan(y_true) | np.isnan(y_pred)) & (bad_flat < 0.5))
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "Directional_Accuracy": np.nan,
            "Skill_Score": np.nan,
            "Peak_Error": np.nan,
            "Data_Points": 0,
            "Bad_Day_Excluded": np.sum(bad_flat > 0.5),
            "Total_Points": len(y_true),
        }

    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))

    min_threshold = 0.1
    valid_indices = y_true_clean > min_threshold
    if np.sum(valid_indices) > 0:
        mape = (
            np.mean(
                np.abs(
                    (y_true_clean[valid_indices] - y_pred_clean[valid_indices])
                    / y_true_clean[valid_indices]
                )
            ) * 100
        )
    else:
        mape = 0.0
    
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    persistence_clean = np.roll(y_true_clean, 1)
    persistence_clean[0] = y_true_clean[0]

    persistence_mse = np.mean((y_true_clean - persistence_clean) ** 2)
    skill_score = 1 - (mse / (persistence_mse + 1e-8))

    if len(y_true_clean) > 1:
        y_true_diff = np.diff(y_true_clean)
        y_pred_diff = np.diff(y_pred_clean)
        directional_accuracy = np.mean(
            (y_true_diff > 0) == (y_pred_diff > 0)
        )
    else:
        directional_accuracy = np.nan

    peak_error = np.max(np.abs(y_true_clean - y_pred_clean))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Skill_Score": skill_score,
        "Directional_Accuracy": directional_accuracy,
        "Peak_Error": peak_error,
        "Data_Points": len(y_true_clean),
        "Bad_Day_Excluded": int(np.sum(bad_flat > 0.5)),
        "Total_Points": len(y_true),
    }

def predict_with_loader(model, loader, device):
    if loader is None:
        return None, None, None

    model.eval()
    all_preds = []
    all_trues = []
    all_bad_days = []
    with torch.no_grad():
        for xb, yb, bmb in loader:
            xb = xb.to(device)
            
            # Forward pass - input now includes window+horizon
            preds = model(xb).cpu().numpy()
            
            # Store results
            all_preds.append(preds)
            all_trues.append(yb.numpy())
            all_bad_days.append(bmb.numpy())
    
    if all_preds:
        preds_concat = np.vstack(all_preds)
        trues_concat = np.vstack(all_trues)
        bad_days_concat = np.vstack(all_bad_days)
    else:
        preds_concat = np.array([])
        trues_concat = np.array([])
        bad_days_concat = np.array([])
    
    return preds_concat, trues_concat, bad_days_concat

def build_continuous_series(dataset, preds, trues, bad_days, scaler, target_col: str = "P", mask_bad_days: bool = True, expected_freq: str = "10min"):
    """
    Build continuous normalized and denormalized time series for single target output
    with time continuity. Missing timestamps are filled with NaNs.

    Args:
        dataset: Dataset object
        preds: Predictions array (n_samples, horizon)
        trues: True values array (n_samples, horizon)
        bad_days: Bad day mask (n_samples, horizon)
        scaler: Scaler object for the target
        target_col: Target column name (default: "P")
        mask_bad_days: Whether to mask bad days
        expected_freq: Expected time resolution (default: 10min)

    Returns:
        (results_norm, results_denorm) where each is a dict with:
        target_col: (true_series, pred_series, bad_day_series)
    """
    if preds is None or trues is None or bad_days is None or len(preds) == 0:
        empty = pd.Series(dtype=float)
        return (
            {target_col: (empty.copy(), empty.copy(), empty.copy())},
            {target_col: (empty.copy(), empty.copy(), empty.copy())}
        )

    print(f"Preds shape     : {preds.shape}")
    print(f"Trues shape     : {trues.shape}")
    print(f"Bad_days shape  : {bad_days.shape}")
    print(f"Dataset length : {len(dataset)}")
    print(f"Horizon         : {preds.shape[1]}")
    print(f"Mask bad days   : {mask_bad_days}")

    # Extract predictions and truths
    P_pred_norm = preds  # (n_samples, horizon) - normalised
    P_true_norm = trues  # (n_samples, horizon) - normalised

    # Denormalize values
    P_pred_denorm = scaler.inverse_transform(P_pred_norm.reshape(-1, 1)).reshape(P_pred_norm.shape)
    P_true_denorm = scaler.inverse_transform(P_true_norm.reshape(-1, 1)).reshape(P_true_norm.shape)
    
    print(f"Denormalized predictions range: [{P_pred_denorm.min():.4f}, {P_pred_denorm.max():.4f}]")
    print(f"Denormalized truths range: [{P_true_denorm.min():.4f}, {P_true_denorm.max():.4f}]")
    
    n_samples = min(len(dataset), P_pred_norm.shape[0], P_true_norm.shape[0], bad_days.shape[0])
    pred_map_norm = {}
    true_map_norm = {}
    pred_map_denorm = {}
    true_map_denorm = {}
    bad_map = {}
    ts_count = {}

    for i in range(n_samples):
        timestamps = dataset.get_timestamps_for_sample(i)
        for j, ts in enumerate(timestamps):
            ts_count[ts] = ts_count.get(ts, 0) + 1
            bad_val = bad_days[i, j]
            
            # Track bad_day for every timestamp
            if ts not in bad_map:
                bad_map[ts] = bad_val

            # Skip bad day samples if masking enabled
            if mask_bad_days and bad_val > 0.5:
                continue
                
            # Store first occurrence for predictions/truths (good days only when masking)
            if ts not in pred_map_norm:
                # Normalized values
                pred_map_norm[ts] = P_pred_norm[i, j]
                true_map_norm[ts] = P_true_norm[i, j]
                # Denormalized values
                pred_map_denorm[ts] = P_pred_denorm[i, j]
                true_map_denorm[ts] = P_true_denorm[i, j]

    overlaps = {k: v for k, v in ts_count.items() if v > 1}
    print(f"Unique timestamps found: {len(ts_count)}")
    print(f"Timestamps: {len(overlaps)}")

    if overlaps:
        print(f"Max overlap count: {max(overlaps.values())}")

    ts_min = min(ts_count.keys())
    ts_max = max(ts_count.keys())

    full_index = pd.date_range(
        start=ts_min,
        end=ts_max,
        freq=expected_freq
    )

    print(f"Expected full grid size: {len(full_index)}")
    print(f"Actual predicted timestamps: {len(pred_map_norm)}")
    print(f"Missing timestamps (NaNs): {len(full_index) - len(pred_map_norm)}")

    pred_series_norm = pd.Series(pred_map_norm, name=f"{target_col}_pred_norm")
    true_series_norm = pd.Series(true_map_norm, name=f"{target_col}_true_norm")
    pred_series_denorm = pd.Series(pred_map_denorm, name=f"{target_col}_pred")
    true_series_denorm = pd.Series(true_map_denorm, name=f"{target_col}_true")
    bad_series = pd.Series(bad_map, name="bad_day")

    pred_series_norm = pred_series_norm.reindex(full_index)
    true_series_norm = true_series_norm.reindex(full_index)
    pred_series_denorm = pred_series_denorm.reindex(full_index)
    true_series_denorm = true_series_denorm.reindex(full_index)
    bad_series = bad_series.reindex(full_index)

    print(f"NaNs pred_norm   : {pred_series_norm.isna().sum()}")
    print(f"NaNs true_norm   : {true_series_norm.isna().sum()}")
    print(f"NaNs pred_denorm : {pred_series_denorm.isna().sum()}")
    print(f"NaNs true_denorm : {true_series_denorm.isna().sum()}")
    print(f"NaNs bad_day     : {bad_series.isna().sum()}")
    print("="*80 + "\n")

    results_norm = {target_col: (true_series_norm, pred_series_norm, bad_series)}
    results_denorm = {target_col: (true_series_denorm, pred_series_denorm, bad_series)}

    return results_norm, results_denorm

def train_loop(model, train_loader, val_loader, epochs, lr, patience, device, model_file):
    """
    Train LSTM with continuous forward passes and masked loss on bad days.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//4
    )
    
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch_idx, (xb, yb, bmb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            bmb = bmb.to(device)
            optimizer.zero_grad()            
            pred = model(xb)
            
            # Compute loss only on good days
            loss = masked_mae_with_bad_day(pred, yb, bmb)
            
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())

            if batch_idx == 0 and epoch % 10 == 0:
                good_samples = torch.sum(1.0 - bmb).item()
                bad_samples = torch.sum(bmb).item()
                print(
                    f"Batch {batch_idx}: "
                    f"Good={good_samples}, Bad={bad_samples}, "
                    f"Loss={loss.item():.6f}"
                )
            
            if epoch == 1 and batch_idx == 0:
                print("-" * 100)
                print("\n[DEBUG]")
                print("Input:", xb.shape)
                print("Pred:", pred.shape)
                print("Target:", yb.shape)
                print("-" * 100)

        avg_train_loss = np.mean(train_losses) if train_losses else float("inf")
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb, bmb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                bmb = bmb.to(device)
                pred = model(xb)
                loss = masked_mae_with_bad_day(pred, yb, bmb)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["val_loss"].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val - 1e-6: #early stopping
            best_val = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), model_file)
            print(f"  -> Saved model (val loss: {avg_val_loss:.6f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    try:
        state_dict = torch.load(model_file, map_location=device)
    except Exception:
        try:
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        except Exception:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        state_dict = torch.load(model_file, map_location=device, weights_only=False)

    model.load_state_dict(state_dict)
    return history

def save_scatter_ax(ax, y_true, y_pred, title):
    """
    Create scatter plot, ignoring NaN values.
    """
    # Flatten and remove NaNs
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    # Find valid pairs (both non-NaN)
    valid_mask = ~(np.isnan(yt) | np.isnan(yp))
    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]
    
    if len(yt_valid) == 0:
        ax.text(0.5, 0.5, "No valid data to plot", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return
    
    ax.scatter(yt_valid, yp_valid, s=2, alpha=0.5)
    lims = [min(yt_valid.min(), yp_valid.min()), 
            max(yt_valid.max(), yp_valid.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1)
    ax.set_xlabel("Actual (normalized)")
    ax.set_ylabel("Predicted (normalized)")
    ax.set_title(f"{title} (n={len(yt_valid)})")
    ax.grid(True, alpha=0.3)

def save_training_history_ax(ax, history):
    """
    Plot training and validation loss curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], '-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history["val_loss"], '-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final values
    if epochs:
        ax.text(0.98, 0.98, f"Final Train: {history['train_loss'][-1]:.4f}\nFinal Val: {history['val_loss'][-1]:.4f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def plot_time_series_ax(ax, true_series, pred_series, bad_series, title):
    """
    Plot time series, handling NaNs and bad days.
    """
    if true_series.empty or pred_series.empty:
        ax.text(0.5, 0.5, "No data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return
    
    valid_mask = (~np.isnan(true_series.values)) & (bad_series.values < 0.5)
    if np.any(valid_mask):
        valid_times = true_series.index[valid_mask]
        ax.plot(valid_times, true_series.values[valid_mask], 
                label="Actual", linewidth=1.5, alpha=0.8)
        ax.plot(valid_times, pred_series.reindex(true_series.index).values[valid_mask], 
                "--", label="Predicted", linewidth=1.2, alpha=0.8)
        
        # Calculate and display metrics
        y_true = true_series.values[valid_mask]
        y_pred = pred_series.reindex(true_series.index).values[valid_mask]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        ax.text(0.02, 0.98, f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}",
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No valid true values", 
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Value")
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def build_pdf_report(out_pdf_path, model_info, train_stats, val_stats, test_stats,
                     train_shape, val_shape, test_shape,
                     feature_cols, metrics_tr, metrics_va, metrics_te,
                     history, series_dict_train, series_dict_val, series_dict_test=None):

    wrapped_features = textwrap.wrap(", ".join(feature_cols), width=90)
    
    with PdfPages(out_pdf_path) as pdf:
        # PAGE 1: Data Statistics Summary
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        plt.axis("off")
        
        text_lines = [
            "DATA STATISTICS SUMMARY",
            "=" * 50,
            "",
            "TRAIN DATA:",
            f"  Period: {train_stats['start_date']} to {train_stats['end_date']}",
            f"  Days: {train_stats['num_days']}",
            f"  Rows: {train_stats['num_rows']}",
            f"  Bad days: {train_stats['bad_days']} ({train_stats['bad_days_pct']:.1f}%)",
            "",
        ]
        
        if train_stats['feature_stats']:
            text_lines.append("  Features Statistics (scaled values):")
            for feat, stats in train_stats['feature_stats'].items():
                text_lines.append(f"    {feat:15s}: min={stats['min']:6.3f}, max={stats['max']:6.3f}, "
                                 f"mean={stats['mean']:6.3f}, std={stats['std']:6.3f}")
        
        text_lines.extend([
            "",
            "VALIDATION DATA:",
        ])
        if val_stats is not None:
            text_lines.extend([
                f"  Period: {val_stats['start_date']} to {val_stats['end_date']}",
                f"  Days: {val_stats['num_days']}",
                f"  Rows: {val_stats['num_rows']}",
                f"  Bad days: {val_stats['bad_days']} ({val_stats['bad_days_pct']:.1f}%)",
            ])
        else:
            text_lines.append("  Created from train split")
        
        text_lines.extend([
            "",
            "TEST DATA:",
        ])
        
        if test_stats:
            text_lines.extend([
                f"  Period: {test_stats['start_date']} to {test_stats['end_date']}",
                f"  Days: {test_stats['num_days']}",
                f"  Rows: {test_stats['num_rows']}",
                f"  Bad days: {test_stats['bad_days']} ({test_stats['bad_days_pct']:.1f}%)",
            ])
        else:
            text_lines.append("  Not available")
        
        # Get flag configuration from model_info
        use_bad_day = model_info['data_config']['features']['use_bad_day']
        mask_bad_days = model_info['data_config']['features']['mask_bad_days']
        
        # Determine mode description
        if not use_bad_day and not mask_bad_days:
            mode_desc = "Mode 1: Don't feed bad_day as input & Don't mask bad days in loss"
        elif use_bad_day and mask_bad_days:
            mode_desc = "Mode 2: Feed bad_day as input & Mask bad days in loss"
        elif not use_bad_day and mask_bad_days:
            mode_desc = "Mode 3: Don't feed bad_day as input & Mask bad days in loss"
        elif use_bad_day and not mask_bad_days:
            mode_desc = "Mode 4: Feed bad_day as input & Don't mask bad days in loss"
        
        text_lines.extend([
            "",
            "=" * 50,
            "",
            "MODEL CONFIGURATION",
            mode_desc,
            "",
            "Bad_day Configuration:",
            f"  use_bad_day={use_bad_day} (bad_day {'included' if use_bad_day else 'excluded'} as input feature)",
            f"  mask_bad_days={mask_bad_days} (bad days {'masked' if mask_bad_days else 'not masked'} in loss calculation)",
            "",
            "Features (in feed order):",
        ])
        
        # Add features in feed order
        for i, feat in enumerate(feature_cols):
            text_lines.append(f"  {i+1}. {feat}")
        
        text_lines.extend([
            "",
            "Hyperparameters:",
            f"  WINDOW={model_info['dataset_info']['window']}, HORIZON={model_info['dataset_info']['horizon']}, BATCH_SIZE={model_info['training_config']['hyperparameters']['batch_size']}",
            f"  EPOCHS={model_info['training_config']['hyperparameters']['epochs']}, LR={model_info['training_config']['hyperparameters']['lr']:.6f}, PATIENCE={model_info['training_config']['hyperparameters']['patience']}",
            "",
            "Model parameters:",
            f"  hidden_size={model_info['architecture']['hidden_size']}, num_layers={model_info['architecture']['num_layers']}, dropout={model_info['architecture']['dropout']}",
            f"  total_parameters={model_info['architecture']['total_params']:,}",
            "",
            "Data splits (sequences):",
            f"  Train sequences: {train_shape}",
            f"  Val sequences:   {val_shape}",
            f"  Test sequences:  {test_shape if test_stats else 'N/A'}",
        ])
        
        plt.text(
            0.02, 0.98,
            "\n".join(text_lines),
            va="top", ha="left",
            fontsize=8, family="monospace",
            transform=fig.transFigure
        )
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: Metrics Table
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        
        # Get flag configuration again for metrics page
        use_bad_day = model_info['data_config']['features']['use_bad_day']
        mask_bad_days = model_info['data_config']['features']['mask_bad_days']
        
        # Build text content
        text_lines = [
            "MODEL PERFORMANCE METRICS",
            f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Bad_day Configuration: use_bad_day={use_bad_day}, mask_bad_days={mask_bad_days}",
            "",
            "Feature list (in feed order):"
        ]
        text_lines.extend(["  " + line for line in wrapped_features])
        plt.text(
            0.02, 0.98,
            "\n".join(text_lines),
            va="top", ha="left",
            fontsize=9, family="monospace",
            transform=fig.transFigure
        )
        
        # Metrics table
        metric_keys = [
            "MAE", "RMSE", "MAPE", "R2", 
            "Directional_Accuracy", "Skill_Score", "Peak_Error", 
            "Data_Points", "Bad_Day_Excluded", "Total_Points"
        ]
        
        def make_metrics_table(ax, title):
            # Create columns based on whether test data exists
            if test_stats and metrics_te is not None:
                table_data = [["Metric", "Train", "Val", "Test"]]
                
                for key in metric_keys:
                    if key == "Data_Points" or key == "Bad_Day_Excluded" or key == "Total_Points":
                        row = [
                            key,
                            f"{metrics_tr.get(key, 0)}",
                            f"{metrics_va.get(key, 0)}",
                            f"{metrics_te.get(key, 0)}" if metrics_te else "N/A"
                        ]
                    else:
                        row = [
                            key,
                            f"{metrics_tr.get(key, np.nan):.4f}",
                            f"{metrics_va.get(key, np.nan):.4f}",
                            f"{metrics_te.get(key, np.nan):.4f}" if metrics_te else "N/A"
                        ]
                    table_data.append(row)
            else:
                table_data = [["Metric", "Train", "Val"]]
                
                for key in metric_keys:
                    if key == "Data_Points" or key == "Bad_Day_Excluded" or key == "Total_Points":
                        row = [
                            key,
                            f"{metrics_tr.get(key, 0)}",
                            f"{metrics_va.get(key, 0)}",
                        ]
                    else:
                        row = [
                            key,
                            f"{metrics_tr.get(key, np.nan):.4f}",
                            f"{metrics_va.get(key, np.nan):.4f}",
                        ]
                    table_data.append(row)
            
            # Create table
            table = ax.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            
            # Header styling
            num_cols = len(table_data[0])
            for j in range(num_cols):
                table[(0, j)].set_facecolor('#DDDDDD')
                table[(0, j)].set_text_props(weight='bold')
            
            ax.set_title(title, fontsize=10, weight='bold', pad=20)
            ax.axis('off')
        
        # Add table
        ax_table = fig.add_axes([0.1, 0.15, 0.8, 0.35])
        table_title = "METRICS"
        if mask_bad_days:
            table_title += " (computed only on good days)"
        else:
            table_title += " (computed on all days)"
        
        if test_stats:
            table_title += " - Train/Val/Test"
        make_metrics_table(ax_table, table_title)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 3: Scatter plots + history
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        target_col = list(series_dict_train.keys())[0]

        # Prepare data for scatter plots from normalized series
        train_true_series_denorm, train_pred_series_denorm, train_bad_series_denorm = series_dict_train[target_col]
        val_true_series_denorm, val_pred_series_denorm, val_bad_series_denorm = series_dict_val[target_col]
        
        # Filter out NaNs and bad days for scatter plots
        train_valid_mask = ((~np.isnan(train_true_series_denorm.values)) & (~np.isnan(train_pred_series_denorm.values)) & (train_bad_series_denorm == 0))
        val_valid_mask = ((~np.isnan(val_true_series_denorm.values)) & (~np.isnan(val_pred_series_denorm.values)) & (val_bad_series_denorm == 0))
        
        train_y_true_denorm = train_true_series_denorm.values[train_valid_mask]
        train_y_pred_denorm = train_pred_series_denorm.values[train_valid_mask]
        val_y_true_denorm = val_true_series_denorm.values[val_valid_mask]
        val_y_pred_denorm = val_pred_series_denorm.values[val_valid_mask]
        
        save_scatter_ax(axes[0, 0], train_y_true_denorm, train_y_pred_denorm, "Train: Pred vs Actual (De-Normalized)")
        save_scatter_ax(axes[0, 1], val_y_true_denorm, val_y_pred_denorm, "Val: Pred vs Actual (De-Normalized)")
        
        # Error distribution histograms
        if len(train_y_true_denorm) > 0:
            train_errors = train_y_pred_denorm - train_y_true_denorm
            axes[1, 0].hist(train_errors, bins=50, alpha=0.7, edgecolor='black', density=False)
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
            axes[1, 0].axvline(x=np.mean(train_errors), color='green', linestyle='-', linewidth=1.5, 
                              label=f'Mean: {np.mean(train_errors):.3f}')
            axes[1, 0].set_title("Train Error Distribution (De-Normalized)")
            axes[1, 0].set_xlabel("Error (Predicted - Actual)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "No train data", ha='center', va='center')
        
        save_training_history_ax(axes[1, 1], history)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # PAGE 4: Time series plots
        if test_stats and series_dict_test is not None:
            fig, axes = plt.subplots(3, 1, figsize=(11.69, 16))
            train_true_series, train_pred_series, train_bad_series = series_dict_train[target_col]
            val_true_series, val_pred_series, val_bad_series = series_dict_val[target_col]
            if series_dict_test is not None:
                test_true_series, test_pred_series, test_bad_series = series_dict_test[target_col]
            
            plot_time_series_ax(axes[0], train_true_series, train_pred_series, train_bad_series, "Train: Actual vs Predicted")
            plot_time_series_ax(axes[1], val_true_series, val_pred_series, val_bad_series, "Validation: Actual vs Predicted")
            plot_time_series_ax(axes[2], test_true_series, test_pred_series, test_bad_series, "Test: Actual vs Predicted")
            axes[2].set_xlabel("Time")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(11.69, 12))
            train_true_series, train_pred_series, train_bad_series = series_dict_train[target_col]
            val_true_series, val_pred_series, val_bad_series = series_dict_val[target_col]
            
            plot_time_series_ax(axes[0], train_true_series, train_pred_series, train_bad_series, "Train: Actual vs Predicted")
            plot_time_series_ax(axes[1], val_true_series, val_pred_series, val_bad_series, "Validation: Actual vs Predicted")
            axes[1].set_xlabel("Time")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # PAGE 5: Additional Analysis
        if test_stats and series_dict_test is not None:
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            
            # Test scatter plot
            test_true_series_denorm, test_pred_series_denorm, test_bad_series_denorm = series_dict_test[target_col]
            test_valid_mask = ((~np.isnan(test_true_series_denorm.values)) & (~np.isnan(test_pred_series_denorm.values)) & (test_bad_series_denorm == 0))
            test_y_true_denorm = test_true_series_denorm.values[test_valid_mask]
            test_y_pred_denorm = test_pred_series_denorm.values[test_valid_mask]
            
            save_scatter_ax(axes[0, 0], test_y_true_denorm, test_y_pred_denorm, "Test: Pred vs Actual")
            
            # Test error distribution (de-normalized)
            if len(test_y_true_denorm) > 0:
                test_errors = test_y_pred_denorm - test_y_true_denorm
                axes[0, 1].hist(test_errors, bins=50, alpha=0.7, edgecolor='black', density=False, color='orange')
                axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
                axes[0, 1].axvline(x=np.mean(test_errors), color='green', linestyle='-', linewidth=1.5, 
                                  label=f'Mean: {np.mean(test_errors):.3f}')
                axes[0, 1].set_title("Test Error Distribution (De-Normalized)")
                axes[0, 1].set_xlabel("Error (Predicted - Actual)")
                axes[0, 1].set_ylabel("Frequency")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, "No test data", ha='center', va='center')
            
            # Data coverage comparison
            split_names = ['Train', 'Val']
            train_true_series_norm, _, _ = series_dict_train[target_col]
            val_true_series_norm, _, _ = series_dict_val[target_col]
            split_counts = [len(train_true_series_norm.dropna()), len(val_true_series_norm.dropna())]
            colors = ['blue', 'green']
            
            if series_dict_test is not None:
                test_true_series_norm, _, _ = series_dict_test[target_col]
                split_names.append('Test')
                split_counts.append(len(test_true_series_norm.dropna()))
                colors.append('orange')
            
            axes[1, 0].bar(split_names, split_counts, color=colors)
            axes[1, 0].set_title('Data Points Comparison (After Bad Day Filtering)')
            axes[1, 0].set_ylabel('Number of Points')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(split_counts):
                axes[1, 0].text(i, v + max(split_counts) * 0.01, str(v), ha='center')
            
            # Test error over time (using denormalized data)
            if test_y_true_denorm.size > 0:
                valid_mask = ~np.isnan(test_y_true_denorm)
                if np.any(valid_mask):
                    test_times = test_true_series_denorm.index[test_valid_mask][valid_mask]
                    test_errors_masked = (test_y_true_denorm - test_y_pred_denorm)[valid_mask]

                    axes[1, 1].plot(test_times, test_errors_masked, '-', alpha=0.7, linewidth=1)
                    axes[1, 1].axhline(y=0, linestyle='--', linewidth=1)
                    axes[1, 1].set_title("Test Errors Over Time (De-Normalized)")
                    axes[1, 1].set_xlabel("Time")
                    axes[1, 1].set_ylabel("Error")
                    axes[1, 1].grid(True, alpha=0.3)
                    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        print(f"PDF report saved to {out_pdf_path}")

def save_daily_pred_vs_true_plots(
    df: pd.DataFrame,
    series_dict: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    split_name: str,
    plots_dir: Path,
    n_days: int = 10,
    irr_col: str = "Irr",
    min_points_per_day: int = 45,  
    random_seed: int | None = 42,
):
    if random_seed is not None:
        random.seed(random_seed)

    df = df.copy()
    
    # Get the target column name from the series_dict keys
    target_col = list(series_dict.keys())[0]
    true_series, pred_series, _ = series_dict[target_col]
    
    df = df.join(true_series.rename(f"{target_col}_true"), how="left")
    df = df.join(pred_series.rename(f"{target_col}_pred"), how="left")

    df["date"] = df.index.date

    valid_days = []

    for day, day_df in df.groupby("date"):
        # skip bad days
        if (day_df["bad_day"] == 1).any():
            continue
        day_df_valid = day_df.dropna(subset=[f"{target_col}_true", f"{target_col}_pred", irr_col])
        if day_df_valid.empty:
            continue

        # ensure sufficient samples in the day
        if len(day_df_valid) < min_points_per_day:
            continue

        valid_days.append(day)

    if len(valid_days) == 0:
        print("[ERROR] No valid complete days found")
        return

    n_select = min(n_days, len(valid_days))
    selected_days = random.sample(valid_days, n_select)

    print(f"[INFO] Selected {n_select} random complete days for plotting")

    for day in selected_days:
        day_df = df[df["date"] == day]
        day_df = day_df.dropna(subset=[f"{target_col}_true", f"{target_col}_pred", irr_col])

        times = day_df.index
        month = times[0].to_period("M")

        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(
            times,
            day_df[f"{target_col}_true"],
            label="True",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax1.plot(
            times,
            day_df[f"{target_col}_pred"],
            label="Pred",
            linestyle="--",
            marker="x",
            markersize=4,
        )

        ax1.set_xlabel("Time (hour)")
        ax1.set_ylabel(f"{target_col} (unscaled)")
        ax1.set_title(f"{split_name} | {month} | {day}")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

        ax2 = ax1.twinx()
        ax2.plot(times, day_df[irr_col], label="Irr (scaled)", alpha=0.6)
        ax2.set_ylabel("Irr (scaled)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            ncol=3,
            fontsize=8,
        )

        plt.tight_layout()
        fname = plots_dir / f"{split_name.lower()}_{day}_pred_vs_true.png"
        plt.savefig(fname, dpi=150)
        plt.close()

def get_data_statistics(df: pd.DataFrame, split_name: str = "") -> dict:
    if df is None or len(df) == 0:
        return None
    
    stats = {
        "split_name": split_name,
        "num_rows": len(df),
        "start_date": df.index.min().strftime('%Y-%m-%d'),
        "end_date": df.index.max().strftime('%Y-%m-%d'),
        "num_days": (df.index.max() - df.index.min()).days + 1,
    }
    
    # Calculate bad days
    if 'bad_day' in df.columns:
        bad_days = df['bad_day'].sum()
        stats['bad_days'] = int(bad_days)
        stats['bad_days_pct'] = (bad_days / len(df)) * 100
    else:
        stats['bad_days'] = 0
        stats['bad_days_pct'] = 0.0
    
    # Calculate feature statistics
    feature_stats = {}
    for col in df.columns:
        if col != 'bad_day' and pd.api.types.is_numeric_dtype(df[col]):
            feature_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'nan_count': int(df[col].isna().sum()),
                'nan_pct': float(df[col].isna().sum() / len(df) * 100)
            }
    
    stats['feature_stats'] = feature_stats
    
    return stats

def run_lstm_training(
    training_data_dir: str,
    output_dir: str = None,
    window: int = 48,
    horizon: int = 12,
    batch_size: int = 36,
    epochs: int = 200,
    lr: float = 0.0001,
    patience: int = 30,
    hidden_size: int = 32,
    num_layers: int = 4,
    dropout: float = 0.1,
    target_col: str = "P",
    use_bad_day: bool = False,
    mask_bad_days: bool = False,
    validation_split: float = 0.15,
    random_seed: int = 42,
):
    """
    Check if train data exists
    Check if validation data exists and is not empty
    If no validation data, split from train using sequences
    Check if test data exists and is not empty
    Only evaluate test if it exists
    
    Training uses overlapping sequences (MultiStepDataset with stride=1)
    Evaluation uses non-overlapping sequences (NonOverlappingMultiStepDataset with stride=horizon)
    
    Flag modes:
    1. use_bad_day=False, mask_bad_days=False: Don't feed bad_day as input & Don't mask bad days in loss
    2. use_bad_day=True, mask_bad_days=True: Feed bad_day as input & Mask bad days in loss
    3. use_bad_day=False, mask_bad_days=True: Don't feed bad_day as input & Mask bad days in loss
    4. use_bad_day=True, mask_bad_days=False: Feed bad_day as input & Don't mask bad days in loss
    """
    def set_all_seeds(seed: int):
        import os
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass        
        print(f"All random seeds set to: {seed}")
        print(f"Deterministic mode: ON")
    # Set all seeds at the beginning
    set_all_seeds(random_seed)
    # Print flag configuration
    print("\nFlag config:")
    print(f"use_bad_day={use_bad_day}")
    print(f"mask_bad_days={mask_bad_days}")
    
    if not use_bad_day and not mask_bad_days:
        print("\nMODE 1: Don't feed bad_day as input & Don't mask bad days in loss")
    elif use_bad_day and mask_bad_days:
        print("\nMODE 2: Feed bad_day as input & Mask bad days in loss")
    elif not use_bad_day and mask_bad_days:
        print("\nMODE 3: Don't feed bad_day as input & Mask bad days in loss")
    elif use_bad_day and not mask_bad_days:
        print("\nMODE 4: Feed bad_day as input & Don't mask bad days in loss")
    
    # Set output directory
    OUT_DIR = Path(output_dir) if output_dir else Path(training_data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    TRAIN_FPATH = Path(training_data_dir) / "train_scaled.parquet"
    VAL_FPATH = Path(training_data_dir) / "val_scaled.parquet"
    TEST_FPATH = Path(training_data_dir) / "test_scaled.parquet"
    timestamp = datetime.now().strftime("%d.%m.%Y.%H%M%S")
    MODEL_FILE = OUT_DIR / f"{timestamp}_model.pt"
    METRICS_LOG = OUT_DIR / "training_metrics.txt"
    PLOTS_DIR = OUT_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Load the P scaler
    P_SCALER_FILE = Path(training_data_dir) / "p_scaling.pkl"
    
    # Load scaler
    with open(P_SCALER_FILE, "rb") as f:
        scaler_data = pickle.load(f)
        p_scaler = scaler_data["scaler"]
    
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE} (CPU forced)")
    
    required_files = [TRAIN_FPATH, P_SCALER_FILE]
    for fpath in required_files:
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    # Check if validation data exists AND IS NOT EMPTY
    val_data_exists = False
    val_df = None
    if VAL_FPATH.exists():
        try:
            val_df = pd.read_parquet(VAL_FPATH)
            if len(val_df) > 0:
                val_data_exists = True
                print(f"\nValidation data found: YES ({len(val_df)} rows)")
            else:
                print(f"\nValidation data found but empty: will split from train")
                val_df = None
        except Exception as e:
            warnings.warn(f"\nError reading validation data: {e}. Will split from train.")
            val_df = None
    else:
        print(f"\nValidation data not found: will split from train")
    
    # Check if test data exists and is not empty
    test_data_exists = False
    test_df = None
    if TEST_FPATH.exists():
        try:
            test_df = pd.read_parquet(TEST_FPATH)
            if len(test_df) > 0:
                test_data_exists = True
                print(f"Test data found: YES ({len(test_df)} rows)")
            else:
                print(f"Test data found but empty: SKIPPING test evaluation")
                test_df = None
        except Exception as e:
            warnings.warn(f"Error reading test data: {e}. Skipping test evaluation.")
            test_df = None
    else:
        print(f"Test data not found: SKIPPING test evaluation")
    
    print(f"Validation data available: {'YES (separate file)' if val_data_exists else 'NO (will split from train)'}")
    print(f"Test data available: {'YES' if test_data_exists else 'NO'}")
    
    print("\nLoading data...")
    train_df = pd.read_parquet(TRAIN_FPATH)
    
    # Get data statistics
    train_stats = get_data_statistics(train_df, "Train")
    val_stats = None
    test_stats = None
    
    initial_feature_cols = [c for c in train_df.columns if c != target_col]    
    print(f"\nInitial features available: {initial_feature_cols}")    
    
    print("\n Creating training datasets (Overlapping with stride = 1)")
    print("-" * 100)
    
    if val_data_exists:
        val_stats = get_data_statistics(val_df, "Validation")
        print(f"Using separate validation dataset")
        print(f"Train range: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Val range: {val_df.index.min()} to {val_df.index.max()}")
        
        # TRAINING datasets (overlapping)
        train_ds = MultiStepDataset(train_df, initial_feature_cols, target_col, window, horizon,
                                   use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
        val_ds = MultiStepDataset(val_df, initial_feature_cols, target_col, window, horizon,
                                 use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
        
        print(f"\n[OVERLAPPING DATASETS FOR TRAINING]")
        print(f"Train sequences : {len(train_ds)} (stride=1)")
        print(f"Val sequences   : {len(val_ds)} (stride=1)")
        
    else:
        # Split validation from train using sequences
        print(f"\nCreating validation split from training data ({validation_split:.0%})")
        print(f"Train range: {train_df.index.min()} to {train_df.index.max()}")
        
        # Create base dataset from training data
        train_ds_base = MultiStepDataset(train_df, initial_feature_cols, target_col, window, horizon,
                                        use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
        
        # Extract sequences + timestamps
        X_seq, y_seq, bad_seq, ts_seq = [], [], [], []
        for i in range(len(train_ds_base)):
            x, y, bad = train_ds_base[i]
            X_seq.append(x.numpy())
            y_seq.append(y.numpy())
            bad_seq.append(bad.numpy())
            ts_seq.append(train_ds_base.get_timestamps_for_sample(i))

        X_seq = np.stack(X_seq)
        y_seq = np.stack(y_seq)
        bad_seq = np.stack(bad_seq)
        
        # Split train / val
        X_tr, X_va, y_tr, y_va, bad_tr, bad_va, ts_tr, ts_va = split_train_val_from_sequences(
            X_seq, y_seq, bad_seq, ts_seq,
            validation_split=validation_split,
            use_validation_set=True,
        )
        
        # Create custom dataset classes for split data
        class SplitDataset(Dataset):
            def __init__(self, X, y, bad, timestamps):
                self.X = X
                self.y = y
                self.bad = bad
                self.timestamps = timestamps
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.bad[idx]
            
            def get_timestamps_for_sample(self, idx):
                return self.timestamps[idx]
        
        # Create datasets
        train_ds = SplitDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), 
                               torch.from_numpy(bad_tr), ts_tr)
        val_ds = SplitDataset(torch.from_numpy(X_va), torch.from_numpy(y_va),
                             torch.from_numpy(bad_va), ts_va)
    
    # Create test dataset for TRAINING (overlapping) if available
    test_ds_train = None
    if test_data_exists:
        test_stats = get_data_statistics(test_df, "Test")
        print(f"\nTest range: {test_df.index.min()} to {test_df.index.max()}")
        
        test_ds_train = MultiStepDataset(test_df, initial_feature_cols, target_col, window, horizon,
                                        use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
        print(f"\nTest sequences: {len(test_ds_train)} (stride=1, for training evaluation)")
    
    print("\nCreating eval datasets (non-overlapping, stride = horizon)")
    print("-" * 100)
    
    # Create non-overlapping datasets for evaluation
    train_ds_eval = NonOverlappingMultiStepDataset(train_df, initial_feature_cols, target_col, window, horizon,
                                                  use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
    
    if val_data_exists:
        val_ds_eval = NonOverlappingMultiStepDataset(val_df, initial_feature_cols, target_col, window, horizon,
                                                    use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
    else:
        val_ds_eval = None
        print("Creating non-overlapping validation from train split...")
    
    test_ds_eval = None
    if test_data_exists:
        test_ds_eval = NonOverlappingMultiStepDataset(test_df, initial_feature_cols, target_col, window, horizon,
                                                     use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
        print(f"Test sequences (non-overlapping): {len(test_ds_eval)} (stride={horizon})")
    
    if hasattr(train_ds, 'get_feature_columns'):
        actual_feature_cols = train_ds.get_feature_columns()
    elif 'train_ds_base' in locals():
        actual_feature_cols = train_ds_base.get_feature_columns()
    else:
        actual_feature_cols = initial_feature_cols
        if use_bad_day and 'bad_day' in train_df.columns and 'bad_day' not in actual_feature_cols:
            actual_feature_cols = actual_feature_cols + ['bad_day']
    
    input_size = len(actual_feature_cols)
    
    print(f"\nActual features used (in feed order): {actual_feature_cols}")
    print(f"Input size: {input_size} features")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(random_seed)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    xb, yb, bmb = next(iter(train_loader))
    print("-" * 100)
    print("\n[DEBUG TRAINING LOADER]")
    print("xb:", xb.shape)   # (B, W+H, F)
    print("yb:", yb.shape)   # (B, H)
    print("bmb:", bmb.shape) # (B, H)
    print("-" * 100)

    print("\nCreating model...")
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
        forecast_steps=horizon
    )
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
        
    print("\nStarting training...")
    history = train_loop(model, train_loader, val_loader, epochs, lr, patience, 
                        DEVICE, MODEL_FILE)
    
    # Load best model
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("\n" + "="*60)
    print("MAKING PREDICTIONS FOR EVALUATION (NON-OVERLAPPING)")
    print("="*60)
    
    # Create dataloaders for evaluation (non-overlapping)
    train_loader_eval = DataLoader(
        train_ds_eval, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    if val_data_exists:
        val_loader_eval = DataLoader(
            val_ds_eval, 
            batch_size=batch_size, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        val_loader_eval = DataLoader(
            val_ds,  # Use original val_ds (already split properly)
            batch_size=batch_size, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        print("Using original validation split for evaluation (already properly split)")
    
    # Test dataloader for evaluation
    test_loader_eval = DataLoader(
        test_ds_eval, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    ) if test_ds_eval else None
    
    print("\nMaking predictions on NON-OVERLAPPING sequences...")
    preds_train_eval, trues_train_eval, bad_train_eval = predict_with_loader(model, train_loader_eval, DEVICE)
    preds_val_eval, trues_val_eval, bad_val_eval = predict_with_loader(model, val_loader_eval, DEVICE)
    
    # Test predictions (only if test exists)
    preds_test_eval, trues_test_eval, bad_test_eval = None, None, None
    if test_loader_eval:
        preds_test_eval, trues_test_eval, bad_test_eval = predict_with_loader(model, test_loader_eval, DEVICE)
        print(f"Test predictions (non-overlapping): {preds_test_eval.shape if preds_test_eval is not None else 'None'}")
    
    print(f"\nPrediction shapes (non-overlapping):")
    print(f"Train: {preds_train_eval.shape if preds_train_eval is not None else 'None'}")
    print(f"Val:   {preds_val_eval.shape if preds_val_eval is not None else 'None'}")
    if preds_test_eval is not None:
        print(f"Test:  {preds_test_eval.shape}")
    
    print("\nBuilding continuous series from NON-OVERLAPPING predictions...")
    # Build continuous series (normalized and denormalized)
    series_dict_train_norm, series_dict_train_denorm = build_continuous_series(
        train_ds_eval, preds_train_eval, trues_train_eval, bad_train_eval, 
        p_scaler, target_col=target_col, mask_bad_days=mask_bad_days  # Add target_col
    )

    if val_data_exists and val_ds_eval is not None:
        series_dict_val_norm, series_dict_val_denorm = build_continuous_series(
            val_ds_eval, preds_val_eval, trues_val_eval, bad_val_eval, 
            p_scaler, target_col=target_col, mask_bad_days=mask_bad_days  # Add target_col
        )
    else:
        series_dict_val_norm, series_dict_val_denorm = build_continuous_series(
            val_ds, preds_val_eval, trues_val_eval, bad_val_eval, 
            p_scaler, target_col=target_col, mask_bad_days=mask_bad_days  # Add target_col
        )

    if test_ds_eval and preds_test_eval is not None:
        series_dict_test_norm, series_dict_test_denorm = build_continuous_series(
            test_ds_eval, preds_test_eval, trues_test_eval, bad_test_eval, 
            p_scaler, target_col=target_col, mask_bad_days=mask_bad_days  # Add target_col
        )

    print(f"\nSeries lengths from CONTINUOUS SERIES:")
    train_true_series_norm, _, _ = series_dict_train_norm[target_col]
    val_true_series_norm, _, _ = series_dict_val_norm[target_col]
    print(f"Train: {len(train_true_series_norm.dropna())} valid points (from continuous series)")
    print(f"Val:   {len(val_true_series_norm.dropna())} valid points (from continuous series)")
    if series_dict_test_norm is not None:
        test_true_series_norm, _, _ = series_dict_test_norm[target_col]
        print(f"Test:  {len(test_true_series_norm.dropna())} valid points (from continuous series)")
    
    print("\nComputing metrics from CONTINUOUS NORMALIZED SERIES...")
    train_true_series_norm, train_pred_series_norm, train_bad_series = series_dict_train_norm[target_col]
    val_true_series_norm, val_pred_series_norm, val_bad_series = series_dict_val_norm[target_col]
    
    metrics_tr = compute_masked_metrics_with_bad_day(train_true_series_norm, train_pred_series_norm, train_bad_series)
    metrics_va = compute_masked_metrics_with_bad_day(val_true_series_norm, val_pred_series_norm, val_bad_series)
    metrics_te = None
    
    if series_dict_test_norm is not None:
        test_true_series_norm, test_pred_series_norm, test_bad_series = series_dict_test_norm[target_col]
        metrics_te = compute_masked_metrics_with_bad_day(test_true_series_norm, test_pred_series_norm, test_bad_series)
    
    # Save metrics
    with open(METRICS_LOG, "w") as f:
        f.write(f"LSTM Model\n")
        f.write(f"Window={window}, Horizon={horizon}\n")
        f.write(f"Model: hidden_size={hidden_size}, layers={num_layers}, dropout={dropout}\n")
        f.write(f"\nFlag Configuration:\n")
        f.write(f"  use_bad_day={use_bad_day}\n")
        f.write(f"  mask_bad_days={mask_bad_days}\n")
        
        if not use_bad_day and not mask_bad_days:
            f.write("  MODE 1: Don't feed bad_day as input & Don't mask bad days in loss\n")
        elif use_bad_day and mask_bad_days:
            f.write("  MODE 2: Feed bad_day as input & Mask bad days in loss\n")
        elif not use_bad_day and mask_bad_days:
            f.write("  MODE 3: Don't feed bad_day as input & Mask bad days in loss\n")
        elif use_bad_day and not mask_bad_days:
            f.write("  MODE 4: Feed bad_day as input & Don't mask bad days in loss\n")
        
        f.write(f"\nFeatures (in feed order):\n")
        for i, feat in enumerate(actual_feature_cols):
            f.write(f"  {i+1}. {feat}\n")
        
        f.write(f"\nTRAINING: Overlapping sequences (stride=1)\n")
        f.write(f"EVALUATION: Non-overlapping sequences (stride={horizon})\n")
        f.write(f"Metrics computed from: Continuous normalized series (10-min grid)\n")
        f.write(f"Validation: {'Separate dataset' if val_data_exists else f'Split from train ({validation_split:.0%})'}\n")
        f.write(f"Test data: {'Available' if test_data_exists else 'Not available'}\n")
        f.write(f"\nSequence counts:\n")
        f.write(f"  Train overlapping: {len(train_ds)}\n")
        f.write(f"  Train non-overlapping: {len(train_ds_eval)}\n")
        f.write(f"  Val overlapping: {len(val_ds)}\n")
        f.write(f"  Val non-overlapping: {len(val_ds_eval) if val_ds_eval else len(val_ds)}\n")
        if test_ds_train:
            f.write(f"  Test overlapping: {len(test_ds_train)}\n")
        if test_ds_eval:
            f.write(f"  Test non-overlapping: {len(test_ds_eval)}\n")
        f.write("\n")
        
        f.write("TRAIN METRICS (from continuous normalized series)\n")
        for k, v in metrics_tr.items():  # Direct access, not metrics_tr[target_col]
            f.write(f"  {k}: {v}\n")

        f.write("\nVAL METRICS (from continuous normalized series)\n")
        for k, v in metrics_va.items():  # Direct access, not metrics_va[target_col]
            f.write(f"  {k}: {v}\n")

        if metrics_te:
            f.write("\nTEST METRICS (from continuous normalized series)\n")
            for k, v in metrics_te.items():  # Direct access, not metrics_te[target_col]
                f.write(f"  {k}: {v}\n")
    
    print("\n=== METRICS (FROM CONTINUOUS NORMALIZED SERIES) ===")
    print("TRAIN:", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_tr.items()})
    print("VAL:  ", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_va.items()})
    if metrics_te:
        print("TEST: ", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_te.items()})
    
    # Save sample plots - Use DENORMALIZED series and the same random seed consistently
    save_daily_pred_vs_true_plots(train_df, series_dict_train_denorm, split_name="Train", plots_dir=PLOTS_DIR, random_seed=random_seed)
    
    if val_data_exists:
        val_df_for_plots = val_df
    else:
        val_df_for_plots = train_df
    save_daily_pred_vs_true_plots(val_df_for_plots, series_dict_val_denorm, split_name="Validation", plots_dir=PLOTS_DIR, random_seed=random_seed)
    
    if test_data_exists and series_dict_test_denorm is not None:
        save_daily_pred_vs_true_plots(test_df, series_dict_test_denorm, split_name="Test", plots_dir=PLOTS_DIR, random_seed=random_seed)
    
    # Build model info
    model_info = {
        "model_class": "LSTMModel",
        "architecture": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "output_size": 1,
            "forecast_steps": horizon,
            "activation": "ReLU (prevents negative predictions)",
            "lstm_batch_first": True,
            "total_params": total_params,
        },
        
        "data_config": {
            "features": {
                "feature_cols": actual_feature_cols,
                "target_col": target_col,
                "use_bad_day": use_bad_day,
                "mask_bad_days": mask_bad_days,
                "total_features": input_size,
            },
            "window_horizon": {
                "window": window,
                "horizon": horizon,
                "input_shape": f"({window + horizon}, {input_size})",
                "output_shape": f"({horizon},)",
            },
            "scaler_info": {
                "p_scaler_file": "p_scaling.pkl",
                "feature_scaler_file": "scalers.pkl",
                "p_scaler_type": "MinMaxScaler",
                "feature_scaler_types": "MinMaxScaler",
                "nan_handling": "nan_to_num(nan=0.0, copy=True)",
                "dtype": "float32",
            },
        },
        
        "training_config": {
            "hyperparameters": {
                "optimizer": "Adam",
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "grad_clip_max_norm": 1.0,
            },
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience": patience // 4
            },
            "loss_function": "masked_mae_with_bad_day",
            "dataloader": {
                "shuffle_train": False,
                "drop_last": False,
                "worker_init_fn": "seed_worker",
                "generator_seed": random_seed,
            },
            "random_seed": random_seed,
            "deterministic": True,
            "reproducibility": {
                "python_hash_seed": "set",
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "use_deterministic_algorithms": True,
            }
        },
        
        "dataset_info": {
            "window": window,
            "horizon": horizon,
            "use_bad_day": use_bad_day,
            "mask_bad_days": mask_bad_days,
            "training_strategy": "Overlapping sequences (stride=1) for training",
            "evaluation_strategy": f"Non-overlapping sequences (stride={horizon}) for evaluation",
            "sequence_calculation": f"max_idx = len(df) - (window={window} + horizon={horizon}) + 1",
            "input_timestamps": f"[i : i + window + horizon]",
            "target_timestamps": f"[i + window : i + window + horizon]",
        },
        
        "data_split_info": {
            "splits": {
                "train": {
                    "start": str(train_df.index.min()),
                    "end": str(train_df.index.max()),
                    "num_rows": len(train_df),
                    "num_sequences_overlapping": len(train_ds),
                    "num_sequences_non_overlapping": len(train_ds_eval),
                    "bad_day_stats": {
                        "bad_days": int(train_df['bad_day'].sum() if 'bad_day' in train_df.columns else 0),
                        "bad_days_pct": float((train_df['bad_day'].sum() / len(train_df) * 100) if 'bad_day' in train_df.columns else 0.0)
                    } if 'bad_day' in train_df.columns else None
                },
                "validation": {
                    "source": "separate_dataset" if val_data_exists else "split_from_train",
                    "start": str(val_df.index.min()) if val_data_exists and len(val_df) > 0 else None,
                    "end": str(val_df.index.max()) if val_data_exists and len(val_df) > 0 else None,
                    "num_rows": len(val_df) if val_data_exists else 0,
                    "num_sequences_overlapping": len(val_ds),
                    "num_sequences_non_overlapping": len(val_ds_eval) if val_ds_eval else len(val_ds),
                    "validation_split": validation_split if not val_data_exists else None,
                },
                "test": {
                    "available": test_data_exists,
                    "start": str(test_df.index.min()) if test_data_exists else None,
                    "end": str(test_df.index.max()) if test_data_exists else None,
                    "num_rows": len(test_df) if test_data_exists else 0,
                    "num_sequences_overlapping": len(test_ds_train) if test_ds_train else 0,
                    "num_sequences_non_overlapping": len(test_ds_eval) if test_ds_eval else 0,
                }
            },
            "total_sequences": {
                "train_overlapping": len(train_ds),
                "train_non_overlapping": len(train_ds_eval),
                "validation_overlapping": len(val_ds),
                "validation_non_overlapping": len(val_ds_eval) if val_ds_eval else len(val_ds),
                "test_overlapping": len(test_ds_train) if test_ds_train else 0,
                "test_non_overlapping": len(test_ds_eval) if test_ds_eval else 0,
            }
        },
        
        "training_principle": "LSTM forward pass always runs, PV learning masked on bad days",
        
        "flag_mode": {
            "use_bad_day": use_bad_day,
            "mask_bad_days": mask_bad_days,
            "description": "Mode 1" if not use_bad_day and not mask_bad_days else 
                          "Mode 2" if use_bad_day and mask_bad_days else
                          "Mode 3" if not use_bad_day and mask_bad_days else
                          "Mode 4"
        },
        
        "environment": {
            "device": str(DEVICE),
            "device_name": "CPU (forced)",
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": False,
        },
        
        "file_paths": {
            "training_data_dir": str(Path(training_data_dir).absolute()),
            "output_dir": str(Path(output_dir).absolute()) if output_dir else str(Path(training_data_dir).absolute()),
            "model_file": f"{timestamp}_model.pt",
            "scaler_dir": str(Path(training_data_dir).absolute()),
            "metrics_log": "training_metrics.txt",
            "plots_dir": "plots",
        },
        
        "metrics_info": {
            "computed_from": "continuous_normalized_series",
            "persistence_method": "lag_1_persistence",
            "series_continuity": "10-minute grid with NaN filling",
            "bad_day_filtering": f"{'Applied' if mask_bad_days else 'Not applied'}",
        }
    }
    
    # Save model info
    with open(OUT_DIR / "model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    
    import json
    with open(OUT_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"\nModel info saved to {OUT_DIR / 'model_info.pkl'}")
    print(f"Model info (readable) saved to {OUT_DIR / 'model_info.json'}")
    
    # Create PDF report
    out_pdf = OUT_DIR / "report.pdf"
    print("\nGenerating PDF report...")
    build_pdf_report(out_pdf, model_info, train_stats, val_stats, test_stats,
                len(train_ds_eval), 
                len(val_ds_eval) if val_ds_eval else len(val_ds),
                len(test_ds_eval) if test_ds_eval else 0,
                actual_feature_cols, 
                metrics_tr, metrics_va, metrics_te,
                history, 
                series_dict_train_denorm, series_dict_val_denorm, series_dict_test_denorm)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to {MODEL_FILE.name}")
    print(f"Metrics logged to {METRICS_LOG}")
    print(f"Model info saved to {OUT_DIR / 'model_info.pkl'}")
    print(f"PDF report saved to {out_pdf}")
    print(f"Plots saved to {PLOTS_DIR}")
    results = {
        "model": model,
        "metrics": {
            "train": metrics_tr,
            "val": metrics_va,
            "test": metrics_te,
        },
        "model_info": model_info,
        "series_normalized": {
            "train": series_dict_train_norm,
            "val": series_dict_val_norm,
            "test": series_dict_test_norm,
        },
        "series_denormalized": {
            "train": series_dict_train_denorm,
            "val": series_dict_val_denorm,
            "test": series_dict_test_denorm,
        },
        "history": history
    }

    return results

def main():
    training_data_dir = "/path/to/your/data"
    results = run_lstm_training(
        training_data_dir=training_data_dir,
        window=48,
        horizon=12,
        batch_size=32,
        epochs=100,
        lr=0.0001,
        patience=15,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        use_bad_day=True,
        mask_bad_days=True,
        validation_split=0.15,
    )
    return results

if __name__ == "__main__":
    main()