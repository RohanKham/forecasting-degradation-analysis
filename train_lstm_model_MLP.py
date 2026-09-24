#Version: 3.0
# Handles more than one output, with age (days_since_install_*) separated
# out of the LSTM's recurrent input and fed through its own small MLP head,
# combined with the LSTM's final hidden state per horizon step.
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.dates as mdates
import warnings
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import seaborn as sns

class AgeAwareMultiOutputLSTMModel(nn.Module):
    """
    LSTM over [window + horizon] steps of age-free features (past + known
    future covariates). For each horizon step, the LSTM's own hidden state
    at that step is combined with a per-step age embedding, and an MLP head
    maps the pair to num_outputs values.

    multiplicative_age=False (default):
        out = MLP([lstm_h ; age_embed])
        Age and weather/time-of-day can interact through the hidden layer.

    multiplicative_age=True:
        out = base(lstm_h) * (1 + gain(age_embed))
        Mirrors P = P_fresh * degradation(age). gain starts at exactly 0,
        so the model begins as a plain LSTM and learns the age effect on top.

    No activation on the output: the last layer is linear so gradients never
    die. Clamp to >= 0 at inference (see predict_with_loader).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_outputs,
        forecast_steps=1,
        dropout=0.2,
        age_num_features: int = 1,
        age_hidden: int = 8,
        age_embed_dim: int = 8,
        head_hidden: int = 32,
        multiplicative_age: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.forecast_steps = forecast_steps
        self.dropout = dropout
        self.age_num_features = age_num_features
        self.age_hidden = age_hidden
        self.age_embed_dim = age_embed_dim
        self.head_hidden = head_hidden
        self.multiplicative_age = multiplicative_age

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Age MLP: nonlinearity only BETWEEN linears, none at the end.
        self.age_head = nn.Sequential(
            nn.Linear(age_num_features, age_hidden),
            nn.ReLU(),
            nn.Linear(age_hidden, age_embed_dim),
        )

        if multiplicative_age:
            # base: LSTM state only
            self.base_head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, num_outputs),
            )
            # gain: age embedding only, zero-initialised -> gain = 1 at start
            self.gain_head = nn.Linear(age_embed_dim, num_outputs)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size + age_embed_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, num_outputs),
            )

    def forward(self, x, age):
        """
        x:   (B, window + horizon, input_size)  -- age-free features
        age: (B, horizon, age_num_features)
        returns: (B, horizon, num_outputs)
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))               # (B, W+H, hidden)

        # One state per forecast step. Each has seen the window plus the
        # known future covariates (Irr, hour/day encodings) up to that step.
        lstm_h = lstm_out[:, -self.forecast_steps:, :]      # (B, H, hidden)

        age_embed = self.age_head(age)                      # (B, H, age_embed_dim)

        if self.multiplicative_age:
            base = self.base_head(lstm_h)                   # (B, H, num_outputs)
            gain = 1.0 + self.gain_head(age_embed)          # (B, H, num_outputs)
            return base * gain

        combined = torch.cat([lstm_h, age_embed], dim=-1)   # (B, H, hidden + embed)
        return self.head(combined)                          # (B, H, num_outputs)

def detect_bad_day_columns(df: pd.DataFrame, target_cols: List[str]) -> Dict[str, str]:
    """
    Automatically detect which bad_day_* column corresponds to each target.
    Supports prefixes: P_, I_, U_
    """
    bad_day_mapping = {}
    all_bad_day_cols = [col for col in df.columns if col.startswith('bad_day_')]

    print(f"\nDetecting bad_day columns for targets: {target_cols}")
    print(f"Available bad_day columns: {all_bad_day_cols}")

    for target in target_cols:
        if '_' in target:
            patterns = []

            known_prefixes = ['P_', 'I_', 'U_']
            prefix_found = None
            for prefix in known_prefixes:
                if target.startswith(prefix):
                    prefix_found = prefix
                    break

            if prefix_found:
                parts = target.split('_')
                if len(parts) > 1:
                    patterns.append(f"bad_day_{parts[-1]}")

                suffix = target[len(prefix_found):]
                suffix_parts = suffix.split('_')

                patterns.append(f"bad_day_{suffix_parts[-1]}")
                if len(suffix_parts) > 1:
                    patterns.append(f"bad_day_{'_'.join(suffix_parts)}")
                    patterns.append(f"bad_day_{'_'.join(suffix_parts[1:])}")
            else:
                parts = target.split('_')
                patterns.append(f"bad_day_{parts[-1]}")
                patterns.append(f"bad_day_{'_'.join(parts)}")

            found = False
            for pattern in patterns:
                if pattern in all_bad_day_cols:
                    bad_day_mapping[target] = pattern
                    print(f"  {target} : {pattern}")
                    found = True
                    break

            if not found:
                print(f"WARNING: No bad_day column found for target '{target}'")
                if all_bad_day_cols:
                    bad_day_mapping[target] = all_bad_day_cols[0]
                    print(f"  Using fallback: {target} : {all_bad_day_cols[0]}")
                else:
                    bad_day_mapping[target] = None
        else:
            print(f"WARNING: Target '{target}' doesn't follow expected naming pattern")
            if all_bad_day_cols:
                bad_day_mapping[target] = all_bad_day_cols[0]
            else:
                bad_day_mapping[target] = None

    return bad_day_mapping


def _resolve_feature_and_age_cols(
    df: pd.DataFrame,
    feature_cols: List[str],
    age_cols: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Splits the candidate feature list into (recurrent_feature_cols, age_cols).

    This is where age/feature separation is actually enforced: age_cols are
    removed from feature_cols if present, so the caller doesn't have to
    remember to pre-filter them out -- passing the full original feature
    list plus age_cols is enough, the dataset classes handle the split
    themselves.

    If age_cols is None or empty, this is a no-op (age_cols -> []), so old
    call sites that don't pass age_cols still work, just with no separate
    age head input required (the AgeAware* classes still expect an `age`
    tensor at __getitem__ time in that case it will just be empty-width).
    """
    age_cols = list(age_cols) if age_cols else []

    missing_age = [c for c in age_cols if c not in df.columns]
    if missing_age:
        raise ValueError(f"age_cols not found in dataframe: {missing_age}")

    resolved_feature_cols = [c for c in feature_cols if c not in age_cols]

    removed = [c for c in feature_cols if c in age_cols]
    if removed:
        print(f"  Removed from recurrent feature_cols (moved to age head): {removed}")

    return resolved_feature_cols, age_cols


class AgeAwareMultiOutputDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        target_cols: List[str],
        age_cols: List[str],
        window: int,
        horizon: int,
        use_bad_day: bool = True,
        mask_bad_days: bool = True,
        bad_day_mapping: Dict[str, str] = None,
    ):
        """
        Creates sliding window samples for time series | stride = 1.
        window: Lookback period (e.g., 48 timesteps)
        Input X includes: [window + horizon] timesteps of RECURRENT features
        (age_cols excluded from X, see age_cols below).
        age: [horizon] timesteps of age feature(s), fed to a separate head.
        Targets y includes: only [horizon] timesteps of target.
        horizon: Forecast horizon (e.g., 36 timesteps)
        bad_day: Binary masks for low-quality data.

        age_cols: columns (e.g. 'days_since_install_P_perovskite_1') that are
        removed from the recurrent input and instead sliced over just the
        horizon portion, for use by a separate age head in the model. Pass
        the full candidate feature_cols list including age_cols -- they are
        split out automatically.
        """
        self.window = window
        self.horizon = horizon
        self.use_bad_day = use_bad_day
        self.mask_bad_days = mask_bad_days
        self.target_cols = target_cols
        self.num_outputs = len(target_cols)
        self.original_feature_cols = feature_cols.copy()

        if bad_day_mapping is None:
            self.bad_day_mapping = detect_bad_day_columns(df, target_cols)
        else:
            self.bad_day_mapping = bad_day_mapping

        # Create actual feature columns based on use_bad_day flag
        if use_bad_day and self.bad_day_mapping:
            unique_bad_cols = set(self.bad_day_mapping.values())
            unique_bad_cols = [col for col in unique_bad_cols if col is not None]

            for col in unique_bad_cols:
                if col not in feature_cols and col in df.columns:
                    feature_cols = feature_cols + [col]

            feature_cols = feature_cols
            print(f"Added bad_day columns to features: {unique_bad_cols}")
        elif use_bad_day and not self.bad_day_mapping:
            print(f"WARNING: use_bad_day=True but no bad_day mapping found")
        else:
            feature_cols = [col for col in feature_cols if not col.startswith('bad_day_')]

        # Split age_cols out of the recurrent feature set.
        self.feature_cols, self.age_cols = _resolve_feature_and_age_cols(df, feature_cols, age_cols)

        # Get feature data (recurrent, age-free)
        self.X = df[self.feature_cols].to_numpy(np.float32)

        # Get age data (horizon-only, fed to the age head)
        self.age = df[self.age_cols].to_numpy(np.float32) if self.age_cols else np.zeros((len(df), 0), dtype=np.float32)

        # Get all target data
        self.y = df[target_cols].to_numpy(np.float32)  # (N, num_outputs)
        self.ts = df.index

        # Prepare bad_day arrays for each target
        self.bad_day_arrays = {}
        if self.bad_day_mapping:
            for target_idx, target_name in enumerate(target_cols):
                bad_col = self.bad_day_mapping.get(target_name)
                if bad_col and bad_col in df.columns:
                    self.bad_day_arrays[target_idx] = df[bad_col].to_numpy(np.float32)
                    print(f"Target {target_name} will use {bad_col} for masking")
                else:
                    print(f"WARNING: No bad_day column for target '{target_name}', using all good days")
                    self.bad_day_arrays[target_idx] = np.zeros(len(df), dtype=np.float32)
        else:
            for target_idx in range(self.num_outputs):
                self.bad_day_arrays[target_idx] = np.zeros(len(df), dtype=np.float32)

        max_idx = len(df) - (window + horizon) + 1
        self.valid_indices = list(range(max_idx))
        self.n = len(self.valid_indices)

        print(f"\nAgeAwareMultiOutputDataset Configuration:")
        print(f"use_bad_day={use_bad_day}, mask_bad_days={mask_bad_days}")
        print(f"Recurrent features used: {self.feature_cols}")
        print(f"Age columns (separate head): {self.age_cols}")
        print(f"Targets: {target_cols}")
        print(f"Input shape per sample: ({window + horizon}, {len(self.feature_cols)})")
        print(f"Age shape per sample: ({horizon}, {len(self.age_cols)})")
        print(f"Target shape per sample: ({horizon}, {self.num_outputs})")
        print(f"Total samples: {self.n}")

        # Calculate statistics for bad days in horizon
        bad_day_stats = {}
        for target_idx in range(self.num_outputs):
            bad_counts = []
            for idx in self.valid_indices:
                bad_in_horizon = self.bad_day_arrays[target_idx][idx + window: idx + window + horizon]
                bad_counts.append(np.sum(bad_in_horizon > 0.5))

            total_bad = sum(bad_counts)
            target_name = target_cols[target_idx]
            bad_day_stats[target_name] = total_bad

        for target_name, total_bad in bad_day_stats.items():
            print(f"  {target_name}: {total_bad} bad horizon periods "
                  f"({total_bad / (self.n * horizon) * 100:.1f}% of steps)")

        # Debug: first sample alignment
        i = self.valid_indices[0]
        print("\n[DEBUG AGE-AWARE MULTI-OUTPUT DATASET]")
        print(f"X covers: {self.ts[i]} -> {self.ts[i + self.window + self.horizon - 1]}")
        print(f"age covers: {self.ts[i + self.window]} -> {self.ts[i + self.window + self.horizon - 1]}")
        print(f"y covers: {self.ts[i + self.window]} -> {self.ts[i + self.window + self.horizon - 1]}")
        print("First 5 y values (all targets):")
        for j, col in enumerate(target_cols):
            print(f"  {col}: {self.y[i + self.window : i + self.window + 5, j]}")
        if self.age_cols:
            print("First 5 age values (all age cols):")
            for j, col in enumerate(self.age_cols):
                print(f"  {col}: {self.age[i + self.window : i + self.window + 5, j]}")
        print("First 5 bad_day flags per target:")
        for j, col in enumerate(target_cols):
            bad_vals = self.bad_day_arrays[j][i + self.window : i + self.window + 5]
            print(f"  {col}: {bad_vals}")
        print("\n" + "-" * 100)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Input includes window + horizon timesteps, age-free
        x = self.X[i : i + self.window + self.horizon].copy()  # (window+horizon, features)

        # Age is a horizon-length sequence only -- no recurrence in the age
        # head, so no window portion is needed.
        age = self.age[i + self.window : i + self.window + self.horizon].copy()  # (horizon, age_num_features)

        y = self.y[i + self.window : i + self.window + self.horizon].copy()  # (horizon, num_outputs)

        bad_day_masks = []
        for target_idx in range(self.num_outputs):
            bad_mask = self.bad_day_arrays[target_idx][i + self.window : i + self.window + self.horizon].copy()
            bad_day_masks.append(bad_mask)

        bad_day_mask = np.stack(bad_day_masks, axis=-1) if bad_day_masks else np.zeros((self.horizon, self.num_outputs))

        # Also mask NaN positions in y
        nan_mask = np.isnan(y)  # (horizon, num_outputs)
        bad_day_mask = np.where(nan_mask, 1.0, bad_day_mask)

        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        age = np.nan_to_num(age, nan=0.0).astype(np.float32)

        if not self.mask_bad_days:
            bad_day_mask = np.zeros_like(bad_day_mask, dtype=np.float32)
        else:
            bad_day_mask = bad_day_mask.astype(np.float32)

        return (
            torch.from_numpy(x),             # (window+horizon, features) -- age-free
            torch.from_numpy(age),           # (horizon, age_num_features)
            torch.from_numpy(y),             # (horizon, num_outputs)
            torch.from_numpy(bad_day_mask),  # (horizon, num_outputs)
        )

    def get_timestamps_for_sample(self, idx):
        i = self.valid_indices[idx]
        return self.ts[i + self.window : i + self.window + self.horizon]

    def get_feature_columns(self):
        """Return the actual recurrent feature columns used in the dataset (age-free)."""
        return self.feature_cols

    def get_age_columns(self):
        """Return the age columns fed to the separate age head."""
        return self.age_cols

    def get_target_columns(self):
        """Return the target columns"""
        return self.target_cols

    def get_bad_day_mapping(self):
        """Return the bad_day mapping dictionary"""
        return self.bad_day_mapping


class AgeAwareNonOverlappingMultiOutputDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        target_cols: List[str],
        age_cols: List[str],
        window: int,
        horizon: int,
        use_bad_day: bool = True,
        mask_bad_days: bool = True,
        bad_day_mapping: Dict[str, str] = None,
    ):
        """
        Creates NON-OVERLAPPING sliding horizon samples for multi-output time
        series with target-specific bad day masking, and age separated out
        of the recurrent input (see AgeAwareMultiOutputDataset docstring).
        """
        self.window = window
        self.horizon = horizon
        self.use_bad_day = use_bad_day
        self.mask_bad_days = mask_bad_days
        self.target_cols = target_cols
        self.num_outputs = len(target_cols)
        self.original_feature_cols = feature_cols.copy()

        if bad_day_mapping is None:
            self.bad_day_mapping = detect_bad_day_columns(df, target_cols)
        else:
            self.bad_day_mapping = bad_day_mapping

        if use_bad_day and self.bad_day_mapping:
            unique_bad_cols = set(self.bad_day_mapping.values())
            unique_bad_cols = [col for col in unique_bad_cols if col is not None]

            for col in unique_bad_cols:
                if col not in feature_cols and col in df.columns:
                    feature_cols = feature_cols + [col]

            feature_cols = feature_cols
        elif use_bad_day and not self.bad_day_mapping:
            pass
        else:
            feature_cols = [col for col in feature_cols if not col.startswith('bad_day_')]

        self.feature_cols, self.age_cols = _resolve_feature_and_age_cols(df, feature_cols, age_cols)

        self.X = df[self.feature_cols].to_numpy(np.float32)
        self.age = df[self.age_cols].to_numpy(np.float32) if self.age_cols else np.zeros((len(df), 0), dtype=np.float32)
        self.y = df[target_cols].to_numpy(np.float32)
        self.ts = df.index

        self.bad_day_arrays = {}
        if self.bad_day_mapping:
            for target_idx, target_name in enumerate(target_cols):
                bad_col = self.bad_day_mapping.get(target_name)
                if bad_col and bad_col in df.columns:
                    self.bad_day_arrays[target_idx] = df[bad_col].to_numpy(np.float32)
                else:
                    self.bad_day_arrays[target_idx] = np.zeros(len(df), dtype=np.float32)
        else:
            for target_idx in range(self.num_outputs):
                self.bad_day_arrays[target_idx] = np.zeros(len(df), dtype=np.float32)

        total_length = len(df)
        stride = horizon
        max_idx = total_length - (window + horizon) + 1

        self.valid_indices = []
        i = 0
        while i < max_idx:
            self.valid_indices.append(i)
            i += stride

        self.n = len(self.valid_indices)

        print(f"AgeAwareNonOverlappingMultiOutputDataset Configuration (stride={horizon}):")
        print(f"use_bad_day={use_bad_day}, mask_bad_days={mask_bad_days}")
        print(f"Recurrent features used: {self.feature_cols}")
        print(f"Age columns (separate head): {self.age_cols}")
        print(f"Targets: {target_cols}")
        print(f"Input shape per sample: ({window + horizon}, {len(self.feature_cols)})")
        print(f"Age shape per sample: ({horizon}, {len(self.age_cols)})")
        print(f"Target shape per sample: ({horizon}, {self.num_outputs})")
        print(f"Total non-overlapping samples: {self.n}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.X[i : i + self.window + self.horizon].copy()
        age = self.age[i + self.window : i + self.window + self.horizon].copy()
        y = self.y[i + self.window : i + self.window + self.horizon].copy()

        bad_day_masks = []
        for target_idx in range(self.num_outputs):
            bad_mask = self.bad_day_arrays[target_idx][
                i + self.window : i + self.window + self.horizon
            ].copy()
            bad_day_masks.append(bad_mask)

        bad_day_mask = np.stack(bad_day_masks, axis=-1) if bad_day_masks else np.zeros((self.horizon, self.num_outputs))

        nan_mask = np.isnan(y)  # (horizon, num_outputs)
        bad_day_mask = np.where(nan_mask, 1.0, bad_day_mask)

        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        age = np.nan_to_num(age, nan=0.0).astype(np.float32)

        if not self.mask_bad_days:
            bad_day_mask = np.zeros_like(bad_day_mask, dtype=np.float32)
        else:
            bad_day_mask = bad_day_mask.astype(np.float32)

        return (
            torch.from_numpy(x),             # (window+horizon, features) -- age-free
            torch.from_numpy(age),           # (horizon, age_num_features)
            torch.from_numpy(y),             # (horizon, num_outputs)
            torch.from_numpy(bad_day_mask),  # (horizon, num_outputs)
        )

    def get_timestamps_for_sample(self, idx):
        i = self.valid_indices[idx]
        return self.ts[i + self.window : i + self.window + self.horizon]

    def get_feature_columns(self):
        return self.feature_cols

    def get_age_columns(self):
        return self.age_cols

    def get_target_columns(self):
        return self.target_cols

    def get_valid_indices(self):
        return self.valid_indices


def split_train_val_from_sequences(
    X_seq,
    age_seq,
    y_seq,
    bad_seq,
    ts_seq=None,
    validation_split: float = 0.1,
    use_validation_set: bool = True,
):
    """
    Split TRAIN sequences into train / val using modulo logic.
    Now also splits the age sequence array alongside X/y/bad.
    """
    if not use_validation_set or validation_split <= 0:
        if ts_seq is not None:
            return X_seq, None, age_seq, None, y_seq, None, bad_seq, None, ts_seq, None
        else:
            return X_seq, None, age_seq, None, y_seq, None, bad_seq, None

    n = round(1 / validation_split)
    idx = np.arange(len(X_seq))

    val_mask = (idx % n) == 0
    train_mask = ~val_mask

    X_train = X_seq[train_mask]
    age_train = age_seq[train_mask]
    y_train = y_seq[train_mask]
    bad_train = bad_seq[train_mask]
    X_val = X_seq[val_mask]
    age_val = age_seq[val_mask]
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

    return X_train, X_val, age_train, age_val, y_train, y_val, bad_train, bad_val, ts_train, ts_val


def masked_mae_with_multiple_bad_days(pred, true, bad_day_mask):
    """
    Compute MAE with target-specific bad day masking.

    Args:
        pred: (B, H, num_outputs) predictions
        true: (B, H, num_outputs) ground truth
        bad_day_mask: (B, H, num_outputs) bad_day mask (1=bad_day, 0=good_day)

    Returns:
        Scalar MAE loss computed only on good days across all outputs
    """
    good_mask = 1.0 - bad_day_mask  # (B, H, num_outputs)

    diff = torch.abs(pred - true)
    masked_diff = diff * good_mask

    total_good_elements = torch.sum(good_mask) + 1e-8

    return torch.sum(masked_diff) / total_good_elements


def compute_masked_metrics_with_bad_day_per_target(
    y_true,
    y_pred,
    bad_day_mask,
    target_idx: int,
    target_name: str,
):
    """
    Calculate metrics for a single target output from CONTINUOUS series.
    Uses target-specific bad day mask.
    """
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    bad_flat = np.asarray(bad_day_mask).flatten()

    y_true_flat = y_true_flat.astype(float)
    y_pred_flat = y_pred_flat.astype(float)
    bad_flat = bad_flat.astype(float)

    mask = (~(np.isnan(y_true_flat) | np.isnan(y_pred_flat)) & (bad_flat < 0.5))

    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]

    if len(y_true_clean) == 0:
        return {
            "target": target_name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "Directional_Accuracy": np.nan,
            "Skill_Score": np.nan,
            "Peak_Error": np.nan,
            "Data_Points": 0,
            "Bad_Day_Excluded": 0,
            "Total_Points": 0,
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
        "target": target_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Skill_Score": skill_score,
        "Directional_Accuracy": directional_accuracy,
        "Peak_Error": peak_error,
        "Data_Points": len(y_true_clean),
        "Bad_Day_Excluded": int(np.sum(bad_flat > 0.5)),
        "Total_Points": len(y_true_flat),
    }


def compute_metrics_from_continuous_series(
    series_dict_norm: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    target_cols: List[str],
    horizon: int = 36
) -> Dict[str, Dict]:
    """
    Compute metrics for all targets from continuous normalized series.
    Each target uses its own bad day series.
    """
    all_metrics = {}

    for idx, target_name in enumerate(target_cols):
        true_series, pred_series, bad_series = series_dict_norm[target_name]

        common_index = true_series.index.intersection(pred_series.index).intersection(bad_series.index)

        y_true = true_series.loc[common_index].values
        y_pred = pred_series.loc[common_index].values
        bad_mask = bad_series.loc[common_index].values

        metrics = compute_masked_metrics_with_bad_day_per_target(y_true, y_pred, bad_mask, idx, target_name)
        all_metrics[target_name] = metrics

    return all_metrics


def predict_with_loader(model, loader, device):
    """
    Make predictions with the age-aware multi-output model.
    Returns predictions for all targets with target-specific bad day masks.
    """
    if loader is None:
        return None, None, None

    model.eval()
    all_preds = []
    all_trues = []
    all_bad_days = []

    with torch.no_grad():
        for xb, age_b, yb, bmb in loader:
            xb = xb.to(device)
            age_b = age_b.to(device)

            #preds = model(xb, age_b).cpu().numpy()  # (B, H, num_outputs)
            preds = model(xb, age_b).clamp(min=0).cpu().numpy()

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


def build_continuous_series_multi_output(dataset, preds, trues, bad_days, scaler_dict,
                                          target_cols, mask_bad_days: bool = True,
                                          expected_freq: str = "10min"):
    """
    Build continuous normalized and denormalized time series for each target output
    with target-specific bad day masking.

    Returns:
        results_norm, results_denorm:
        Dict[target] = (true_series, pred_series, bad_day_series_for_this_target)
    """
    if preds is None or trues is None or bad_days is None or len(preds) == 0:
        empty = pd.Series(dtype=float)
        return (
            {col: (empty.copy(), empty.copy(), empty.copy()) for col in target_cols},
            {col: (empty.copy(), empty.copy(), empty.copy()) for col in target_cols}
        )

    num_outputs = len(target_cols)
    results_norm = {}
    results_denorm = {}

    print("\n" + "=" * 80)
    print("BUILDING CONTINUOUS SERIES WITH TARGET-SPECIFIC BAD DAY MASKING")
    print("=" * 80)

    print(f"Preds shape     : {preds.shape}")
    print(f"Trues shape     : {trues.shape}")
    print(f"Bad_days shape  : {bad_days.shape}")
    print(f"Dataset length : {len(dataset)}")
    print(f"Horizon         : {preds.shape[1]}")
    print(f"Targets         : {num_outputs}")
    print(f"Mask bad days   : {mask_bad_days}")

    for idx, target_name in enumerate(target_cols):
        print(f"\n--- Processing target: {target_name} [{idx}] ---")

        scaler = scaler_dict[target_name]

        target_pred_norm = preds[:, :, idx]  # (n_samples, horizon) - NORMALIZED
        target_true_norm = trues[:, :, idx]  # (n_samples, horizon) - NORMALIZED
        target_bad_days = bad_days[:, :, idx]  # (n_samples, horizon) - TARGET-SPECIFIC

        target_pred_denorm = scaler.inverse_transform(target_pred_norm.reshape(-1, 1)).reshape(target_pred_norm.shape)
        target_true_denorm = scaler.inverse_transform(target_true_norm.reshape(-1, 1)).reshape(target_true_norm.shape)

        print(f"Denormalized predictions range: [{target_pred_denorm.min():.4f}, {target_pred_denorm.max():.4f}]")
        print(f"Denormalized truths range: [{target_true_denorm.min():.4f}, {target_true_denorm.max():.4f}]")

        n_samples = min(len(dataset), target_pred_norm.shape[0], target_true_norm.shape[0], target_bad_days.shape[0])

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

                bad_val = target_bad_days[i, j]

                if ts not in bad_map:
                    bad_map[ts] = bad_val

                if mask_bad_days and bad_val > 0.5:
                    continue

                if ts not in pred_map_norm:
                    pred_map_norm[ts] = target_pred_norm[i, j]
                    true_map_norm[ts] = target_true_norm[i, j]
                    pred_map_denorm[ts] = target_pred_denorm[i, j]
                    true_map_denorm[ts] = target_true_denorm[i, j]

        if ts_count:
            ts_min = min(ts_count.keys())
            ts_max = max(ts_count.keys())
        else:
            empty_series = pd.Series(dtype=float)
            results_norm[target_name] = (empty_series.copy(), empty_series.copy(), empty_series.copy())
            results_denorm[target_name] = (empty_series.copy(), empty_series.copy(), empty_series.copy())
            continue

        full_index = pd.date_range(
            start=ts_min,
            end=ts_max,
            freq=expected_freq
        )

        print(f"Expected full grid size    : {len(full_index)}")
        print(f"Actual predicted timestamps: {len(pred_map_norm)}")
        print(f"Missing timestamps (NaNs) : {len(full_index) - len(pred_map_norm)}")

        pred_series_norm = pd.Series(pred_map_norm, name=f"{target_name}_pred_norm")
        true_series_norm = pd.Series(true_map_norm, name=f"{target_name}_true_norm")
        pred_series_denorm = pd.Series(pred_map_denorm, name=f"{target_name}_pred")
        true_series_denorm = pd.Series(true_map_denorm, name=f"{target_name}_true")
        bad_series = pd.Series(bad_map, name=f"bad_day_{target_name}")

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

        results_norm[target_name] = (
            true_series_norm,
            pred_series_norm,
            bad_series
        )

        results_denorm[target_name] = (
            true_series_denorm,
            pred_series_denorm,
            bad_series
        )

    print("=" * 80 + "\n")
    return results_norm, results_denorm


def train_loop(model, train_loader, val_loader, epochs, lr, patience, device, model_file):
    """
    Train the age-aware multi-output LSTM with target-specific bad day masking.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 4
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch_idx, (xb, age_b, yb, bmb) in enumerate(train_loader):
            xb = xb.to(device)
            age_b = age_b.to(device)
            yb = yb.to(device)  # (B, H, num_outputs)
            bmb = bmb.to(device)  # (B, H, num_outputs)

            optimizer.zero_grad()
            pred = model(xb, age_b)  # (B, H, num_outputs)

            loss = masked_mae_with_multiple_bad_days(pred, yb, bmb)

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
                print("\n[DEBUG MODEL]")
                print("Input x:", xb.shape)
                print("Input age:", age_b.shape)
                print("Pred:", pred.shape)
                print("Target:", yb.shape)
                print("Bad mask:", bmb.shape)
                print("-" * 100)

        avg_train_loss = np.mean(train_losses) if train_losses else float("inf")
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, age_b, yb, bmb in val_loader:
                xb = xb.to(device)
                age_b = age_b.to(device)
                yb = yb.to(device)
                bmb = bmb.to(device)
                pred = model(xb, age_b)
                loss = masked_mae_with_multiple_bad_days(pred, yb, bmb)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["val_loss"].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val - 1e-6:
            best_val = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), model_file)
            print(f"  Saved model (val loss: {avg_val_loss:.6f})")
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


# ---------------- Plotting helpers ----------------
def save_scatter_ax(ax, y_true, y_pred, bad_series, title, mask_bad_days=True):
    """Create scatter plot, ignoring NaNs and target-specific bad days."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    bad = bad_series.values.flatten() if bad_series is not None else None

    nan_mask = np.isnan(yt) | np.isnan(yp)

    if mask_bad_days and bad is not None:
        bad_mask = bad >= 0.5
    else:
        bad_mask = np.zeros_like(nan_mask, dtype=bool)

    valid_mask = (~nan_mask) & (~bad_mask)

    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]

    if len(yt_valid) == 0:
        ax.text(0.5, 0.5, "No valid data to plot",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return

    ax.scatter(yt_valid, yp_valid, s=2, alpha=0.5)

    lims = [
        min(yt_valid.min(), yp_valid.min()),
        max(yt_valid.max(), yp_valid.max())
    ]

    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1)
    ax.set_xlabel("Actual (normalized)")
    ax.set_ylabel("Predicted (normalized)")
    ax.set_title(f"{title} (n={len(yt_valid)})")
    ax.grid(True, alpha=0.3)


def save_training_history_ax(ax, history):
    """Plot training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], '-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history["val_loss"], '-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE Loss')
    ax.set_title('Training History (All Targets Combined)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if epochs:
        ax.text(0.98, 0.98, f"Final Train: {history['train_loss'][-1]:.4f}\nFinal Val: {history['val_loss'][-1]:.4f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_time_series_ax(ax, true_series, pred_series, bad_series, title, mask_bad_days=True):
    """Plot time series excluding NaNs and target-specific bad days."""
    if true_series.empty or pred_series.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return

    yt = true_series.values
    yp = pred_series.reindex(true_series.index).values
    bad = bad_series.values if bad_series is not None else None

    nan_mask = np.isnan(yt) | np.isnan(yp)

    if mask_bad_days and bad is not None:
        bad_mask = bad >= 0.5
    else:
        bad_mask = np.zeros_like(nan_mask, dtype=bool)

    valid_mask = (~nan_mask) & (~bad_mask)

    if not np.any(valid_mask):
        ax.text(0.5, 0.5, "No valid data",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    times = true_series.index[valid_mask]
    y_true = yt[valid_mask]
    y_pred = yp[valid_mask]

    ax.plot(times, y_true, label="Actual", linewidth=1.5, alpha=0.85)
    ax.plot(times, y_pred, "--", label="Predicted", linewidth=1.2, alpha=0.85)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ax.text(0.02, 0.98,
            f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}",
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Value")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def save_error_histogram_ax(ax, y_true, y_pred, bad_series, title, bins=60, mask_bad_days=True):
    """Plot error distribution histogram with target-specific bad day exclusion."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    bad = bad_series.values.flatten() if bad_series is not None else None

    nan_mask = np.isnan(yt) | np.isnan(yp)

    if mask_bad_days and bad is not None:
        bad_mask = bad >= 0.5
    else:
        bad_mask = np.zeros_like(nan_mask, dtype=bool)

    valid_mask = (~nan_mask) & (~bad_mask)

    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]

    if len(yt_valid) == 0:
        ax.text(0.5, 0.5, "No valid data",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return

    errors = yp_valid - yt_valid
    mean_err = np.mean(errors)

    ax.hist(errors, bins=bins, density=False, alpha=0.7)

    ax.axvline(0.0, linestyle='--', linewidth=1.5, color='red', label='Zero error')
    ax.axvline(mean_err, linestyle='-', linewidth=1.5, color='green', label=f'Mean = {mean_err:.4f}')

    ax.set_title(title)
    ax.set_xlabel("Prediction Error (Pred \u2212 True)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.25)


def save_daily_pred_vs_true_plots_multi_output(
    df: pd.DataFrame,
    series_dict: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    split_name: str,
    plots_dir: Path,
    n_days: int = 10,
    irr_col: str = "Irr",
    min_points_per_day: int = 45,
    random_seed: int | None = 42,
):
    """
    Save daily prediction vs true plots for multiple targets
    with target-specific bad day exclusion.
    """
    if random_seed is not None:
        random.seed(random_seed)

    df = df.copy()

    for target_name, (true_series, pred_series, bad_series) in series_dict.items():
        df = df.join(true_series.rename(f"{target_name}_true"), how="left")
        df = df.join(pred_series.rename(f"{target_name}_pred"), how="left")
        df = df.join(bad_series.rename(f"{target_name}_bad"), how="left")

    df["date"] = df.index.date

    valid_days = []
    target_cols = list(series_dict.keys())

    for day, day_df in df.groupby("date"):
        bad_cols = [f"{t}_bad" for t in target_cols]
        has_bad_day = False
        for col in bad_cols:
            if col in day_df.columns and (day_df[col] == 1).any():
                has_bad_day = True
                break

        if has_bad_day:
            continue

        required_cols = []
        for t in target_cols:
            required_cols.extend([f"{t}_true", f"{t}_pred"])
        required_cols.append(irr_col)

        day_df_valid = day_df.dropna(subset=required_cols)

        if day_df_valid.empty or len(day_df_valid) < min_points_per_day:
            continue

        valid_days.append(day)

    if len(valid_days) == 0:
        print(f"[WARNING] No valid complete days found for {split_name}")
        return

    n_select = min(n_days, len(valid_days))
    selected_days = random.sample(valid_days, n_select)

    print(f"[INFO] Selected {n_select} random complete days for plotting ({split_name})")

    for day in selected_days:
        day_df = df[df["date"] == day]

        num_targets = len(target_cols)
        fig, axes = plt.subplots(num_targets, 1, figsize=(10, 4 * num_targets))
        if num_targets == 1:
            axes = [axes]

        times = day_df.index
        month = times[0].to_period("M")

        for idx, target_name in enumerate(target_cols):
            ax1 = axes[idx]

            true_col = f"{target_name}_true"
            pred_col = f"{target_name}_pred"

            day_df_plot = day_df.dropna(subset=[true_col, pred_col, irr_col])

            if len(day_df_plot) == 0:
                continue

            plot_times = day_df_plot.index

            ax1.plot(
                plot_times,
                day_df_plot[true_col],
                label=f"{target_name} True",
                linewidth=2,
                marker="o",
                markersize=3,
            )
            ax1.plot(
                plot_times,
                day_df_plot[pred_col],
                label=f"{target_name} Pred",
                linestyle="--",
                marker="x",
                markersize=4,
            )

            ax1.set_xlabel("Time (hour)")
            ax1.set_ylabel(f"{target_name} (denormalized)")
            ax1.set_title(f"{split_name} | {month} | {day} | {target_name}")
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            fig.autofmt_xdate()

            ax2 = ax1.twinx()
            ax2.plot(plot_times, day_df_plot[irr_col], label="Irr (scaled)", alpha=0.6, color='orange')
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
        fname = plots_dir / f"{split_name.lower()}_{day}_pred_vs_true_multi.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def save_correlation_heatmaps(
    df_test: pd.DataFrame,
    series_dict_test: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    target_cols: List[str],
    plots_dir: Path,
):
    """
    Create correlation heatmap between target actual/pred values and all input features.
    Uses only TEST data. Excludes bad_day_* columns.
    """
    if df_test is None or len(df_test) == 0:
        print("No test data available for correlation heatmap")
        return

    if series_dict_test is None:
        print("No test series data available for correlation heatmap")
        return

    combined_df = df_test.copy()

    for target_name in target_cols:
        if target_name in series_dict_test:
            true_s, pred_s, _ = series_dict_test[target_name]
            combined_df[f"{target_name}_actual"] = true_s
            combined_df[f"{target_name}_pred"] = pred_s

    feature_cols = [col for col in combined_df.columns
                    if not col.startswith('bad_day_')
                    and col not in target_cols
                    and not col.endswith('_actual')
                    and not col.endswith('_pred')]

    target_actual_pred_cols = []
    for target_name in target_cols:
        actual_col = f"{target_name}_actual"
        pred_col = f"{target_name}_pred"
        if actual_col in combined_df.columns:
            target_actual_pred_cols.append(actual_col)
        if pred_col in combined_df.columns:
            target_actual_pred_cols.append(pred_col)

    corr_cols = feature_cols + target_actual_pred_cols
    corr_cols = [col for col in corr_cols if col in combined_df.columns]

    if not corr_cols or not target_actual_pred_cols:
        print("No valid columns for correlation heatmap")
        return

    corr_df = combined_df[corr_cols].corr()

    target_corr = corr_df.loc[target_actual_pred_cols, feature_cols]

    fig, ax = plt.subplots(figsize=(16, len(target_actual_pred_cols) * 0.5 + 2))

    sns.heatmap(
        target_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Correlation Coefficient'}
    )

    ax.set_title('Correlation: Target Actual/Pred vs All Input Features (Test Set - Denormalized)',
                 fontsize=14, pad=20)
    ax.set_xlabel('Input Features', fontsize=12)
    ax.set_ylabel('Target Actual/Pred', fontsize=12)

    plt.tight_layout()
    corr_file = plots_dir / "correlation_heatmap_targets_vs_features_test.png"
    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {corr_file}")


def save_target_vs_irr_scatter(
    df_test: pd.DataFrame,
    series_dict_test: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    target_cols: List[str],
    plots_dir: Path,
    irr_col: str = "Irr",
):
    """
    Create scatter plots for each target (actual and pred) vs Irr.
    Uses only TEST data. Each target gets its own figure with 2 subplots.
    """
    if df_test is None or len(df_test) == 0:
        print("No test data available for Irr scatter plots")
        return

    if series_dict_test is None:
        print("No test series data available for Irr scatter plots")
        return

    if irr_col not in df_test.columns:
        print(f"Warning: '{irr_col}' column not found in test dataframe. Skipping Irr scatter plots.")
        return

    for target_name in target_cols:
        if target_name not in series_dict_test:
            print(f"Warning: No data found for target '{target_name}'. Skipping.")
            continue

        true_s, pred_s, _ = series_dict_test[target_name]

        common_idx = df_test.index.intersection(true_s.index).intersection(pred_s.index)

        if len(common_idx) == 0:
            print(f"Warning: No common timestamps for target '{target_name}'. Skipping.")
            continue

        irr_values = df_test.loc[common_idx, irr_col].values
        actual_values = true_s.loc[common_idx].values
        pred_values = pred_s.loc[common_idx].values

        valid_mask = ~(np.isnan(irr_values) | np.isnan(actual_values) | np.isnan(pred_values))
        irr_clean = irr_values[valid_mask]
        actual_clean = actual_values[valid_mask]
        pred_clean = pred_values[valid_mask]

        if len(irr_clean) == 0:
            print(f"Warning: No valid data for target '{target_name}' after removing NaNs. Skipping.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(irr_clean, actual_clean, s=2, alpha=0.5, c='blue')
        axes[0].set_xlabel(f'{irr_col} (Denormalized)', fontsize=11)
        axes[0].set_ylabel(f'{target_name} Actual (Denormalized)', fontsize=11)
        axes[0].set_title(f'{target_name} Actual vs {irr_col}', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        corr_actual = np.corrcoef(irr_clean, actual_clean)[0, 1]
        axes[0].text(0.02, 0.98, f'Corr: {corr_actual:.3f}',
                     transform=axes[0].transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        axes[1].scatter(irr_clean, pred_clean, s=2, alpha=0.5, c='red')
        axes[1].set_xlabel(f'{irr_col} (Denormalized)', fontsize=11)
        axes[1].set_ylabel(f'{target_name} Pred (Denormalized)', fontsize=11)
        axes[1].set_title(f'{target_name} Pred vs {irr_col}', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        corr_pred = np.corrcoef(irr_clean, pred_clean)[0, 1]
        axes[1].text(0.02, 0.98, f'Corr: {corr_pred:.3f}',
                     transform=axes[1].transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(f'{target_name} vs {irr_col} Scatter Plots - Test Set (n={len(irr_clean):,})',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        scatter_file = plots_dir / f"{target_name}_vs_{irr_col}_scatter_test.png"
        plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {scatter_file}")


def get_data_statistics(df: pd.DataFrame, split_name: str = "",
                         bad_day_mapping: Dict[str, str] = None) -> dict:
    """Get comprehensive statistics for a DataFrame with target-specific bad days."""
    if df is None or len(df) == 0:
        return None

    stats = {
        "split_name": split_name,
        "num_rows": len(df),
        "start_date": df.index.min().strftime('%Y-%m-%d'),
        "end_date": df.index.max().strftime('%Y-%m-%d'),
        "num_days": (df.index.max() - df.index.min()).days + 1,
    }

    if bad_day_mapping:
        bad_day_counts = {}
        for target, bad_col in bad_day_mapping.items():
            if bad_col and bad_col in df.columns:
                bad_day_counts[target] = int(df[bad_col].sum())
            else:
                bad_day_counts[target] = 0

        stats['bad_day_counts'] = bad_day_counts

        if bad_day_counts:
            bad_cols = [col for col in bad_day_mapping.values() if col and col in df.columns]
            if bad_cols:
                combined_bad = (df[bad_cols] == 1).any(axis=1)
                stats['bad_days'] = int(combined_bad.sum())
                stats['bad_days_pct'] = (stats['bad_days'] / len(df)) * 100

    return stats


def save_target_visualizations(
    target_name: str,
    series_dict_train: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    series_dict_val: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    series_dict_test: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]],
    plots_dir: Path,
    mask_bad_days: bool = True
):
    """
    Save per-target visualizations:
    1. Scatter plot (1 row, 3 cols: train/val/test)
    2. Error histogram (1 row, 3 cols: train/val/test)
    3. Continuous series (3 rows, 1 col: train/val/test)
    """
    splits = [
        ("Train", series_dict_train),
        ("Val", series_dict_val),
        ("Test", series_dict_test) if series_dict_test else None
    ]
    splits = [s for s in splits if s is not None]

    # 1. SCATTER PLOT
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    for idx, (split_name, series_dict) in enumerate(splits):
        if target_name in series_dict:
            true_s, pred_s, bad_s = series_dict[target_name]
            save_scatter_ax(
                axes[idx],
                true_s.values,
                pred_s.reindex(true_s.index).values,
                bad_s,
                f"{split_name}",
                mask_bad_days=mask_bad_days
            )
        else:
            axes[idx].set_visible(False)

    plt.suptitle(f"{target_name} - Scatter Plots", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    scatter_file = plots_dir / f"{target_name}_scatter.png"
    plt.savefig(scatter_file, dpi=150)
    plt.close()
    print(f"Saved: {scatter_file}")

    # 2. ERROR HISTOGRAM
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    for idx, (split_name, series_dict) in enumerate(splits):
        if target_name in series_dict:
            true_s, pred_s, bad_s = series_dict[target_name]
            save_error_histogram_ax(
                axes[idx],
                true_s.values,
                pred_s.reindex(true_s.index).values,
                bad_s,
                f"{split_name}",
                bins=60,
                mask_bad_days=mask_bad_days
            )
        else:
            axes[idx].set_visible(False)

    plt.suptitle(f"{target_name} - Error Histograms", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    hist_file = plots_dir / f"{target_name}_histogram.png"
    plt.savefig(hist_file, dpi=150)
    plt.close()
    print(f"Saved: {hist_file}")

    # 3. CONTINUOUS SERIES
    fig, axes = plt.subplots(len(splits), 1, figsize=(12, 5 * len(splits)), sharex=False)
    if len(splits) == 1:
        axes = [axes]

    for idx, (split_name, series_dict) in enumerate(splits):
        if target_name in series_dict:
            true_s, pred_s, bad_s = series_dict[target_name]
            plot_time_series_ax(
                axes[idx],
                true_s,
                pred_s,
                bad_s,
                f"{split_name}",
                mask_bad_days=mask_bad_days
            )
        else:
            axes[idx].set_visible(False)

    plt.suptitle(f"{target_name} - Continuous Time Series", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    series_file = plots_dir / f"{target_name}_continuous.png"
    plt.savefig(series_file, dpi=150)
    plt.close()
    print(f"Saved: {series_file}")


def save_all_metrics_to_txt(
    metrics_tr: Dict[str, Dict],
    metrics_va: Dict[str, Dict],
    metrics_te: Dict[str, Dict],
    target_cols: List[str],
    output_file: Path,
    model_info: dict
):
    """
    Save all metrics for all targets to a common text file.
    """
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AGE-AWARE MULTI-OUTPUT LSTM MODEL METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Targets: {', '.join(target_cols)}\n")
        f.write(f"  Age columns: {model_info['data_config']['features'].get('age_cols', [])}\n")
        f.write(f"  Bad day mapping: {model_info['data_config'].get('bad_day_mapping', {})}\n")
        f.write(f"  Window: {model_info['data_config']['window_horizon']['window']}, Horizon: {model_info['data_config']['window_horizon']['horizon']}\n")
        f.write(f"  Hidden size: {model_info['architecture']['hidden_size']}, Layers: {model_info['architecture']['num_layers']}\n")
        f.write(f"  Age head: hidden={model_info['architecture'].get('age_hidden')}, embed_dim={model_info['architecture'].get('age_embed_dim')}\n")
        f.write(f"  Dropout: {model_info['architecture']['dropout']}\n")
        f.write(f"  use_bad_day: {model_info['data_config']['features']['use_bad_day']}, mask_bad_days: {model_info['data_config']['features']['mask_bad_days']}\n")
        f.write(f"  Total parameters: {model_info['architecture']['total_params']:,}\n\n")

        f.write("=" * 80 + "\n")
        f.write("METRICS (computed from continuous normalized series with bad day exclusion)\n")
        f.write("=" * 80 + "\n\n")

        for target_name in target_cols:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"TARGET: {target_name}\n")
            f.write(f"{'=' * 80}\n")

            f.write("\nTRAIN METRICS:\n")
            f.write("-" * 40 + "\n")
            if target_name in metrics_tr:
                for k, v in metrics_tr[target_name].items():
                    if k == 'target':
                        continue
                    if isinstance(v, float):
                        f.write(f"  {k:25s}: {v:.6f}\n")
                    else:
                        f.write(f"  {k:25s}: {v}\n")

            f.write("\nVALIDATION METRICS:\n")
            f.write("-" * 40 + "\n")
            if target_name in metrics_va:
                for k, v in metrics_va[target_name].items():
                    if k == 'target':
                        continue
                    if isinstance(v, float):
                        f.write(f"  {k:25s}: {v:.6f}\n")
                    else:
                        f.write(f"  {k:25s}: {v}\n")

            if metrics_te and target_name in metrics_te:
                f.write("\nTEST METRICS:\n")
                f.write("-" * 40 + "\n")
                for k, v in metrics_te[target_name].items():
                    if k == 'target':
                        continue
                    if isinstance(v, float):
                        f.write(f"  {k:25s}: {v:.6f}\n")
                    else:
                        f.write(f"  {k:25s}: {v}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF METRICS REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Saved metrics to: {output_file}")


# ---------------- Main training function for multi-output ----------------
def run_lstm_training_multi_output(
    training_data_dir: str,
    output_dir: str = None,
    target_cols: List[str] = None,
    age_cols: List[str] = None,
    window: int = 48,
    horizon: int = 12,
    batch_size: int = 36,
    epochs: int = 200,
    lr: float = 0.0001,
    patience: int = 30,
    hidden_size: int = 32,
    num_layers: int = 4,
    dropout: float = 0.1,
    age_hidden: int = 8,
    age_embed_dim: int = 8,
    use_bad_day: bool = False,
    mask_bad_days: bool = False,
    validation_split: float = 0.15,
    random_seed: int = 42,
):
    """
    Run age-aware multi-output LSTM training with target-specific bad day
    masking. Age columns (e.g. 'days_since_install_<target>') are excluded
    from the LSTM's recurrent input and instead fed through a small,
    separate MLP head, combined with the LSTM's final hidden state per
    horizon step before the output layer.

    age_cols: list of column names to treat as age features (removed from
    the recurrent feature set). If None, defaults to
    ['days_since_install_<target>' for target in target_cols] if those
    columns exist in the training data; otherwise no age head input is used
    (age_num_features=0 and the age head degenerates to a constant, since an
    empty-width Linear layer still has a bias term).
    """
    if target_cols is None:
        target_cols = ['P']

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

    set_all_seeds(random_seed)

    print("\n" + "=" * 60)
    print("AGE-AWARE MULTI-OUTPUT LSTM TRAINING WITH TARGET-SPECIFIC BAD DAY MASKING")
    print(f"Target outputs: {target_cols}")
    print(f"FLAG CONFIGURATION:")
    print(f"  use_bad_day={use_bad_day}")
    print(f"  mask_bad_days={mask_bad_days}")
    print("=" * 60 + "\n")

    OUT_DIR = Path(output_dir) if output_dir else Path(training_data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_FPATH = Path(training_data_dir) / "train_scaled.parquet"
    VAL_FPATH = Path(training_data_dir) / "val_scaled.parquet"
    TEST_FPATH = Path(training_data_dir) / "test_scaled.parquet"
    timestamp = datetime.now().strftime("%d.%m.%Y.%H%M%S")
    MODEL_FILE = OUT_DIR / f"{timestamp}_model_multi.pt"
    METRICS_LOG = OUT_DIR / "training_metrics_multi.txt"
    PLOTS_DIR = OUT_DIR / "plots_multi"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    SCALER_FILE = Path(training_data_dir) / "p_scalers.pkl"
    with open(SCALER_FILE, "rb") as f:
        scaler_data = pickle.load(f)

    scaler_dict = {}
    for target_name in target_cols:
        if target_name in scaler_data:
            scaler_dict[target_name] = scaler_data[target_name]
        elif "scaler" in scaler_data:
            scaler_dict[target_name] = scaler_data["scaler"]
        else:
            raise ValueError(f"Scaler for target '{target_name}' not found in {SCALER_FILE}")

    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE} (CPU forced)")

    required_files = [TRAIN_FPATH, SCALER_FILE]
    for fpath in required_files:
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")

    val_data_exists = VAL_FPATH.exists()
    test_data_exists = TEST_FPATH.exists()

    print("\nLoading data...")
    train_df = pd.read_parquet(TRAIN_FPATH)

    val_df = None
    if val_data_exists:
        try:
            val_df = pd.read_parquet(VAL_FPATH)
            if len(val_df) == 0:
                val_df = None
                val_data_exists = False
        except Exception as e:
            warnings.warn(f"Error reading validation data: {e}")
            val_df = None
            val_data_exists = False

    test_df = None
    if test_data_exists:
        try:
            test_df = pd.read_parquet(TEST_FPATH)
            if len(test_df) == 0:
                test_df = None
                test_data_exists = False
        except Exception as e:
            warnings.warn(f"Error reading test data: {e}")
            test_df = None
            test_data_exists = False

    for target in target_cols:
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' not found in training data")

    # Resolve age_cols: default to 'days_since_install_<target>' per target,
    # keeping only the ones that actually exist in the data.
    if age_cols is None:
        candidate_age_cols = [f"days_since_install_{t}" for t in target_cols]
        age_cols = [c for c in candidate_age_cols if c in train_df.columns]
        missing_candidates = [c for c in candidate_age_cols if c not in train_df.columns]
        if missing_candidates:
            print(f"NOTE: default age columns not found in data, skipping: {missing_candidates}")
    else:
        missing_age = [c for c in age_cols if c not in train_df.columns]
        if missing_age:
            raise ValueError(f"age_cols not found in training data: {missing_age}")

    print(f"\nAge columns (separate head input): {age_cols}")

    # Feature columns list: exclude targets AND age_cols (age is handled by
    # the separate head, never the recurrent input).
    initial_feature_cols = [c for c in train_df.columns if c not in target_cols and c not in age_cols]

    print(f"\nInitial recurrent features available: {initial_feature_cols}")
    print(f"Target columns: {target_cols}")

    bad_day_mapping = detect_bad_day_columns(train_df, target_cols)

    train_stats = get_data_statistics(train_df, "Train", bad_day_mapping)
    val_stats = get_data_statistics(val_df, "Validation", bad_day_mapping) if val_df is not None else None
    test_stats = get_data_statistics(test_df, "Test", bad_day_mapping) if test_df is not None else None

    # ============================================
    # CREATE DATASETS FOR TRAINING (OVERLAPPING)
    # ============================================
    print("\n" + "=" * 60)
    print("CREATING TRAINING DATASETS (OVERLAPPING, stride=1)")
    print("=" * 60)

    if val_data_exists:
        train_ds = AgeAwareMultiOutputDataset(
            train_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )
        val_ds = AgeAwareMultiOutputDataset(
            val_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )
    else:
        print(f"\nCreating validation split from training data ({validation_split:.0%})")

        train_ds_base = AgeAwareMultiOutputDataset(
            train_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )

        X_seq, age_seq, y_seq, bad_seq, ts_seq = [], [], [], [], []
        for i in range(len(train_ds_base)):
            x, age, y, bad = train_ds_base[i]
            X_seq.append(x.numpy())
            age_seq.append(age.numpy())
            y_seq.append(y.numpy())
            bad_seq.append(bad.numpy())
            ts_seq.append(train_ds_base.get_timestamps_for_sample(i))

        X_seq = np.stack(X_seq)
        age_seq = np.stack(age_seq)
        y_seq = np.stack(y_seq)
        bad_seq = np.stack(bad_seq)

        X_tr, X_va, age_tr, age_va, y_tr, y_va, bad_tr, bad_va, ts_tr, ts_va = split_train_val_from_sequences(
            X_seq, age_seq, y_seq, bad_seq, ts_seq,
            validation_split=validation_split,
            use_validation_set=True,
        )

        class SplitDataset(Dataset):
            def __init__(self, X, age, y, bad, timestamps, target_cols, age_cols, bad_day_mapping):
                self.X = X
                self.age = age
                self.y = y
                self.bad = bad
                self.timestamps = timestamps
                self.target_cols = target_cols
                self.age_cols = age_cols
                self.bad_day_mapping = bad_day_mapping

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.age[idx], self.y[idx], self.bad[idx]

            def get_timestamps_for_sample(self, idx):
                return self.timestamps[idx]

            def get_target_columns(self):
                return self.target_cols

            def get_age_columns(self):
                return self.age_cols

            def get_bad_day_mapping(self):
                return self.bad_day_mapping

        train_ds = SplitDataset(
            torch.from_numpy(X_tr), torch.from_numpy(age_tr), torch.from_numpy(y_tr),
            torch.from_numpy(bad_tr), ts_tr, target_cols, age_cols, bad_day_mapping
        )
        val_ds = SplitDataset(
            torch.from_numpy(X_va), torch.from_numpy(age_va), torch.from_numpy(y_va),
            torch.from_numpy(bad_va), ts_va, target_cols, age_cols, bad_day_mapping
        )

    # Test dataset (overlapping)
    test_ds_train = None
    if test_data_exists:
        test_ds_train = AgeAwareMultiOutputDataset(
            test_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )

    # ============================================
    # CREATE DATASETS FOR EVALUATION (NON-OVERLAPPING)
    # ============================================
    print("\n" + "=" * 60)
    print("CREATING EVALUATION DATASETS (NON-OVERLAPPING, stride=horizon)")
    print("=" * 60)

    train_ds_eval = AgeAwareNonOverlappingMultiOutputDataset(
        train_df, initial_feature_cols, target_cols, age_cols, window, horizon,
        use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
    )

    if val_data_exists:
        val_ds_eval = AgeAwareNonOverlappingMultiOutputDataset(
            val_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )
    else:
        val_ds_eval = None

    test_ds_eval = None
    if test_data_exists:
        test_ds_eval = AgeAwareNonOverlappingMultiOutputDataset(
            test_df, initial_feature_cols, target_cols, age_cols, window, horizon,
            use_bad_day=use_bad_day, mask_bad_days=mask_bad_days, bad_day_mapping=bad_day_mapping
        )

    actual_feature_cols = train_ds_base.feature_cols
    actual_age_cols = train_ds_base.age_cols

    input_size = len(actual_feature_cols)
    age_num_features = len(actual_age_cols)
    num_outputs = len(target_cols)

    print(f"\nActual recurrent features used: {actual_feature_cols}")
    print(f"Actual age columns used: {actual_age_cols}")
    print(f"Input size (recurrent): {input_size} features")
    print(f"Age input size: {age_num_features} feature(s)")
    print(f"Number of outputs: {num_outputs}")

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

    xb, age_b, yb, bmb = next(iter(train_loader))
    print("\n[DEBUG TRAINING LOADER]")
    print("xb:", xb.shape)     # (B, W+H, F)
    print("age_b:", age_b.shape)  # (B, H, age_num_features)
    print("yb:", yb.shape)     # (B, H, num_outputs)
    print("bmb:", bmb.shape)   # (B, H, num_outputs)
    print("-" * 100 + "\n")

    print("\nCreating age-aware multi-output model...")
    model = AgeAwareMultiOutputLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=num_outputs,
        dropout=dropout,
        forecast_steps=horizon,
        age_num_features=age_num_features,
        age_hidden=age_hidden,
        age_embed_dim=age_embed_dim,
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    print("\nStarting training...")
    history = train_loop(model, train_loader, val_loader, epochs, lr, patience,
                          DEVICE, MODEL_FILE)

    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # ============================================
    # MAKE PREDICTIONS (NON-OVERLAPPING)
    # ============================================
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS FOR EVALUATION (NON-OVERLAPPING)")
    print("=" * 60)

    train_loader_eval = DataLoader(
        train_ds_eval,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    if val_data_exists and val_ds_eval is not None:
        val_loader_eval = DataLoader(
            val_ds_eval,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        val_loader_eval = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )

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

    preds_test_eval, trues_test_eval, bad_test_eval = None, None, None
    if test_loader_eval:
        preds_test_eval, trues_test_eval, bad_test_eval = predict_with_loader(model, test_loader_eval, DEVICE)

    # ============================================
    # BUILD CONTINUOUS SERIES (NORMALIZED AND DENORMALIZED)
    # ============================================
    print("\n" + "=" * 60)
    print("BUILDING CONTINUOUS SERIES FOR ALL TARGETS")
    print("=" * 60)

    series_dict_train_norm, series_dict_train_denorm = build_continuous_series_multi_output(
        train_ds_eval, preds_train_eval, trues_train_eval, bad_train_eval,
        scaler_dict, target_cols, mask_bad_days=mask_bad_days
    )

    if val_data_exists and val_ds_eval is not None:
        series_dict_val_norm, series_dict_val_denorm = build_continuous_series_multi_output(
            val_ds_eval, preds_val_eval, trues_val_eval, bad_val_eval,
            scaler_dict, target_cols, mask_bad_days=mask_bad_days
        )
    else:
        series_dict_val_norm, series_dict_val_denorm = build_continuous_series_multi_output(
            val_ds, preds_val_eval, trues_val_eval, bad_val_eval,
            scaler_dict, target_cols, mask_bad_days=mask_bad_days
        )

    series_dict_test_norm, series_dict_test_denorm = None, None
    if test_ds_eval and preds_test_eval is not None:
        series_dict_test_norm, series_dict_test_denorm = build_continuous_series_multi_output(
            test_ds_eval, preds_test_eval, trues_test_eval, bad_test_eval,
            scaler_dict, target_cols, mask_bad_days=mask_bad_days
        )

    # ============================================
    # COMPUTE METRICS FROM CONTINUOUS NORMALIZED SERIES
    # ============================================
    print("\n" + "=" * 60)
    print("COMPUTING METRICS FROM CONTINUOUS NORMALIZED SERIES")
    print("=" * 60)

    metrics_tr = compute_metrics_from_continuous_series(series_dict_train_norm, target_cols, horizon)
    metrics_va = compute_metrics_from_continuous_series(series_dict_val_norm, target_cols, horizon)
    metrics_te = None
    if series_dict_test_norm is not None:
        metrics_te = compute_metrics_from_continuous_series(series_dict_test_norm, target_cols, horizon)

    # ============================================
    # BUILD MODEL INFO
    # ============================================
    model_info = {
        "model_class": "AgeAwareMultiOutputLSTMModel",
        "architecture": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_outputs": num_outputs,
            "dropout": dropout,
            "forecast_steps": horizon,
            "age_num_features": age_num_features,
            "age_hidden": age_hidden,
            "age_embed_dim": age_embed_dim,
            "activation": "ReLU (prevents negative predictions)",
            "lstm_batch_first": True,
            "total_params": total_params,
        },
        "data_config": {
            "features": {
                "feature_cols": actual_feature_cols,
                "age_cols": actual_age_cols,
                "target_cols": target_cols,
                "use_bad_day": use_bad_day,
                "mask_bad_days": mask_bad_days,
                "total_features": input_size,
            },
            "window_horizon": {
                "window": window,
                "horizon": horizon,
                "input_shape": f"({window + horizon}, {input_size})",
                "age_shape": f"({horizon}, {age_num_features})",
                "output_shape": f"({horizon}, {num_outputs})",
            },
            "bad_day_mapping": bad_day_mapping,
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
            "random_seed": random_seed,
            "deterministic": True,
        },
        "dataset_info": {
            "window": window,
            "horizon": horizon,
            "use_bad_day": use_bad_day,
            "mask_bad_days": mask_bad_days,
        },
        "environment": {
            "device": str(DEVICE),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "metrics_info": {
            "computed_from": "continuous_normalized_series",
            "persistence_method": "lag_1_persistence",
            "bad_day_handling": "target_specific_exclusion"
        }
    }

    with open(OUT_DIR / "model_info_multi.pkl", "wb") as f:
        pickle.dump(model_info, f)

    import json
    with open(OUT_DIR / "model_info_multi.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    save_all_metrics_to_txt(
        metrics_tr, metrics_va, metrics_te,
        target_cols,
        METRICS_LOG,
        model_info
    )

    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS PER TARGET")
    print("=" * 60)

    for target_name in target_cols:
        print(f"\nGenerating visualizations for {target_name}...")
        save_target_visualizations(
            target_name,
            series_dict_train_denorm,
            series_dict_val_denorm,
            series_dict_test_denorm,
            PLOTS_DIR,
            mask_bad_days=mask_bad_days
        )

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    save_training_history_ax(ax, history)
    plt.tight_layout()
    history_file = PLOTS_DIR / "training_history.png"
    plt.savefig(history_file, dpi=150)
    plt.close()
    print(f"\nSaved: {history_file}")

    print("\nSaving daily sample plots...")
    save_daily_pred_vs_true_plots_multi_output(
        train_df, series_dict_train_denorm, split_name="Train",
        plots_dir=PLOTS_DIR, random_seed=random_seed
    )

    val_df_for_plots = val_df if val_data_exists else train_df
    save_daily_pred_vs_true_plots_multi_output(
        val_df_for_plots, series_dict_val_denorm, split_name="Validation",
        plots_dir=PLOTS_DIR, random_seed=random_seed
    )

    if test_data_exists and series_dict_test_denorm is not None:
        save_daily_pred_vs_true_plots_multi_output(
            test_df, series_dict_test_denorm, split_name="Test",
            plots_dir=PLOTS_DIR, random_seed=random_seed
        )

        save_correlation_heatmaps(
            test_df, series_dict_test_denorm, target_cols, PLOTS_DIR
        )

        save_target_vs_irr_scatter(
            test_df, series_dict_test_denorm, target_cols, PLOTS_DIR, irr_col="Irr"
        )

    print(f"\nAge-aware multi-output training completed with target-specific bad day masking!")
    print(f"Targets: {', '.join(target_cols)}")
    print(f"Age columns: {actual_age_cols}")
    print(f"Bad day mapping: {bad_day_mapping}")
    print(f"Model saved to {MODEL_FILE.name}")
    print(f"Metrics logged to {METRICS_LOG}")
    print(f"Plots saved to {PLOTS_DIR}")
    print(f"Metrics computed from continuous normalized series with bad day exclusion")

    return {
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
        "bad_day_mapping": bad_day_mapping,
        "history": history
    }


# ---------------- Main function ----------------
def main():
    """Example usage"""
    training_data_dir = "/path/to/your/data"

    results = run_lstm_training_multi_output(
        training_data_dir=training_data_dir,
        target_cols=['P_normalised_si', 'P_normalised_psc', 'I_normalised_dc', 'U_grid'],
        age_cols=['days_since_install_perovskite'], # defaults to ['days_since_install_<target>', ...] if present
        window=48,
        horizon=12,
        batch_size=32,
        epochs=100,
        lr=0.001,
        patience=15,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        age_hidden=8,
        age_embed_dim=8,
        use_bad_day=True,
        mask_bad_days=True,
        validation_split=0.15,
    )

    return results


if __name__ == "__main__":
    main()