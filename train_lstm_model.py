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

class LSTMModel(nn.Module):
    """
    Two-head LSTM model:
    amp_head: Predicts peak amplitude (using ReLU)
    shape_head: Predicts normalized temporal shape (using Sigmoid)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        forecast_steps: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.amp_head = nn.Linear(hidden_size, output_size)
        self.shape_head = nn.Linear(hidden_size, forecast_steps * output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a multi-step forecast
        Args:
            x: Input tensor of shape (batch, window, input_size)
        Returns:
            Tensor of shape (batch, forecast_steps)
        """
        batch_size = x.size(0)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)

        amplitude = torch.relu(self.amp_head(out))
        shape = torch.sigmoid(self.shape_head(out))
        shape = shape.view(batch_size, self.forecast_steps, self.output_size)

        amplitude = amplitude.view(batch_size, 1, self.output_size)
        out = amplitude * shape

        return out.squeeze(-1)

class LSTMModelSimple(nn.Module):
    """
    Standard multi-step LSTM model with a single linear output head.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        forecast_steps: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, forecast_steps * output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a multi-step forecast
        Args:
            x: Input tensor of shape (batch, window, input_size)
        Returns:
            Tensor of shape (batch, forecast_steps)
        """
        batch_size = x.size(0)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        #out = self.dropout(out)
        out = self.fc(out)
        out = torch.relu(out)
        out = out.view(batch_size, self.forecast_steps, self.output_size)
        return out.squeeze(-1)

class MultiStepDataset(Dataset):
    """
        Creates sliding window samples for time series
        window: Lookback period (e.g., 48 timesteps)
        horizon: Forecast horizon (e.g., 12 timesteps)
        bad_day: Binary mask for low-quality data

        Bad days are handled via loss masking.
        use_bad_day: 
            If True, include bad_day as an input feature
            If False, model learns weather continuity only
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        target_col: str,
        window: int,
        horizon: int,
        use_bad_day: bool = False,
        mask_bad_days: bool = False,
    ):
        self.window = window
        self.horizon = horizon
        self.use_bad_day = use_bad_day
        self.mask_bad_days = mask_bad_days

        if use_bad_day and "bad_day" not in df.columns:
            raise ValueError("use_bad_day=True but 'bad_day' column not found.")

        if not use_bad_day and "bad_day" in feature_cols:
            feature_cols = [c for c in feature_cols if c != "bad_day"]

        self.X = df[feature_cols].to_numpy(np.float32)
        self.y = df[target_col].to_numpy(np.float32)
        self.ts = df.index

        if "bad_day" in df.columns:
            self.bad_day = df["bad_day"].to_numpy(np.float32)
        else:
            self.bad_day = np.zeros_like(self.y, dtype=np.float32)

        max_idx = len(df) - (window + horizon) + 1
        self.valid_indices = list(range(max_idx))
        self.n = len(self.valid_indices)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        i = self.valid_indices[idx]
        x = np.nan_to_num(self.X[i : i + self.window], nan=0.0).astype(np.float32)
        y = np.nan_to_num(self.y[i + self.window : i + self.window + self.horizon], nan=0.0).astype(np.float32)
        bad_day_mask = self.bad_day[i + self.window : i + self.window + self.horizon].astype(np.float32)

        if not self.mask_bad_days:
            bad_day_mask = np.zeros_like(y, dtype=np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(bad_day_mask),
        )

    def get_timestamps_for_sample(self, idx: int):
        """
        Return timestamps corresponding to the forecast horizon of a sample.
        """
        i = self.valid_indices[idx]
        return self.ts[i + self.window : i + self.window + self.horizon]

def masked_mae_with_bad_day(
    pred: torch.Tensor,
    true: torch.Tensor,
    bad_day_mask: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute MAE while excluding bad days from the loss.
    """
    good = 1.0 - bad_day_mask
    diff = torch.abs(pred - true) * good
    return diff.sum() / (good.sum() + 1e-8)

def weighted_mae_with_bad_day(
    pred: torch.Tensor,
    true: torch.Tensor,
    bad_day_mask: torch.Tensor,
    peak_thr: float = 0.6,
    peak_weight: float = 3.0,
    ) -> torch.Tensor:
    """
    Weighted MAE loss for PV forecasting.
    """
    good_mask = 1.0 - bad_day_mask
    peak_mask = (true >= peak_thr).float()
    weights = 1.0 + peak_weight * peak_mask
    abs_err = torch.abs(pred - true)
    weighted_err = abs_err * weights * good_mask
    denom = (weights * good_mask).sum().clamp_min(1e-8)
    return weighted_err.sum() / denom

def compute_masked_metrics_with_bad_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bad_day_mask: np.ndarray,
    peak_thr: float = 0.6,
    ) -> dict:
    """
    Compute evaluation metrics using only valid (non-bad-day) samples.
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    bad = bad_day_mask.flatten()

    good_idx = bad < 0.5
    yt = yt[good_idx]
    yp = yp[good_idx]

    if yt.size == 0:
        return {k: np.nan for k in [
            "MAE", "RMSE", "MAPE", "R2", "Directional_Accuracy",
            "Skill_Score", "Peak_Error", "Peak_Error_PeakOnly",
            "Peak_MAE", "Peak_Coverage"
        ]}

    abs_err = np.abs(yt - yp)
    #average absolute difference between predictions and actual values
    mae = abs_err.mean()
    #square root of average squared errors
    rmse = np.sqrt(np.mean((yt - yp) ** 2))

    eps = max(np.percentile(np.abs(yt), 5) * 0.01, 1e-6)
    valid = np.abs(yt) > eps
    #average percentage error relative to actual values
    mape = np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100 if valid.any() else np.nan

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    #proportion of variance explained by the model
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    pers = np.roll(yt, 1)
    pers[0] = yt[0]
    skill = 1 - np.mean((yt - yp) ** 2) / (np.mean((yt - pers) ** 2) + 1e-8)

    peak_mask = yt > peak_thr
    peak_cov = peak_mask.mean()

    #maximum absolute error across all predictions
    peak_err = abs_err.max()
    #maximum absolute error specifically during high-generation periods
    peak_err_pk = abs_err[peak_mask].max() if peak_mask.any() else np.nan
    #average error specifically during high-generation periods
    peak_mae = abs_err[peak_mask].mean() if peak_mask.any() else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Directional_Accuracy": np.nan,
        "Skill_Score": skill,
        "Peak_Error": peak_err,
        "Peak_Error_PeakOnly": peak_err_pk,
        "Peak_MAE": peak_mae,
        "Peak_Coverage": peak_cov,
        "Data_Points": yt.size,
        "Bad_Day_Excluded": np.sum(bad > 0.5),
        "Total_Points": bad.size,
    }

def predict_with_loader(model, loader, device):
    """
    Run model inference over a DataLoader and return predictions,
    ground truth values, and bad-day masks.
    """
    model.eval()
    all_preds = []
    all_trues = []
    all_bad_days = []

    with torch.no_grad():
        for xb, yb, bmb in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(yb.numpy())
            all_bad_days.append(bmb.numpy())

    preds_concat = np.vstack(all_preds)
    trues_concat = np.vstack(all_trues)
    bad_days_concat = np.vstack(all_bad_days)

    return preds_concat, trues_concat, bad_days_concat

def build_continuous_series(dataset, preds, trues, bad_days, scaler_path, mask_bad_days: bool = True):
    """
    Build continuous denormalized time series from windowed predictions and ground truth while filtering bad days.
    """
    with open(scaler_path, "rb") as f:
        p_scaler = pickle.load(f)["scaler"]

    P_pred = np.vstack(preds)
    P_true = np.vstack(trues)
    bad_mask = np.vstack(bad_days)

    P_pred_denorm = p_scaler.inverse_transform(P_pred.reshape(-1, 1)).reshape(P_pred.shape)
    P_true_denorm = p_scaler.inverse_transform(P_true.reshape(-1, 1)).reshape(P_true.shape)

    n_samples = min(
        len(dataset),
        P_pred_denorm.shape[0],
        P_true_denorm.shape[0],
        bad_mask.shape[0],
    )

    pred_map = {}
    true_map = {}

    for i in range(n_samples):
        timestamps = dataset.get_timestamps_for_sample(i)
        for j, ts in enumerate(timestamps):
            if mask_bad_days and bad_mask[i, j] > 0.5:
                continue
            if ts not in pred_map:
                pred_map[ts] = P_pred_denorm[i, j]
                true_map[ts] = P_true_denorm[i, j]

    ts_sorted = sorted(pred_map)
    pred_series = pd.Series([pred_map[t] for t in ts_sorted], index=ts_sorted, name="P_pred",)
    true_series = pd.Series([true_map[t] for t in ts_sorted], index=ts_sorted, name="P_true",)

    return true_series, pred_series

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

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

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
    Create a scatter plot of predictions versus ground truth,
    ignoring NaN values.
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    valid_mask = ~(np.isnan(yt) | np.isnan(yp))
    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]

    if len(yt_valid) == 0:
        ax.text(0.5, 0.5, "No valid data to plot",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return

    ax.scatter(yt_valid, yp_valid, s=2, alpha=0.5)
    lims = [
        min(yt_valid.min(), yp_valid.min()),
        max(yt_valid.max(), yp_valid.max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1)
    ax.set_xlabel("Actual (normalized)")
    ax.set_ylabel("Predicted (normalized)")
    ax.set_title(f"{title} (n={len(yt_valid)})")
    ax.grid(True, alpha=0.3)


def save_training_history_ax(ax, history):
    """
    Plot training and validation loss curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if epochs:
        ax.text(
            0.98,
            0.98,
            f"Final Train: {history['train_loss'][-1]:.4f}\n"
            f"Final Val: {history['val_loss'][-1]:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

def plot_time_series_ax(ax, true_series, pred_series, title):
    """
    Plot time series, handling NaNs in true series.
    """
    if true_series.empty or pred_series.empty:
        ax.text(0.5, 0.5, "No data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (No data)")
        return
    
    valid_mask = ~np.isnan(true_series.values)
    if np.any(valid_mask):
        valid_times = true_series.index[valid_mask]
        ax.plot(valid_times, true_series.values[valid_mask], 
                label="Actual", linewidth=1.5, alpha=0.8)
        ax.plot(pred_series.index, pred_series.values, 
                "--", label="Predicted", linewidth=1.2, alpha=0.8)
        
        #metrics
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

def build_pdf_report(out_pdf_path, model_info, train_shape, val_shape, test_shape,
                     feature_cols, metrics_tr, metrics_va, metrics_te,
                     history, train_true_series, train_pred_series,
                     val_true_series, val_pred_series,
                     test_true_series, test_pred_series):
    
    wrapped_features = textwrap.wrap(", ".join(feature_cols), width=90)
    
    with PdfPages(out_pdf_path) as pdf:
        #Page 1: summary + table
        fig = plt.figure(figsize=(8.27, 11.69)) 
        plt.axis("off")
        
        text_lines = [
            "LSTM Training Report",
            f"bad_day usage: {'Included as input feature' if model_info.get('use_bad_day', False) else 'Not included (weather only)'}",
            "",
            "Hyperparameters:",
            f"  WINDOW={model_info['window']}, HORIZON={model_info['horizon']}, BATCH_SIZE={model_info['batch_size']}",
            f"  EPOCHS={model_info['epochs']}, LR={model_info['lr']:.6f}, PATIENCE={model_info['patience']}",
            "",
            "Model parameters:",
            f"  hidden_size={model_info['hidden_size']}, num_layers={model_info['num_layers']}, dropout={model_info['dropout']}",
            f"  use_bad_day={model_info.get('use_bad_day', False)}",
            f"  total_parameters={model_info['total_params']:,}",
            "",
            "Data splits:",
            f"  Train rows: {train_shape}",
            f"  Val rows:   {val_shape}",
            f"  Test rows:  {test_shape}",
            "",
            "Feature list:"
        ]
        text_lines.extend(["  " + line for line in wrapped_features])
        plt.text(
            0.02, 0.98,
            "\n".join(text_lines),
            va="top", ha="left",
            fontsize=9, family="monospace",
            transform=fig.transFigure
        )
        
        #metrics table
        metric_keys = [
            "MAE", "RMSE", "MAPE", "R2", 
            "Directional_Accuracy", "Skill_Score", "Peak_Error", 
            "Data_Points", "Bad_Day_Excluded", "Total_Points"
        ]
        
        def make_metrics_table(ax, title):
            table_data = [["Metric", "Train", "Val", "Test"]]
            
            for key in metric_keys:
                if key == "Data_Points" or key == "Bad_Day_Excluded" or key == "Total_Points":
                    row = [
                        key,
                        f"{metrics_tr.get(key, 0)}",
                        f"{metrics_va.get(key, 0)}",
                        f"{metrics_te.get(key, 0)}",
                    ]
                else:
                    row = [
                        key,
                        f"{metrics_tr.get(key, np.nan):.4f}",
                        f"{metrics_va.get(key, np.nan):.4f}",
                        f"{metrics_te.get(key, np.nan):.4f}",
                    ]
                table_data.append(row)
            
            table = ax.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            
            for j in range(4):
                table[(0, j)].set_facecolor('#DDDDDD')
                table[(0, j)].set_text_props(weight='bold')
            
            ax.set_title(title, fontsize=10, weight='bold', pad=20)
            ax.axis('off')
        
        ax_table = fig.add_axes([0.1, 0.15, 0.8, 0.35])
        make_metrics_table(ax_table, "METRICS (computed only on good days)")
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        #Page 2: Scatter plots + history
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        
        # Prepare data for scatter plots
        train_y_true = train_true_series.values[~np.isnan(train_true_series.values)]
        train_y_pred = train_pred_series.reindex(train_true_series.index).values[~np.isnan(train_true_series.values)]
        val_y_true = val_true_series.values[~np.isnan(val_true_series.values)]
        val_y_pred = val_pred_series.reindex(val_true_series.index).values[~np.isnan(val_true_series.values)]
        test_y_true = test_true_series.values[~np.isnan(test_true_series.values)]
        test_y_pred = test_pred_series.reindex(test_true_series.index).values[~np.isnan(test_true_series.values)]
        
        save_scatter_ax(axes[0, 0], train_y_true, train_y_pred, "Train: Pred vs Actual")
        save_scatter_ax(axes[0, 1], val_y_true, val_y_pred, "Val: Pred vs Actual")
        save_scatter_ax(axes[1, 0], test_y_true, test_y_pred, "Test: Pred vs Actual")
        save_training_history_ax(axes[1, 1], history)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        #Page 3: Full series (train/val/test)
        fig, axes = plt.subplots(3, 1, figsize=(11.69, 16))
        
        plot_time_series_ax(axes[0], train_true_series, train_pred_series, "Train: Actual vs Predicted")
        plot_time_series_ax(axes[1], val_true_series, val_pred_series, "Validation: Actual vs Predicted")
        plot_time_series_ax(axes[2], test_true_series, test_pred_series, "Test: Actual vs Predicted")
        
        axes[2].set_xlabel("Time")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        #Page 4: Error Analysis
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        
        # Error distributions
        def plot_error_hist(ax, true_series, pred_series, title):
            if true_series.empty or pred_series.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                return
            
            valid_mask = ~np.isnan(true_series.values)
            if np.any(valid_mask):
                y_true = true_series.values[valid_mask]
                y_pred = pred_series.reindex(true_series.index).values[valid_mask]
                errors = y_true - y_pred
                
                ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', density=True)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
                ax.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=1.5, 
                          label=f'Mean: {np.mean(errors):.3f}')
                ax.set_title(title)
                ax.set_xlabel("Error (Actual - Predicted)")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plot_error_hist(axes[0, 0], train_true_series, train_pred_series, "Train Error Distribution")
        plot_error_hist(axes[0, 1], val_true_series, val_pred_series, "Val Error Distribution")
        plot_error_hist(axes[1, 0], test_true_series, test_pred_series, "Test Error Distribution")
        
        axes[1, 1].axis('off')
        if not test_true_series.empty:
            valid_mask = ~np.isnan(test_true_series.values)
            if np.any(valid_mask):
                test_times = test_true_series.index[valid_mask]
                test_errors = test_true_series.values[valid_mask] - test_pred_series.reindex(test_true_series.index).values[valid_mask]
                axes[1, 1].plot(test_times, test_errors, '-', alpha=0.7, linewidth=1)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
                axes[1, 1].set_title("Test Errors Over Time")
                axes[1, 1].set_xlabel("Time")
                axes[1, 1].set_ylabel("Error")
                axes[1, 1].grid(True, alpha=0.3)
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        print(f"PDF report saved to {out_pdf_path}")

def save_normalized_scatter_plots(preds, trues, bad_days, dataset_name, plots_dir):
    """
    Save normalized [0-1] scatter plots, filtering out bad days.
    """
    yt_flat = trues.flatten()
    yp_flat = preds.flatten()
    bad_flat = bad_days.flatten()
    
    good_idx = bad_flat < 0.5
    yt_valid = yt_flat[good_idx]
    yp_valid = yp_flat[good_idx]
    
    if len(yt_valid) == 0:
        print(f"No valid data for {dataset_name} scatter plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(yt_valid, yp_valid, s=5, alpha=0.5, color='blue')
    
    lim_min = min(yt_valid.min(), yp_valid.min())
    lim_max = max(yt_valid.max(), yp_valid.max())
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.7, linewidth=1)
    
    axes[0].set_xlabel("Actual (normalized)")
    axes[0].set_ylabel("Predicted (normalized)")
    axes[0].set_title(f"{dataset_name}: Predicted vs Actual (n={len(yt_valid)})")
    axes[0].grid(True, alpha=0.3)
    
    #metrics
    mae = np.mean(np.abs(yt_valid - yp_valid))
    rmse = np.sqrt(np.mean((yt_valid - yp_valid) ** 2))
    r2 = 1 - np.sum((yt_valid - yp_valid) ** 2) / np.sum((yt_valid - np.mean(yt_valid)) ** 2)
    
    text_str = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nn: {len(yt_valid)}"
    axes[0].text(0.05, 0.95, text_str, transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    errors = yt_valid - yp_valid
    axes[1].hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1].axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=1.5,
                   label=f'Mean: {np.mean(errors):.3f}')
    axes[1].set_xlabel("Error (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{dataset_name}: Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / f"{dataset_name.lower()}_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot: {plot_path}")

def save_daily_pred_vs_true_plots(
    df: pd.DataFrame,
    true_series: pd.Series,
    pred_series: pd.Series,
    split_name: str,
    plots_dir: Path,
    day_list: list[str] | None = None,
    irr_col: str = "Irr",
    ):
    DEFAULT_DAY_LIST = [
        "26-12-2025",
        "15-11-2025",
        "31-10-2025",
        "11-09-2025",
        "03-08-2025",
        "07-07-2025",
        "17-06-2025",
        "14-05-2025",
        "26-04-2025",
        "16-03-2025",
        "15-02-2025",
        "18-01-2025",
        "27-12-2024",
        "29-11-2024",
    ]
    if day_list is None:
        day_list = DEFAULT_DAY_LIST

    df = df.copy()
    df = df.join(true_series.rename("P_true"), how="left")
    df = df.join(pred_series.rename("P_pred"), how="left")

    df["date"] = df.index.date

    days = [pd.to_datetime(d, dayfirst=True).date() for d in day_list]

    for day in days:
        day_df = df[df["date"] == day]

        if day_df.empty:
            print(f"[SKIP] {day} not found in data")
            continue

        if (day_df["bad_day"] == 1).any():
            print(f"[SKIP] {day} marked as bad_day")
            continue

        day_df = day_df.dropna(subset=["P_true", "P_pred", irr_col])
        if day_df.empty:
            print(f"[SKIP] {day} has no valid data after NaN drop")
            continue

        times = day_df.index
        month = times[0].to_period("M")

        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(times, day_df["P_true"], label="True", linewidth=2, marker="o", markersize=3)
        ax1.plot(times, day_df["P_pred"], label="Pred", linestyle="--", marker="x", markersize=4)

        ax1.set_xlabel("Time (hour)")
        ax1.set_ylabel("P (unscaled)")
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
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3, fontsize=8)

        plt.tight_layout()
        fname = plots_dir / f"{split_name.lower()}_{day}_pred_vs_true.png"
        plt.savefig(fname, dpi=150)
        plt.close()

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
):
    OUT_DIR = Path(output_dir) if output_dir else Path(training_data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    TRAIN_FPATH = Path(training_data_dir) / "train_scaled.parquet"
    VAL_FPATH = Path(training_data_dir) / "val_scaled.parquet"
    TEST_FPATH = Path(training_data_dir) / "test_scaled.parquet"
    MODEL_FILE = OUT_DIR / "best_model.pt"
    METRICS_LOG = OUT_DIR / "training_metrics.txt"
    PLOTS_DIR = OUT_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    #load the P scaler
    P_SCALER_FILE = Path(training_data_dir) / "p_scaling.pkl"
    
    #device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    
    # Check if files exist
    for fpath in [TRAIN_FPATH, VAL_FPATH, TEST_FPATH, P_SCALER_FILE]:
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    train_df = pd.read_parquet(TRAIN_FPATH)
    val_df = pd.read_parquet(VAL_FPATH)
    test_df = pd.read_parquet(TEST_FPATH)
    
    feature_cols = [c for c in train_df.columns if c != target_col]
    #If using bad_day as feature, make sure it's included
    if use_bad_day and 'bad_day' not in feature_cols:
        feature_cols.append('bad_day')
    
    input_size = len([c for c in feature_cols if c != 'bad_day'])
    if use_bad_day:
        input_size += 1  # bad_day will be added as an additional feature
    
    print(f"\nOriginal splits: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Using {input_size} features {'(including bad_day)' if use_bad_day else ''}")
    
    #Create datasets
    train_ds = MultiStepDataset(train_df, feature_cols, target_col, window, horizon,
                               use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
    val_ds = MultiStepDataset(val_df, feature_cols, target_col, window, horizon,
                             use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
    test_ds = MultiStepDataset(test_df, feature_cols, target_col, window, horizon,
                              use_bad_day=use_bad_day, mask_bad_days=mask_bad_days)
    
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    #Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    #Create model
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
        
    # Train model
    print("\nStarting training...")
    history = train_loop(model, train_loader, val_loader, epochs, lr, patience, 
                        DEVICE, MODEL_FILE)
    
    # Load best model
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Make predictions
    print("\nMaking predictions...")
    preds_train, trues_train, bad_train = predict_with_loader(model, train_loader, DEVICE)
    preds_val, trues_val, bad_val = predict_with_loader(model, val_loader, DEVICE)
    preds_test, trues_test, bad_test = predict_with_loader(model, test_loader, DEVICE)
    
    # Compute metrics using only good days
    print("\nComputing metrics (only good days)...")
    metrics_tr = compute_masked_metrics_with_bad_day(trues_train, preds_train, bad_train)
    metrics_va = compute_masked_metrics_with_bad_day(trues_val, preds_val, bad_val)
    metrics_te = compute_masked_metrics_with_bad_day(trues_test, preds_test, bad_test)
    
    print("\nSaving plots to PLOTS_DIR...")
    #Save normalized scatter plots 
    save_normalized_scatter_plots(preds_train, trues_train, bad_train, "Train", PLOTS_DIR)
    save_normalized_scatter_plots(preds_val, trues_val, bad_val, "Validation", PLOTS_DIR)
    save_normalized_scatter_plots(preds_test, trues_test, bad_test, "Test", PLOTS_DIR)

    # Save metrics
    with open(METRICS_LOG, "w") as f:
        f.write(f"LSTM Model\n")
        f.write(f"Window={window}, Horizon={horizon}\n")
        f.write(f"Model: hidden_size={hidden_size}, layers={num_layers}, dropout={dropout}\n")
        f.write(f"bad_day usage: {'Included as input feature' if use_bad_day else 'Not included'}\n")
        
        f.write("TRAIN METRICS (only good days)\n")
        for k, v in metrics_tr.items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\nVAL METRICS (only good days)\n")
        for k, v in metrics_va.items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\nTEST METRICS (only good days)\n")
        for k, v in metrics_te.items():
            f.write(f"  {k}: {v}\n")
    
    print("\n")
    print("TRAIN:", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_tr.items()})
    print("VAL:  ", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_va.items()})
    print("TEST: ", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics_te.items()})
    
    # Build continuous denormalized series
    print("\nBuilding continuous series.")
    train_true_series, train_pred_series = build_continuous_series(train_ds, preds_train, trues_train, bad_train, P_SCALER_FILE, mask_bad_days=mask_bad_days)
    val_true_series, val_pred_series = build_continuous_series(val_ds, preds_val, trues_val, bad_val, P_SCALER_FILE, mask_bad_days=mask_bad_days)
    test_true_series, test_pred_series = build_continuous_series(test_ds, preds_test, trues_test, bad_test, P_SCALER_FILE, mask_bad_days=mask_bad_days)
    
    print(f"Series lengths - Train: {len(train_true_series)}, Val: {len(val_true_series)}, Test: {len(test_true_series)}")

    print(f"Series lengths after filtering bad days:")
    print(f"Train: {len(train_true_series)} (from {len(train_ds) * horizon} possible)")
    print(f"Val:   {len(val_true_series)} (from {len(val_ds) * horizon} possible)")
    print(f"Test:  {len(test_true_series)} (from {len(test_ds) * horizon} possible)")
    
    #Saving random daily pred-vs-true plots (unscaled P, scaled Irr)
    save_daily_pred_vs_true_plots(train_df, train_true_series, train_pred_series, split_name="Train", plots_dir=PLOTS_DIR)
    save_daily_pred_vs_true_plots(val_df, val_true_series, val_pred_series, split_name="Validation", plots_dir=PLOTS_DIR)
    save_daily_pred_vs_true_plots(test_df, test_true_series, test_pred_series, split_name="Test", plots_dir=PLOTS_DIR)

    # Build model info
    model_info = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "window": window,
        "horizon": horizon,
        "feature_cols": [c for c in feature_cols if c != 'bad_day'],
        "target_col": target_col,
        "device": str(DEVICE),
        "total_params": total_params,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "use_bad_day": use_bad_day,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    
    with open(OUT_DIR / "model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    
    #Create PDF report
    out_pdf = OUT_DIR / "report.pdf"
    print("\nGenerating PDF report...")
    build_pdf_report(out_pdf, model_info, 
                    len(train_ds), len(val_ds), len(test_ds),
                    [c for c in feature_cols if c != 'bad_day'],
                    metrics_tr, metrics_va, metrics_te,
                    history, train_true_series, train_pred_series,
                    val_true_series, val_pred_series,
                    test_true_series, test_pred_series)
    
    print(f"\nTraining completed.")
    print(f"Best model saved to {MODEL_FILE}")
    print(f"Metrics logged to {METRICS_LOG}")
    print(f"Model info saved to {OUT_DIR / 'model_info.pkl'}")
    print(f"PDF report saved to {out_pdf}")
    
    return {
        "model": model,
        "metrics": {
            "train": metrics_tr,
            "val": metrics_va,
            "test": metrics_te
        },
        "model_info": model_info,
        "series": {
            "train": (train_true_series, train_pred_series),
            "val": (val_true_series, val_pred_series),
            "test": (test_true_series, test_pred_series)
        }
    }

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
    )
    return results

if __name__ == "__main__":
    main()