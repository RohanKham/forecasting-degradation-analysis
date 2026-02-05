#Version: 1.0 
#Handles only 1 output
import pickle
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from typing import Dict, Tuple, Optional, Any, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import textwrap
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.dates as mdates
import random
import seaborn as sns
from datetime import datetime, timedelta

# Import from train_lstm_model
from train_lstm_model import (
    LSTMModel,
    NonOverlappingMultiStepDataset,
    predict_with_loader,
    build_continuous_series,
    compute_masked_metrics_with_bad_day
)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class LSTMEvaluator:
    """
    LSTM Evaluator using functions from train_lstm_model.py
    """
    def __init__(
        self,
        model_path: str,
        model_info_path: str,
        test_data_path: str,
        scaler_path: str,
        output_dir: str = None,
        random_seed: int = 42
    ):
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.test_data_path = Path(test_data_path)
        self.scaler_path = Path(scaler_path)
        self.random_seed = random_seed
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.model_path.parent / "test_evaluation"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.model_info = None
        self.test_df = None
        self.scaler = None
        self.device = None
        self.model = None
        self.test_ds = None
        self.test_loader = None
        self.predictions = None
        self.ground_truth = None
        self.bad_day_masks = None
        self.true_series_norm = None
        self.pred_series_norm = None
        self.true_series_denorm = None
        self.pred_series_denorm = None
        self.bad_series = None
        self.df_continuous_norm = None
        self.df_continuous_denorm = None
        self.metrics = None
        
        # Set random seeds
        self._set_random_seeds()
        
        print(f"\nLSTMEvaluator initialized")
        print(f"Model: {self.model_path}")
        print(f"Test data: {self.test_data_path}")
        print(f"Output: {self.output_dir}")
    
    def _set_random_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    def load_model_info(self):
        print("Loading model info")
        print("-" * 100)
        
        with open(self.model_info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        print(f"Model info loaded from {self.model_info_path}")
        print(f"Model class: {self.model_info['model_class']}")
        print(f"Window: {self.model_info['dataset_info']['window']}")
        print(f"Horizon: {self.model_info['dataset_info']['horizon']}")
        print(f"Features: {self.model_info['data_config']['features']['feature_cols']}")
        print(f"Target: {self.model_info['data_config']['features']['target_col']}")
    
    def load_test_data(self):
        print("\n Loading test data")
        print("-" * 100)
       
        self.test_df = pd.read_parquet(self.test_data_path)        
        print(f"Test data loaded from {self.test_data_path}")
        print(f"Shape: {self.test_df.shape}")
        print(f"Date range: {self.test_df.index.min()} to {self.test_df.index.max()}")
        print(f"Columns: {list(self.test_df.columns)}")
        
        # Load scaler for target
        with open(self.scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            self.scaler = scaler_data["scaler"]
        print(f"\nScaler loaded from {self.scaler_path}")
    
    def create_device(self):
        self.device = torch.device('cpu')
        print(f"\nUsing device: {self.device}")
    
    def create_model(self):
        print("\nLoading model")
        print("-" * 100)
        
        # Get model parameters
        input_size = self.model_info['architecture']['input_size']
        hidden_size = self.model_info['architecture']['hidden_size']
        num_layers = self.model_info['architecture']['num_layers']
        output_size = self.model_info['architecture']['output_size']
        forecast_steps = self.model_info['architecture']['forecast_steps']
        dropout = self.model_info['architecture']['dropout']
        
        # Create model using imported class
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            forecast_steps=forecast_steps,
            dropout=dropout
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {self.model_path}")
        print(f"Architecture: LSTM({input_size}, {hidden_size}, {num_layers})")
        print(f"Forecast steps: {forecast_steps}")
    
    def create_dataset(self):
        print("\nCreating non-overlapping test dataset")
        print("-" * 100)
        
        window = self.model_info['dataset_info']['window']
        horizon = self.model_info['dataset_info']['horizon']
        use_bad_day = self.model_info['dataset_info']['use_bad_day']
        mask_bad_days = self.model_info['dataset_info']['mask_bad_days']
        feature_cols = self.model_info['data_config']['features']['feature_cols']
        target_col = self.model_info['data_config']['features']['target_col']
        
        self.test_ds = NonOverlappingMultiStepDataset(
            df=self.test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            window=window,
            horizon=horizon,
            use_bad_day=use_bad_day,
            mask_bad_days=mask_bad_days,
            debug=True
        )
        
        print(f"Test dataset created: {len(self.test_ds)} samples")
    
    def create_dataloader(self, batch_size: int = 32):
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        print(f"\nTest dataloader created: batch_size={batch_size}")
    
    def predict(self):
        print("\nMaking predictions")
        print("-" * 100)
        
        self.predictions, self.ground_truth, self.bad_day_masks = \
            predict_with_loader(self.model, self.test_loader, self.device)
            
    def build_continuous_series(self):
        mask_bad_days = self.model_info['dataset_info']['mask_bad_days']
        target_col = self.model_info['data_config']['features']['target_col']
        
        # Get scaler for target
        target_scaler = self.scaler
        
        # Call the imported function
        series_norm, series_denorm = build_continuous_series(
            dataset=self.test_ds,
            preds=self.predictions,
            trues=self.ground_truth,
            bad_days=self.bad_day_masks,
            scaler=target_scaler,
            target_col=target_col,
            mask_bad_days=mask_bad_days,
            expected_freq="10min"
        )
        
        # Extract series from dict
        self.true_series_norm, self.pred_series_norm, self.bad_series_norm = series_norm[target_col]
        self.true_series_denorm, self.pred_series_denorm, self.bad_series_denorm = series_denorm[target_col]
        
        if not np.array_equal(self.bad_series_norm, self.bad_series_denorm):
            print("WARNING: bad_series mismatch between norm and denorm!")
            
    def compute_metrics(self):       
        self.metrics = compute_masked_metrics_with_bad_day(
            true_series=self.true_series_norm,
            pred_series=self.pred_series_norm,
            bad_series=self.bad_series_norm
        )
            
    def create_dataframe_for_plotting(self):
        self.df_continuous_norm = pd.DataFrame({
            'y_true': self.true_series_norm,
            'y_pred': self.pred_series_norm,
            'bad_day': self.bad_series_norm,
        })
        
        self.df_continuous_denorm = pd.DataFrame({
            'y_true': self.true_series_denorm,
            'y_pred': self.pred_series_denorm,
            'bad_day': self.bad_series_denorm
        })
        
        # Add Irr if available
        if 'Irr' in self.test_df.columns:
            irr_series = self.test_df['Irr'].reindex(self.df_continuous_norm.index)
            self.df_continuous_norm['Irr'] = irr_series
            self.df_continuous_denorm['Irr'] = irr_series
        
        print(f"\nDataFrames created for plotting")
    
    def save_metrics_report(self):
        report_path = self.output_dir / "metrics_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TEST EVALUATION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Model Information:\n")
            f.write(f"  Model class: {self.model_info['model_class']}\n")
            f.write(f"  Window: {self.model_info['dataset_info']['window']}\n")
            f.write(f"  Horizon: {self.model_info['dataset_info']['horizon']}\n")
            f.write(f"  Features: {self.model_info['data_config']['features']['feature_cols']}\n")
            f.write(f"  Target: {self.model_info['data_config']['features']['target_col']}\n\n")
            
            f.write("Test Data:\n")
            f.write(f"  Date range: {self.test_df.index.min()} to {self.test_df.index.max()}\n")
            f.write(f"  Sequences: {len(self.test_ds)} non-overlapping\n")
            f.write(f"  Data points: {self.metrics['Data_Points']}\n")
            f.write(f"  Bad days excluded: {self.metrics['Bad_Day_Excluded']}\n")
            f.write(f"  Total points: {self.metrics['Total_Points']}\n\n")
            
            f.write("Metrics:\n")
            f.write(f"  MAE:  {self.metrics['MAE']:.4f}\n")
            f.write(f"  RMSE: {self.metrics['RMSE']:.4f}\n")
            f.write(f"  R²:   {self.metrics['R2']:.4f}\n")
            f.write(f"  MAPE: {self.metrics['MAPE']:.2f}%\n")
            f.write(f"  Skill Score: {self.metrics['Skill_Score']:.4f}\n")
            f.write(f"  Directional Accuracy: {self.metrics['Directional_Accuracy']:.4f}\n")
            f.write(f"  Peak Error: {self.metrics['Peak_Error']:.4f}\n")
            f.write("=" * 80 + "\n")
        
        print(f"Metrics report saved to {report_path}")
    
    def save_predictions(self):
        pred_path = self.output_dir / "predictions.csv"
        self.df_continuous_norm.to_csv(pred_path)
        print(f"\nPredictions saved to {pred_path}")
        
        # Save denormalized
        pred_denorm_path = self.output_dir / "predictions_denormalized.csv"
        self.df_continuous_denorm.to_csv(pred_denorm_path)
        print(f"\nDenormalized predictions saved to {pred_denorm_path}")
    
    def save_plots(self, n_days: int = 10):
        print("Generating plots")
        print("-" * 100)
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Use denormalized data for plotting
        df_plot = self.df_continuous_denorm
        
        # 1. Overall time series plot
        fig, ax = plt.subplots(figsize=(15, 5))
        valid_mask = (df_plot['y_true'].notna() & df_plot['y_pred'].notna() & (df_plot['bad_day'] == 0))
        ax.plot(df_plot.index[valid_mask], df_plot.loc[valid_mask, 'y_true'], 
                'b-', linewidth=0.8, label='True', alpha=0.7)
        ax.plot(df_plot.index[valid_mask], df_plot.loc[valid_mask, 'y_pred'], 
                'r--', linewidth=0.8, label='Pred', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Target')
        ax.set_title('Test Set: True vs Predicted (Full Series)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "01_full_timeseries.png", dpi=150)
        plt.close()
        
        # 2. Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        y_true_valid = df_plot.loc[valid_mask, 'y_true'].values
        y_pred_valid = df_plot.loc[valid_mask, 'y_pred'].values
        ax.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=10)
        
        # Add diagonal line
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Test Set: Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "02_scatter.png", dpi=150)
        plt.close()
        
        # 3. Residuals
        fig, ax = plt.subplots(figsize=(15, 5))
        residuals = y_pred_valid - y_true_valid
        ax.plot(df_plot.index[valid_mask], residuals, 'g-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Residuals (Pred - True)')
        ax.set_title('Test Set: Residuals Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "03_residuals.png", dpi=150)
        plt.close()

        # 4. Error distribution histogram
        y_true_valid = df_plot.loc[valid_mask, 'y_true'].values
        y_pred_valid = df_plot.loc[valid_mask, 'y_pred'].values
        residuals = y_pred_valid - y_true_valid
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, density=False, alpha=0.7)
        residuals_mean = np.mean(residuals)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error (Perfect Prediction)')
        ax.axvline(x=residuals_mean, color='green', linestyle='--', linewidth=2, label=f'Mean Error ({residuals_mean:.4f})')
        ax.set_xlabel("Prediction Error (Pred - True)")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution Histogram")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(plots_dir / "04_error_histogram.png", dpi=150)
        plt.close()
        
        # 5. Daily plots
        print(f"\nGenerating {n_days} daily plots...")
        df_plot_with_date = df_plot.copy()
        df_plot_with_date['date'] = df_plot_with_date.index.date
        
        # Get unique dates with valid data
        valid_dates = (df_plot_with_date.groupby('date')['bad_day'].max())
        valid_dates = valid_dates[valid_dates == 0].index.tolist()

        if len(valid_dates) > 0:
            # Select evenly spaced dates
            n_dates = min(n_days, len(valid_dates))
            step = max(1, len(valid_dates) // n_dates)
            selected_dates = [valid_dates[i] for i in range(0, len(valid_dates), step)][:n_dates]
            
            for date in selected_dates:
                day_data = df_plot_with_date[df_plot_with_date['date'] == date]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot true and predicted
                ax.plot(day_data.index, day_data['y_true'], 'b-', linewidth=2, label='True', alpha=0.8)
                ax.plot(day_data.index, day_data['y_pred'], 'r--', linewidth=2, label='Pred', alpha=0.8)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Target')
                ax.set_title(f'Test Set: {date}')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=45)
                
                # Add Irr on secondary axis if available
                if 'Irr' in day_data.columns and day_data['Irr'].notna().any():
                    ax2 = ax.twinx()
                    ax2.plot(day_data.index, day_data['Irr'], color='orange', 
                            linewidth=1, alpha=0.5, label='Irr')
                    ax2.set_ylabel('Irr', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"day_{date}.png", dpi=150)
                plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def create_pdf_report(self):
        pdf_path = self.output_dir / "test_evaluation_report.pdf"
        
        print("\n" + "=" * 80)
        print("GENERATING PDF REPORT")
        print("=" * 80)
        
        # Use denormalized data for plotting
        df_plot = self.df_continuous_denorm
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title and Info
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('LSTM Test Evaluation Report', fontsize=16, fontweight='bold')
            
            info_text = []
            info_text.append("=" * 80)
            info_text.append("MODEL INFORMATION")
            info_text.append("=" * 80)
            info_text.append(f"Model class: {self.model_info['model_class']}")
            info_text.append(f"Window: {self.model_info['dataset_info']['window']}")
            info_text.append(f"Horizon: {self.model_info['dataset_info']['horizon']}")
            info_text.append(f"Hidden size: {self.model_info['architecture']['hidden_size']}")
            info_text.append(f"Num layers: {self.model_info['architecture']['num_layers']}")
            info_text.append(f"Features: {', '.join(self.model_info['data_config']['features']['feature_cols'])}")
            info_text.append(f"Target: {self.model_info['data_config']['features']['target_col']}")
            info_text.append("")
            info_text.append("=" * 80)
            info_text.append("TEST DATA")
            info_text.append("=" * 80)
            info_text.append(f"Date range: {self.test_df.index.min()} to {self.test_df.index.max()}")
            info_text.append(f"Sequences: {len(self.test_ds)} non-overlapping")
            info_text.append(f"Data points: {self.metrics['Data_Points']}")
            info_text.append(f"Bad days excluded: {self.metrics['Bad_Day_Excluded']}")
            info_text.append("")
            info_text.append("=" * 80)
            info_text.append("METRICS")
            info_text.append("=" * 80)
            info_text.append(f"MAE:  {self.metrics['MAE']:.4f}")
            info_text.append(f"RMSE: {self.metrics['RMSE']:.4f}")
            info_text.append(f"R²:   {self.metrics['R2']:.4f}")
            info_text.append(f"MAPE: {self.metrics['MAPE']:.2f}%")
            info_text.append(f"Skill Score: {self.metrics['Skill_Score']:.4f}")
            info_text.append(f"Directional Accuracy: {self.metrics['Directional_Accuracy']:.4f}")
            info_text.append(f"Peak Error: {self.metrics['Peak_Error']:.4f}")
            info_text.append("=" * 80)
            
            plt.text(0.1, 0.5, '\n'.join(info_text), transform=fig.transFigure,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Time series plot
            fig, ax = plt.subplots(figsize=(11, 8.5))
            valid_mask = (df_plot['y_true'].notna() & df_plot['y_pred'].notna() & (df_plot['bad_day'] == 0))
            ax.plot(df_plot.index[valid_mask], df_plot.loc[valid_mask, 'y_true'],
                   'b-', linewidth=0.8, label='True', alpha=0.7)
            ax.plot(df_plot.index[valid_mask], df_plot.loc[valid_mask, 'y_pred'],
                   'r--', linewidth=0.8, label='Pred', alpha=0.7)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Target', fontsize=12)
            ax.set_title('Test Set: True vs Predicted (Full Series)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Scatter and Residuals
            fig = plt.figure(figsize=(11, 8.5))
            gs = gridspec.GridSpec(2, 1, figure=fig)
            
            # Scatter
            ax1 = fig.add_subplot(gs[0, 0])
            y_true_valid = df_plot.loc[valid_mask, 'y_true'].values
            y_pred_valid = df_plot.loc[valid_mask, 'y_pred'].values
            ax1.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=10)
            min_val = min(y_true_valid.min(), y_pred_valid.min())
            max_val = max(y_true_valid.max(), y_pred_valid.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax1.set_xlabel('True Values', fontsize=10)
            ax1.set_ylabel('Predicted Values', fontsize=10)
            ax1.set_title('Scatter Plot', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Residuals
            ax2 = fig.add_subplot(gs[1, 0])
            residuals = y_pred_valid - y_true_valid
            ax2.plot(df_plot.index[valid_mask], residuals, 'g-', linewidth=0.5, alpha=0.7)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Time', fontsize=10)
            ax2.set_ylabel('Residuals', fontsize=10)
            ax2.set_title('Residuals Over Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 4: Error Distribution Histogram
            y_true_valid = df_plot.loc[valid_mask, 'y_true'].values
            y_pred_valid = df_plot.loc[valid_mask, 'y_pred'].values
            residuals = y_pred_valid - y_true_valid
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.hist(residuals, bins=50, density=False, alpha=0.75)
            residuals_mean = np.mean(residuals)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error (Perfect Prediction)')
            ax.axvline(x=residuals_mean, color='green', linestyle='--', linewidth=2, label=f'Mean Error ({residuals_mean:.4f})')
            ax.set_xlabel("Prediction Error (Pred - True)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Error Distribution Histogram", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Pages 5+: Daily plots (8 per page)
            df_plot_with_date = df_plot.copy()
            df_plot_with_date['date'] = df_plot_with_date.index.date
            valid_dates = (df_plot_with_date.groupby('date')['bad_day'].max())
            valid_dates = valid_dates[valid_dates == 0].index.tolist()
            
            if len(valid_dates) > 0:
                # Select evenly spaced dates (max 16 for 2 pages)
                n_dates = min(16, len(valid_dates))
                step = max(1, len(valid_dates) // n_dates)
                selected_dates = [valid_dates[i] for i in range(0, len(valid_dates), step)][:n_dates]
                
                # Create pages with 8 plots each
                for page_start in range(0, len(selected_dates), 8):
                    page_dates = selected_dates[page_start:page_start + 8]
                    
                    fig = plt.figure(figsize=(11, 8.5))
                    fig.suptitle(f'Daily Predictions (Page {page_start//8 + 1})', 
                               fontsize=14, fontweight='bold')
                    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
                    
                    for idx, date in enumerate(page_dates):
                        row = idx // 2
                        col = idx % 2
                        ax = fig.add_subplot(gs[row, col])
                        
                        day_data = df_plot_with_date[df_plot_with_date['date'] == date]
                        
                        ax.plot(day_data.index, day_data['y_true'], 'b-', 
                               linewidth=1.5, label='True', alpha=0.8)
                        ax.plot(day_data.index, day_data['y_pred'], 'r--', 
                               linewidth=1.5, label='Pred', alpha=0.8)
                        
                        ax.set_ylabel('Target', fontsize=8)
                        ax.tick_params(axis='both', labelsize=7)
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax.tick_params(axis='x', rotation=45)
                        ax.set_title(f'{date}', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        
                        if idx == 0:
                            ax.legend(loc='upper right', fontsize=7)
                        
                        # Add Irr on secondary y-axis if available
                        if 'Irr' in day_data.columns and day_data['Irr'].notna().any():
                            ax2 = ax.twinx()
                            ax2.plot(day_data.index, day_data['Irr'], color='orange',
                                   linewidth=1, alpha=0.5)
                            ax2.set_ylabel('Irr', fontsize=7, color='orange')
                            ax2.tick_params(axis='y', labelsize=6, labelcolor='orange')
                    
                    # Hide empty subplots
                    for idx in range(len(page_dates), 8):
                        row = idx // 2
                        col = idx % 2
                        ax = fig.add_subplot(gs[row, col])
                        ax.axis('off')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        print(f"PDF report saved to {pdf_path}")
    
    def evaluate(self, batch_size: int = 32, n_plot_days: int = 10):
        """
        Complete evaluation pipeline
        """
        print("\n" + "=" * 80)
        print("STARTING TEST EVALUATION")
        print("=" * 80)
        
        try:
            # Load model info
            self.load_model_info()
            
            # Load test data
            self.load_test_data()
            
            # Create device (force CPU)
            self.create_device()
            
            # Create and load model
            self.create_model()
            
            # Create dataset
            self.create_dataset()
            
            # Create dataloader
            self.create_dataloader(batch_size=batch_size)
            
            # Make predictions
            self.predict()
            
            # Build continuous series
            self.build_continuous_series()
            
            # Compute metrics
            self.compute_metrics()
            
            # Create DataFrames for plotting
            self.create_dataframe_for_plotting()
            
            # Save results
            self.save_metrics_report()
            self.save_predictions()
            self.save_plots(n_days=n_plot_days)
            
            # Create PDF report
            self.create_pdf_report()
            
            print("Evaluation completed")
            print("-" * 100)
            print(f"All results saved to: {self.output_dir}")
            print(f"\nFinal Metrics:")
            print(f"  MAE:  {self.metrics['MAE']:.4f}")
            print(f"  RMSE: {self.metrics['RMSE']:.4f}")
            print(f"  R²:   {self.metrics['R2']:.4f}")
            print(f"  MAPE: {self.metrics['MAPE']:.2f}%")
            print(f"  Skill Score: {self.metrics['Skill_Score']:.4f}")
            print(f"  Directional Accuracy: {self.metrics['Directional_Accuracy']:.4f}")
            print(f"  Peak Error: {self.metrics['Peak_Error']:.4f}")
            
            return {
                'metrics': self.metrics,
                'df_continuous_norm': self.df_continuous_norm,
                'df_continuous_denorm': self.df_continuous_denorm,
                'predictions': self.predictions,
                'ground_truth': self.ground_truth,
                'bad_day_masks': self.bad_day_masks
            }
            
        except Exception as e:
            print(f"\nEvaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_summary(self) -> str:
        """Generate a summary string of evaluation results"""
        if self.metrics is None:
            return "Evaluation not run yet."
        
        summary = []
        summary.append("=" * 80)
        summary.append("TEST EVALUATION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Model: {self.model_info['model_class']}")
        summary.append(f"Mode: {self.model_info['flag_mode']['description']}")
        summary.append(f"Test period: {self.test_df.index.min()} to {self.test_df.index.max()}")
        summary.append(f"Sequences: {len(self.test_ds)} non-overlapping")
        summary.append(f"Data points: {self.metrics['Data_Points']} (excluded: {self.metrics['Bad_Day_Excluded']})")
        summary.append("-" * 80)
        summary.append(f"MAE:  {self.metrics['MAE']:.4f}")
        summary.append(f"RMSE: {self.metrics['RMSE']:.4f}")
        summary.append(f"R²:   {self.metrics['R2']:.4f}")
        summary.append(f"MAPE: {self.metrics['MAPE']:.2f}%")
        summary.append(f"Skill Score: {self.metrics['Skill_Score']:.4f}")
        summary.append(f"Directional Accuracy: {self.metrics['Directional_Accuracy']:.4f}")
        summary.append(f"Peak Error: {self.metrics['Peak_Error']:.4f}")
        summary.append("=" * 80)
        
        return "\n".join(summary)


def evaluate_test_only(
    model_path: str,
    model_info_path: str,
    test_data_path: str,
    scaler_path: str,
    output_dir: str = None,
    batch_size: int = 32,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Convenience function for test-only evaluation.
    
    Args:
        model_path: Path to trained model (.pt file)
        model_info_path: Path to model_info.pkl
        test_data_path: Path to test_scaled.parquet
        scaler_path: Path to p_scaling.pkl
        output_dir: Directory to save evaluation results
        batch_size: Batch size for inference
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = LSTMEvaluator(
        model_path=model_path,
        model_info_path=model_info_path,
        test_data_path=test_data_path,
        scaler_path=scaler_path,
        output_dir=output_dir,
        random_seed=random_seed
    )
    
    results = evaluator.evaluate(batch_size=batch_size)
    
    # Print summary
    print("\n" + evaluator.generate_summary())
    
    return results


def main():
    MODEL_PATH = "lstm_run_2026_01_31_182240_si/lstm_results/31.01.2026.182441_model.pt"
    MODEL_INFO_PATH = "lstm_run_2026_01_31_182240_si/lstm_results/model_info.pkl"
    TEST_DATA_PATH = "lstm_run_2026_01_31_114538_psc/training_data/test_scaled.parquet"
    SCALER_PATH = "lstm_run_2026_01_31_182240_si/training_data/p_scaling.pkl"
    OUTPUT_DIR = "lstm_run_2026_01_31_182240_si/lstm_results/evaluationspsc"
    BATCH_SIZE = 32
    RANDOM_SEED = 42

    print("Running evaluator with specified paths:")
    print("Model:", MODEL_PATH)
    print("Model info:", MODEL_INFO_PATH)
    print("Test data:", TEST_DATA_PATH)
    print("Scaler:", SCALER_PATH)
    print("Output:", OUTPUT_DIR)

    results = evaluate_test_only(
        model_path=MODEL_PATH,
        model_info_path=MODEL_INFO_PATH,
        test_data_path=TEST_DATA_PATH,
        scaler_path=SCALER_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED
    )

    return results


if __name__ == "__main__":
    main()