#Version: 1.0 
#Handles only 1 output
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureEnggPipeline:
    """
        Class to create feature and scale for time-series PV power forecasting.
    """
    def __init__(
        self,
        out_dir: str = "./training_data",
        window: int = 48,
        horizon: int = 36,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.scaler_file = self.out_dir/"scalers.pkl"
        self.p_scaler_file = self.out_dir/"p_scaling.pkl"

        self.window = window
        self.horizon = horizon

        self.sensor_cols = ["Irr", "bad_day",]
        self.env_cols = ["temp_C", "humidity", "precip_mm", "precip_indicator", "cloud_cover",]
        self.engineered_cyclical = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "weekday_sin", "weekday_cos"]
        self.target_col = "P_normalised" #replace with TARGET set in lstm_pipeline_entry

    def print_stats(self, df: pd.DataFrame, cols: list, label: str):
        print(f"\nStats: {label}.")
        for col in cols:
            if col not in df.columns:
                print(f"{col:20s}: (missing)")
                continue

            series = df[col].dropna().astype(float)
            if series.empty:
                print(f"{col:20s}: (all NaN)")
            else:
                print(
                    f"{col:20s}: "
                    f"min={series.min():.6f},"
                    f"max={series.max():.6f},"
                    f"mean={series.mean():.6f}"
                )

    def load_input_csv(self, path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)

            required = ["_time", "P"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Missing required column: {c}")

            df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
            if df["_time"].isna().any():
                raise RuntimeError("Some _time rows could not be parsed as datetime.")

            df = df.sort_values("_time").set_index("_time")

            expected_freq = '10min'
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
            
            if len(df) == len(full_range) and df.index.isin(full_range).all():
                print(f"Timestamps are continuous with {expected_freq} frequency: YES")
            else:
                print(f"Timestamps are continuous with {expected_freq} frequency: NO")

            nan_counts = df.isna().sum()
            has_nans = nan_counts.any()

            if has_nans:
                print("-" * 100)
                print("NaN count per column:")
                nan_df = nan_counts[nan_counts > 0].to_frame(name='NaN Count')
                nan_df['Percentage'] = (nan_df['NaN Count'] / len(df) * 100).round(2)
                print(nan_df)
            else:
                print("\nInput dataframe has no NaNs.")
            
            print(f"Index timezone: {df.index.tz}")
            print(f"Index dtype: {df.index.dtype}")
            return df

    def add_time_features(self, df):
        print("\nAdding time features.")
        print("-" * 100)
        df = df.copy()
        df["hour"] = df.index.hour
        df["dayofyear"] = df.index.dayofyear
        df["month"] = df.index.month
        df["weekday"] = df.index.weekday

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        return df
    
    def retain_only_features_and_target(self, df):
        """
        Keep only feature columns + target
        """
        keep = (self.sensor_cols + self.env_cols + self.engineered_cyclical + [self.target_col])
        keep = [c for c in keep if c in df.columns]
        return df[keep]

    #NEW: date based splitting called if create_val=True
    def temporal_fixed_date_split(
        self,
        df: pd.DataFrame,
        start_date_train: str = "2024-11-28",
        end_date_train: str = "2025-10-28",
        start_date_test: str = "2025-10-29",
        end_date_test: str = "2025-12-31",
        val_fraction: float = 0.1,
        val_from_train_end: bool = True
    ):
        """
        Fixed date temporal split:
        - Train: start_date_train to end_date_train
        - Test: start_date_test to end_date_test
        - Validation: x% of train from behind (if val_fraction > 0)
        """
        df = df.copy().sort_index()        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        train_start = pd.to_datetime(start_date_train)
        train_end = pd.to_datetime(end_date_train)
        test_start = pd.to_datetime(start_date_test)
        test_end = pd.to_datetime(end_date_test)
        
        if df.index.tz is not None:
            df_tz = df.index.tz
            print(f"DataFrame has timezone: {df_tz}. Converting comparison dates to same timezone.")
            
            train_start = train_start.tz_localize(df_tz) if train_start.tz is None else train_start.tz_convert(df_tz)
            train_end = train_end.tz_localize(df_tz) if train_end.tz is None else train_end.tz_convert(df_tz)
            test_start = test_start.tz_localize(df_tz) if test_start.tz is None else test_start.tz_convert(df_tz)
            test_end = test_end.tz_localize(df_tz) if test_end.tz is None else test_end.tz_convert(df_tz)
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        
        # Split train into train and validation if val_fraction > 0
        val_df = None
        if val_fraction > 0:
            if val_from_train_end:
                # Take validation from the end of training period
                val_size = int(len(train_df) * val_fraction)
                val_df = train_df.iloc[-val_size:]
                train_df = train_df.iloc[:-val_size]
            else:
                # Take validation from the beginning of training period
                val_size = int(len(train_df) * val_fraction)
                val_df = train_df.iloc[:val_size]
                train_df = train_df.iloc[val_size:]
        
        def print_info(name, d):
            if len(d) == 0:
                print(f"{name}: 0 rows | No data")
                return
                
            print(
                f"{name}: {len(d)} rows | "
                f"{d.index.min().date()} → {d.index.max().date()} "
                f"({(d.index.max() - d.index.min()).days + 1} days)"
            )
            if "bad_day" in d.columns:
                print(f"  bad_day==1: {(d['bad_day']==1).sum()}")
            if "P" in d.columns:
                print(f"  P NaNs    : {d['P'].isna().sum()}")
        
        print("\nFixed date temporal split")
        print("-" * 100)
        print_info("TRAIN", train_df)
        if val_df is not None:
            print_info("VAL  ", val_df)
        print_info("TEST ", test_df)
        
        if val_df is not None:
            return train_df, val_df, test_df
        else:
            return train_df, test_df
    
    #NEW called if create_val=False, val will be created upsteam
    def temporal_fixed_date_split_train_test(
        self,
        df: pd.DataFrame,
        start_date_train: str = "2024-11-28", #"2024-08-16", "2024-11-28",
        end_date_train: str = "2025-10-28", #"2025-07-15", "2025-10-28",
        start_date_test: str = "2025-10-29", #"2025-07-16", "2025-10-29",
        end_date_test: str = "2025-12-31", #"2025-09-14", "2025-12-31",
    ):

        """
        Fixed date temporal split (train and test only):
        - Train: start_date_train to end_date_train
        - Test: start_date_test to end_date_test
        """
        df = df.copy().sort_index()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        train_start = pd.to_datetime(start_date_train)
        train_end = pd.to_datetime(end_date_train)
        test_start = pd.to_datetime(start_date_test)
        test_end = pd.to_datetime(end_date_test)
        
        if df.index.tz is not None:
            df_tz = df.index.tz
            print(f"DataFrame has timezone: {df_tz}. Converting comparison dates to same timezone.")
            
            train_start = train_start.tz_localize(df_tz) if train_start.tz is None else train_start.tz_convert(df_tz)
            train_end = train_end.tz_localize(df_tz) if train_end.tz is None else train_end.tz_convert(df_tz)
            test_start = test_start.tz_localize(df_tz) if test_start.tz is None else test_start.tz_convert(df_tz)
            test_end = test_end.tz_localize(df_tz) if test_end.tz is None else test_end.tz_convert(df_tz)
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        
        def print_info(name, d):
            if len(d) == 0:
                print(f"{name}: 0 rows | No data")
                return
                
            print(
                f"{name}: {len(d)} rows | "
                f"{d.index.min().date()} → {d.index.max().date()} "
                f"({(d.index.max() - d.index.min()).days + 1} days)"
            )
            if "bad_day" in d.columns:
                print(f"  bad_day==1: {(d['bad_day']==1).sum()}")
            if "P" in d.columns:
                print(f"  P NaNs    : {d['P'].isna().sum()}")
        
        print("\nFixed date temporal split (train/test only)")
        print_info("TRAIN", train_df)
        print_info("TEST ", test_df)
        
        return train_df, test_df
    
    def scale_features_and_save(
            self, 
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame | None,
            sensor_cols: list,
            env_cols: list,
            engineered_cyclical: list,
            target_col: str,
            scaler_file: Path,
            p_scaler_file: Path
            ):
        """
            Fit MinMax scalers on train (sensor_cols + env_cols),
            Fit MinMax scaler for target on train,
            Transform train/val/test accordingly,
            Not fit/transform engineered_cyclical,
            Save:
            - scaler_file
            - p_scaler_file
            Returns transformed dataframes and scalers dicts.
        """
        # define which features use StandardScaler so they are not scaled using MinMaxScaler; for future currently not important
        neg_scaler_cols = ["P_Irr_ratio"] 

        all_feature_cols = list(dict.fromkeys(sensor_cols + env_cols)) 
        missing = [c for c in all_feature_cols if c not in train_df.columns]
        if missing:
            print(f"Missing columns in train for scaling: {missing}")
            all_feature_cols = [c for c in all_feature_cols if c in train_df.columns]

        dfs_for_stats = [train_df, val_df]
        if test_df is not None:
            dfs_for_stats.append(test_df)

        dfs_non_empty = [
            df for df in dfs_for_stats 
            if df is not None and not df.empty and not df.dropna(how="all").empty
        ]

        self.print_stats(pd.concat(dfs_non_empty, ignore_index=True), all_feature_cols + [target_col] + engineered_cyclical, "Before scaling (all splits concatenated)",)
        print("\n")
        feature_scalers = {}
        for col in all_feature_cols:
            col_data = train_df[[col]].astype(float)
            if col_data.dropna().empty:
                print(f"Train has no valid data for {col}, skipping scaler for this feature.")
                continue
            if col in neg_scaler_cols:
                scaler = StandardScaler()
                print(f"Using StandardScaler for {col}")
            else:
                scaler = MinMaxScaler()
                print(f"Using MinMaxScaler for {col}")

            scaler.fit(col_data)
            feature_scalers[col] = scaler
            if isinstance(scaler, MinMaxScaler):
                print(f"Fitted MinMax: data_min={scaler.data_min_[0]:.6f}, data_max={scaler.data_max_[0]:.6f}")
            else:
                print(f"Fitted StandardScaler: mean={scaler.mean_[0]:.6f}, std={scaler.scale_[0]:.6f}")

        p_scaler = MinMaxScaler()
        p_scaler.fit(train_df[[target_col]].astype(float))
        p_info = {
            "scaler": p_scaler,
            "data_min": float(p_scaler.data_min_[0]),
            "data_max": float(p_scaler.data_max_[0])
        }
        print(f"Fitted P scaler: data_min={p_info['data_min']:.6f}, data_max={p_info['data_max']:.6f}")

        # Transform datasets using fitted scalers (train fitted then transform train/val/test)
        def transform_df(df, feature_scalers, p_scaler, target_col):
            df_t = df.copy()
            for col, scaler in feature_scalers.items():
                if col in df_t.columns:
                    col_vals = df_t[[col]].astype(float)
                    mask = col_vals.notna().values.ravel()
                    if mask.any():
                        transformed = scaler.transform(col_vals[mask])
                        df_t.loc[col_vals.index[mask], col] = transformed.flatten()
            # Transform P target
            if target_col in df_t.columns:
                p_vals = df_t[[target_col]].astype(float)
                mask = p_vals.notna().values.ravel()
                if mask.any():
                    df_t.loc[p_vals.index[mask], target_col] = (p_scaler.transform(p_vals[mask])[:, 0])
            return df_t

        train_scaled = transform_df(train_df, feature_scalers, p_scaler, target_col)
        val_scaled   = transform_df(val_df, feature_scalers, p_scaler, target_col)
        test_scaled = (transform_df(test_df, feature_scalers, p_scaler, target_col) if test_df is not None else None)

        #Save feature scalers (sensor+env)
        scaler_payload = {
            "feature_scalers": feature_scalers,
            "feature_list": list(feature_scalers.keys()),
        }
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler_payload, f)
        print(f"\nSaved feature scalers to: {scaler_file}")

        with open(p_scaler_file, "wb") as f:
            pickle.dump(p_info, f)
        print(f"Saved P scaler info to: {p_scaler_file}")

        print("\nAfter transform (TRAIN)")
        self.print_stats(train_scaled, all_feature_cols + [target_col], "TRAIN (post-transform)")
        self.print_stats(train_scaled, engineered_cyclical, "TRAIN engineered_cyclical (unscaled)")

        print("\nAfter transform (VAL)")
        self.print_stats(val_scaled, all_feature_cols + [target_col], "VAL (post-transform)")
        self.print_stats(val_scaled, engineered_cyclical, "VAL engineered_cyclical (unscaled)")

        if test_scaled is not None:
            print("\nAfter transform (TEST)")
            self.print_stats(test_scaled, all_feature_cols + [target_col], "TEST (post-transform)")
            self.print_stats(test_scaled, engineered_cyclical, "TEST engineered_cyclical (unscaled)")

        return train_scaled, val_scaled, test_scaled, scaler_payload, p_info

    def plot_correlation_matrix(
        self, df: pd.DataFrame, exclude_cols: list = None, save_path: str = "correlation_matrix.png", title: str = "Correlation Matrix"):
        if exclude_cols is None:
            exclude_cols = []
        cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        if not cols:
            print("No numeric columns available for correlation")
            return

        sub = df[cols]
        sub = sub.dropna(axis=1, how="all")
        sub = sub.loc[:, sub.std(numeric_only=True) > 0]
        sub = sub.dropna()

        if sub.shape[0] < 2 or sub.shape[1] < 2:
            print("Not enough valid data for correlation matrix")
            return

        corr = sub.corr()

        # final safety check
        if np.isnan(corr.values).all():
            print("Correlation matrix is all-NaN, skipping plot")
            return

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved correlation matrix to {save_path}")

    def process_pipeline(self, input_csv: Path, create_val: bool = False,):
        df = self.load_input_csv(input_csv)
        df = self.add_time_features(df)
        for col_group, cols in [("sensor_cols", self.sensor_cols), ("env_cols", self.env_cols), ("engineered", self.engineered_cyclical)]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"Missing columns in {col_group}: {missing}.")

        #Remove columns other than feature list and target
        df = self.retain_only_features_and_target(df)

        #Split
        if create_val:
            train_df, val_df, test_df = self.temporal_fixed_date_split(df)
        else:
            train_df, test_df = self.temporal_fixed_date_split_train_test(df)
            val_df = pd.DataFrame(columns=train_df.columns)

        #Fit scalers on train and transform all splits (engineered cyclical not scaled)
        train_scaled, val_scaled, test_scaled, scaler_payload, p_info = self.scale_features_and_save(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            sensor_cols=self.sensor_cols,
            env_cols=self.env_cols,
            engineered_cyclical=self.engineered_cyclical,
            target_col=self.target_col,
            scaler_file=self.scaler_file,
            p_scaler_file=self.p_scaler_file
        )

        #Save final scaled dataframes 
        train_scaled.to_parquet(self.out_dir / "train_scaled.parquet")
        val_scaled.to_parquet(self.out_dir / "val_scaled.parquet")
        if test_scaled is not None: test_scaled.to_parquet(self.out_dir / "test_scaled.parquet")
        print(f"\nSaved scaled splits to {self.out_dir}")

        self.plot_correlation_matrix(train_scaled, exclude_cols=self.engineered_cyclical, save_path=self.out_dir / "train_correlation_matrix.png", title="Train Set Correlation")
        self.plot_correlation_matrix(val_scaled, exclude_cols=self.engineered_cyclical,save_path=self.out_dir / "val_correlation_matrix.png", title="Val Set Correlation")
        if test_scaled is not None: self.plot_correlation_matrix(test_scaled, exclude_cols=self.engineered_cyclical, save_path=self.out_dir / "test_correlation_matrix.png",title="Test Set Correlation")

        return train_scaled, val_scaled, test_scaled, scaler_payload, p_info

