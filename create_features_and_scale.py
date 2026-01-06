from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

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

        self.sensor_cols = ["Irr", "bad_day", "Irr_lag6", "Irr_lag12", "Irr_lag24",]
        self.env_cols = ["temp_C", "humidity", "precip_mm", "precip_indicator", "cloud_cover", "humidity_lag6", "humidity_lag12", "humidity_lag24"]
        self.engineered_cyclical = ["hour_sin", "hour_cos",]
        self.target_col = "P"

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
                print("NaN count per column:")
                nan_df = nan_counts[nan_counts > 0].to_frame(name='NaN Count')
                nan_df['Percentage'] = (nan_df['NaN Count'] / len(df) * 100).round(2)
                print(nan_df)
            else:
                print("\nInput dataframe has no NaNs.")
            return df

    def add_time_features(self, df):
        print("\nAdding time features.")
        df = df.copy()
        df["hour"] = df.index.hour
        df["dayofyear"] = df.index.dayofyear
        df["month"] = df.index.month

        unique_days = df.index.normalize().unique()
        day_of_dataset_map = {day: i+1 for i, day in enumerate(unique_days)}
        df["day_of_dataset"] = df.index.normalize().map(day_of_dataset_map)
        df["week_of_dataset"] = ((df["day_of_dataset"] - 1) // 7) + 1

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        total_days = df["day_of_dataset"].max()
        df["day_of_dataset_sin"] = np.sin(2 * np.pi * df["day_of_dataset"] / total_days)
        df["day_of_dataset_cos"] = np.cos(2 * np.pi * df["day_of_dataset"] / total_days)

        total_weeks = df["week_of_dataset"].max()
        df["week_of_dataset_sin"] = np.sin(2 * np.pi * df["week_of_dataset"] / total_weeks)
        df["week_of_dataset_cos"] = np.cos(2 * np.pi * df["week_of_dataset"] / total_weeks)

        return df.drop(columns=["hour", "dayofyear", "month", "day_of_dataset", "week_of_dataset"])
    
    def add_power_features(
        self,
        df,
        window: int = 3,
        night_start: int = 22,
        night_end: int = 4,
    ):
        """
        - P_diff = P(t) - P(t-1)
        - P_smooth = rolling mean of P
        Night-time values are set to zero.
        """

        print("\nAdding power features.")
        df = df.copy().sort_index()

        if "P" not in df.columns:
            raise RuntimeError("P column missing")
        if "Irr" not in df.columns:
            raise RuntimeError("Irr column missing")
        df["_hour"] = df.index.hour

        night_mask = (df["_hour"] >= night_start) | (df["_hour"] < night_end)

        df["P_diff"] = df["P"].diff(1)
        
        # Invalidate night + NaN
        df.loc[night_mask | df["P_diff"].isna(), "P_diff"] = 0.0
        P_day = df["P"].where(~night_mask)
        df["P_smooth"] = (
            P_day
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        df.loc[night_mask, "P_smooth"] = 0.0

        print("\nStats: power features")
        print(f"Total rows: {len(df)}")
        print("\n[P_diff]")
        print(f"Zero entries: {(df['P_diff'] == 0).sum()}")
        print("\n[P_smooth]")
        print(f"NaN entries:  {df['P_smooth'].isna().sum()}")
        print(f"Zero entries: {(df['P_smooth'] == 0).sum()}")

        df = df.drop(columns=["_hour"])
        return df

    def add_lag_features(
        self,
        df,
        p_cols="Irr",        
        lags=[6, 9, 12],
        bad_day_col="bad_day",
        night_start=22,
        night_end=4,
    ):
        """
        within same day
        Lag values zeroed if within night hours (22:00 → 04:00)
        """
        print("\nAdding lag features.")
        df = df.copy().sort_index()
        if isinstance(p_cols, str):
            p_cols = [p_cols]
        df["_date"] = df.index.normalize()
        df["_hour"] = df.index.hour

        night_mask = (df["_hour"] >= night_start) | (df["_hour"] < night_end)

        for col in p_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            for lag in lags:
                lag_name = f"{col}_lag{lag}"

                df[lag_name] = (
                    df.groupby("_date")[col]
                    .shift(lag)
                )

                invalid_mask = (
                    night_mask |
                    (df[lag_name].isna())
                )

                df.loc[invalid_mask, lag_name] = 0.0

        df = df.drop(columns=["_date", "_hour"])
        return df
    
    def retain_only_features_and_target(self, df):
        """
        Keep only feature columns + target
        """
        keep = (
            self.sensor_cols
            + self.env_cols
            + self.engineered_cyclical
            + [self.target_col]
        )
        keep = [c for c in keep if c in df.columns]
        return df[keep]

    def seasonal_temporal_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
    ):
        """
            - Split by month
            - Inside each month, split by full days
        """
        df = df.copy().sort_index()
        df["date"] = df.index.normalize()
        df["month_for_split"] = df.index.month

        train_days, val_days, test_days = [], [], []

        for month in sorted(df["month_for_split"].unique()):
            month_days = (
                df.loc[df["month_for_split"] == month, "date"]
                .drop_duplicates()
                .sort_values()
            )

            n_days = len(month_days)
            if n_days == 0:
                continue

            n_train = int(np.floor(train_frac * n_days))
            n_val = int(np.floor(val_frac * n_days))

            train_days.extend(month_days[:n_train])
            val_days.extend(month_days[n_train:n_train + n_val])
            test_days.extend(month_days[n_train + n_val:])

        train = df[df["date"].isin(train_days)].drop(columns=["date", "month_for_split"])
        val   = df[df["date"].isin(val_days)].drop(columns=["date", "month_for_split"])
        test  = df[df["date"].isin(test_days)].drop(columns=["date", "month_for_split"])

        def print_split_info(split_df, name):
            if split_df.empty:
                print(f"{name}: empty")
                return

            print(
                f"{name}: {len(split_df)} rows | "
                f"{split_df.index.min()} → {split_df.index.max()}"
            )
            print(f"bad_day == 1 : {(split_df.get('bad_day') == 1).sum() if 'bad_day' in split_df else 'N/A'}")
            print(f"NaT rows     : {split_df.index.isna().sum()}")
            print(f"P is NaN     : {split_df['P'].isna().sum() if 'P' in split_df else 'N/A'}")

        print("\nSeasonal temporal split")
        print_split_info(train, "TRAIN")
        print_split_info(val, "VAL")
        print_split_info(test, "TEST")

        return train, val, test
        
    def temporal_split_full(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.7,
        val_frac: float = 0.2,
        test_frac: float = 0.1,
    ):
        """
            - Split entire df
        """
        df = df.copy().sort_index()
        df["date"] = df.index.normalize()

        unique_days = df["date"].drop_duplicates().sort_values()
        n_days = len(unique_days)

        n_train = int(np.floor(train_frac * n_days))
        n_val = int(np.floor(val_frac * n_days))

        train_days = unique_days[:n_train]
        val_days = unique_days[n_train:n_train + n_val]
        test_days = unique_days[n_train + n_val:]

        train_df = df[df["date"].isin(train_days)].drop(columns=["date"])
        val_df   = df[df["date"].isin(val_days)].drop(columns=["date"])
        test_df  = df[df["date"].isin(test_days)].drop(columns=["date"])

        def print_split_info(split_df, name):
            if split_df.empty:
                print(f"{name}: empty")
                return

            print(
                f"{name}: {len(split_df)} rows | "
                f"{split_df.index.min()} → {split_df.index.max()}"
            )
            print(f"bad_day == 1 : {(split_df.get('bad_day') == 1).sum() if 'bad_day' in split_df else 'N/A'}")
            print(f"NaT rows     : {split_df.index.isna().sum()}")
            print(f"P is NaN     : {split_df['P'].isna().sum() if 'P' in split_df else 'N/A'}")

        print("\nFull temporal split")
        print_split_info(train_df, "TRAIN")
        print_split_info(val_df, "VAL")
        print_split_info(test_df, "TEST")

        return train_df, val_df, test_df

    def scale_features_and_save(
            self, 
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame,
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

        self.print_stats(pd.concat([train_df, val_df, test_df]), all_feature_cols + [target_col] + engineered_cyclical, "Before scaling (all splits concatenated)")
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

        # Transform datasets using fitted scalers (train fitted > transform train/val/test)
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
        test_scaled  = transform_df(test_df, feature_scalers, p_scaler, target_col)

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

        print("\nAfter transform (TEST)")
        self.print_stats(test_scaled, all_feature_cols + [target_col], "TEST (post-transform)")
        self.print_stats(test_scaled, engineered_cyclical, "TEST engineered_cyclical (unscaled)")

        return train_scaled, val_scaled, test_scaled, scaler_payload, p_info

    def plot_correlation_matrix(self, df: pd.DataFrame, exclude_cols: list = None, save_path: str = "correlation_matrix.png", title: str = "Correlation Matrix"):
        if exclude_cols is None:
            exclude_cols = []
        cols = [c for c in df.columns if c not in exclude_cols]
        corr = df[cols].dropna().corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def process_pipeline(self, input_csv: Path):
        df = self.load_input_csv(input_csv)
        df = self.add_time_features(df)
        df = self.add_power_features(df, window=3)
        #adding lag features to provide explicit cyclical references (6, 12, 24 hours) 
        #helps the LSTM model learn temporal dependencies more directly and reliably
        df = self.add_lag_features(df, p_cols=["Irr", "temp_C", "humidity"], lags=[6, 12, 24])

        for col_group, cols in [("sensor_cols", self.sensor_cols), ("env_cols", self.env_cols), ("engineered", self.engineered_cyclical)]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"Missing columns in {col_group}: {missing}.")

        #Remove columns other than feature list and target
        df = self.retain_only_features_and_target(df)

        #Split
        train_df, val_df, test_df = self.seasonal_temporal_split(df) # Or use temporal_split_full 
        
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
        test_scaled.to_parquet(self.out_dir / "test_scaled.parquet")
        print(f"\nSaved scaled splits to {self.out_dir}")

        self.plot_correlation_matrix(train_scaled, exclude_cols=self.engineered_cyclical, save_path=self.out_dir / "train_correlation_matrix.png", title="Train Set Correlation")
        self.plot_correlation_matrix(val_scaled, exclude_cols=self.engineered_cyclical,save_path=self.out_dir / "val_correlation_matrix.png", title="Val Set Correlation")
        self.plot_correlation_matrix(test_scaled, exclude_cols=self.engineered_cyclical, save_path=self.out_dir / "test_correlation_matrix.png",title="Test Set Correlation")

        return train_scaled, val_scaled, test_scaled, scaler_payload, p_info

