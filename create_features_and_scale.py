from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

MODULE_CONFIG = {
    'P_solon_1':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_solon_2':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_solon_3':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_solon_5':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_solon_6':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_solon_7':      {'install_date': '2024-07-26', 'rated_wp': 250, 'family': 'solon'},
    'P_sanyo_2':      {'install_date': '2024-07-26', 'rated_wp': 200, 'family': 'sanyo'},
    'P_sanyo_3':      {'install_date': '2024-07-26', 'rated_wp': 200, 'family': 'sanyo'},
    'P_sanyo_4':      {'install_date': '2024-07-26', 'rated_wp': 200, 'family': 'sanyo'},
    'P_sanyo_5':      {'install_date': '2024-07-26', 'rated_wp': 200, 'family': 'sanyo'},
    'P_perovskite_1': {'install_date': '2024-11-28', 'rated_wp': 110, 'family': 'perovskite'},
    'P_perovskite_2': {'install_date': '2024-11-28', 'rated_wp': 110, 'family': 'perovskite'},
}

class FeatureEnggPipeline:
    """
    Class to create features and scale for time-series PV power forecasting.
    Supports multiple target columns and one-hot encoding for categorical columns.
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
        self.p_scaler_file = self.out_dir/"p_scalers.pkl"

        self.window = window
        self.horizon = horizon

        self.sensor_cols = ["Irr",
                            #"bad_day_sanyo_2", "bad_day_sanyo_3", "bad_day_sanyo_4", "bad_day_sanyo_5",
                            #"bad_day_solon_1", "bad_day_solon_2", "bad_day_solon_3","bad_day_solon_5", "bad_day_solon_6", "bad_day_solon_7",
                            "bad_day_perovskite_1", #"bad_day_perovskite_2",
                            ]
        self.env_cols = ["temp_C", "humidity", "precip_mm", "precip_indicator", "cloud_cover",]
        self.engineered_cyclical = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "weekday_sin", "weekday_cos",]
        self.target_cols = [#"P_sanyo_2", "P_sanyo_3", "P_sanyo_4", "P_sanyo_5",
                            #"P_solon_1", "P_solon_2", "P_solon_3", "P_solon_5", "P_solon_6", "P_solon_7",
                            "P_perovskite_1", #"P_perovskite_2",
                            ]

        #populated by add_age_features()
        self.age_cols: list[str] = []

    def print_stats(self, df: pd.DataFrame, cols: list, label: str, eps=1e-3):
        """
        Print detailed statistics for specified columns in a DataFrame.
        """
        print(f"\nStats: {label}")
        
        for col in cols:
            if col not in df.columns:
                print(f"{col:25s}: (missing)")
                continue
            
            series = df[col].dropna().astype(float)
            if series.empty:
                print(f"{col:25s}: (all NaN)")
                continue
            
            values = series.values
            
            # Basic stats
            mn, mx = values.min(), values.max()
            mean, std = values.mean(), values.std()
            
            print(f"\n{col:25s}")
            print(f"  min/max     : {mn:.4f} / {mx:.4f}")
            print(f"  mean/std    : {mean:.4f} / {std:.4f}")

    def load_input_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        if "_time" not in df.columns:
            raise ValueError("Missing required column: _time")

        df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
        if df["_time"].isna().any():
            raise RuntimeError("Some _time rows could not be parsed as datetime.")

        df = df.sort_values("_time").set_index("_time")

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10min")
        continuous = len(df) == len(full_range) and df.index.isin(full_range).all()
        print(f"Timestamps continuous at 10min: {'YES' if continuous else 'NO'}")

        nan_counts = df.isna().sum()
        if nan_counts.any():
            print("\nNaN count per column:")
            nan_df = nan_counts[nan_counts > 0].to_frame(name="NaN Count")
            nan_df["Percentage"] = (nan_df["NaN Count"] / len(df) * 100).round(2)
            print(nan_df)
        else:
            print("\nInput dataframe has no NaNs.")

        print(f"Index timezone : {df.index.tz}")
        print(f"Index dtype    : {df.index.dtype}")
        print(f"Detected targets: {[c for c in self.target_cols if c in df.columns]}")
        return df

    def add_time_features(self, df):
        print("\nAdding time features.")
        df = df.copy()
        df["hour"] = df.index.hour
        df["dayofyear"] = df.index.dayofyear
        df["month"] = df.index.month
        df["weekday"] = df.index.weekday

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

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        total_days = df["day_of_dataset"].max()
        df["day_of_dataset_sin"] = np.sin(2 * np.pi * df["day_of_dataset"] / total_days)
        df["day_of_dataset_cos"] = np.cos(2 * np.pi * df["day_of_dataset"] / total_days)

        total_weeks = df["week_of_dataset"].max()
        df["week_of_dataset_sin"] = np.sin(2 * np.pi * df["week_of_dataset"] / total_weeks)
        df["week_of_dataset_cos"] = np.cos(2 * np.pi * df["week_of_dataset"] / total_weeks)

        return df

    def _get_target_families(self, module_config: dict) -> dict[str, list[str]]:
        """
        Map each family to its list of target modules. Only includes targets that are in self.target_cols.
        
        Example:
            Input: MODULE_CONFIG with solon, sanyo, perovskite modules
            Output: {
                'silicon': ['P_solon_1', 'P_solon_2', ..., 'P_sanyo_2', ...],
                'perovskite': ['P_perovskite_1', 'P_perovskite_2']
            }
        """
        family_to_targets = {}
        
        for target, cfg in module_config.items():
            if target not in self.target_cols:
                continue  #skip if not in active targets
            
            family = cfg.get('family', 'unknown')
            
            # Group solon + sanyo as 'silicon'; perovskite stays as 'perovskite'
            if family in ['solon', 'sanyo']:
                group_key = 'silicon'
            elif family == 'perovskite':
                group_key = 'perovskite'
            else:
                group_key = family
            
            if group_key not in family_to_targets:
                family_to_targets[group_key] = []
            family_to_targets[group_key].append(target)
        
        return family_to_targets

    def add_age_features(
        self,
        df: pd.DataFrame,
        module_config: dict,
    ) -> pd.DataFrame:
        """
        Add age features grouped by technology family.
        
        If all targets belong to a single family (e.g., all silicon or all perovskite), create one age feature: days_since_install_
        If targets span multiple families (e.g., both silicon and perovskite), create one age feature per family: days_since_install_silicon, days_since_install_perovskite
        
        Pre-install rows are masked in corresponding bad_day_ columns.
        """
        print("\nAdding family based age features.")
        print("-" * 80)
        df = df.copy()
        tz = df.index.tz
        step = 10 / (60 * 24)

        self.age_cols = []

        #Determine target family and install dates
        family_to_targets = self._get_target_families(module_config)
        family_to_install_date = {}
        
        for family, targets in family_to_targets.items():
            if targets:
                target = targets[0]
                family_to_install_date[family] = module_config[target]['install_date']

        print(f"Target families detected: {list(family_to_targets.keys())}")
        print(f"silicon: {family_to_targets.get('silicon', [])}")
        print(f"perovskite: {family_to_targets.get('perovskite', [])}")

        #Create one age feature per family
        for family, install_date_str in family_to_install_date.items():
            col_name = f"days_since_install_{family}"
            
            install = pd.Timestamp(install_date_str)
            if tz is not None and install.tzinfo is None:
                install = install.tz_localize(tz)
            elif tz is None and install.tzinfo is not None:
                install = install.tz_localize(None)
            
            # Fractional days since install
            elapsed_days = (df.index - install).total_seconds() / 86400.0
            elapsed_days = np.clip(elapsed_days.astype(np.float32), 0.0, None)
            df[col_name] = elapsed_days
            
            self.age_cols.append(col_name)
            
            #Create/update bad_day_ column for pre-install masking
            bad_col = f"bad_day_{family}"
            pre_install = df.index < install
            n_pre = int(pre_install.sum())
            
            if bad_col in df.columns:
                df.loc[pre_install, bad_col] = 1.0
                if n_pre:
                    print(f"  {col_name:40s}: range [{elapsed_days.min():.1f}, "
                          f"{elapsed_days.max():.1f}] days  |  "
                          f"pre-install rows masked in {bad_col}: {n_pre}")
            else:
                # Create the bad_day column and register it
                df[bad_col] = np.where(pre_install, 1.0, 0.0)
                if bad_col not in self.sensor_cols:
                    self.sensor_cols.append(bad_col)
                print(f"  {col_name:40s}: range [{elapsed_days.min():.1f}, "
                      f"{elapsed_days.max():.1f}] days  |  "
                      f"created {bad_col} ({n_pre} pre-install rows masked)")
        
        print(f"\nAge columns added ({len(self.age_cols)}): {self.age_cols}")
        return df

    def retain_only_features_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only feature columns + all target columns.
        """
        # Combine all feature columns
        keep = (
            self.sensor_cols
            + self.env_cols
            + self.engineered_cyclical
            + self.age_cols 
            + self.target_cols
        )
        keep = [c for c in keep if c in df.columns]
        print(f"\nRetaining {len(keep)} columns for modelling")
        print(f"  Targets : {[c for c in self.target_cols if c in keep]}")
        print(f"  Age cols: {[c for c in self.age_cols    if c in keep]}")
        print(f"  Features: {[c for c in keep if c not in self.target_cols]}")
        return df[keep]

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
        - Train: start_date_train - end_date_train
        - Test: start_date_test - end_date_test
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
        
        def print_info(self, name, d):
            if len(d) == 0:
                print(f"{name}: 0 rows | No data")
                return
            
            print(
                f"{name}: {len(d)} rows | "
                f"{d.index.min().date()} → {d.index.max().date()} "
                f"({(d.index.max() - d.index.min()).days + 1} days)"
            )
            
            # Print stats for all bad_day columns
            bad_day_cols = [col for col in d.columns if col.startswith('bad_day_')]
            for bad_col in bad_day_cols:
                if bad_col in d.columns:
                    print(f"  {bad_col}==1: {(d[bad_col]==1).sum()}")
            
            # Print NaN counts for each target column
            for target_col in self.target_cols:
                if target_col in d.columns:
                    print(f"  {target_col} NaNs: {d[target_col].isna().sum()}")
        
        print("\nFixed date temporal split")
        print_info("TRAIN", train_df)
        if val_df is not None:
            print_info("VAL  ", val_df)
        print_info("TEST ", test_df)
        
        if val_df is not None:
            return train_df, val_df, test_df
        else:
            return train_df, test_df
    
    def temporal_fixed_date_split_train_test(
        self,
        df: pd.DataFrame,
        start_date_train: str = "2024-11-28", #"2024-11-28" for psc and #"2024-08-16" for si
        end_date_train: str = "2025-10-28",
        start_date_test: str = "2025-10-29",
        end_date_test: str = "2026-06-18", #"2025-12-31", 
    ):
        """
        Fixed date temporal split (train and test only):
        - Train: start_date_train - end_date_train
        - Test: start_date_test - end_date_test
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
            
            # Print NaN counts for each target column
            for target_col in self.target_cols:
                if target_col in d.columns:
                    print(f"  {target_col} NaNs: {d[target_col].isna().sum()}")
        
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
        age_cols: list,
        engineered_cyclical: list,
        target_cols: list,
        scaler_file: Path,
        p_scaler_file: Path
    ):
        """
        MinMax-scale sensor_cols + env_cols + age_cols on train, then transform all splits.  
        engineered_cyclical and target_cols are handled separately
        (cyclical left unscaled; each target gets its own MinMaxScaler).
        """
        not_to_scale   = set(engineered_cyclical + target_cols)
        feature_cols   = [
            c for c in list(dict.fromkeys(sensor_cols + env_cols + age_cols))
            if c not in not_to_scale
        ]
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            print(f"WARNING — columns missing from train, skipping: {missing}")
            feature_cols = [c for c in feature_cols if c in train_df.columns]

        all_splits = [train_df, val_df] + ([test_df] if test_df is not None else [])
        self.print_stats(
            pd.concat(all_splits),
            feature_cols + target_cols + engineered_cyclical,
            "Before scaling (all splits concatenated)",
        )

        feature_scalers: dict[str, MinMaxScaler] = {}
        for col in feature_cols:
            data = train_df[[col]].astype(float)
            if data.dropna().empty:
                print(f"  Skipping {col} — no valid train data.")
                continue
            sc = MinMaxScaler()
            sc.fit(data)
            feature_scalers[col] = sc
            print(f"  MinMax {col:45s}: "
                  f"[{sc.data_min_[0]:.4f}, {sc.data_max_[0]:.4f}]")

        target_scalers: dict[str, MinMaxScaler] = {}
        for tc in target_cols:
            if tc not in train_df.columns:
                print(f"  WARNING: target '{tc}' not in train — skipping scaler.")
                continue
            data = train_df[[tc]].astype(float)
            if data.dropna().empty:
                print(f"  WARNING: no valid data for '{tc}' — skipping scaler.")
                continue
            sc = MinMaxScaler()
            sc.fit(data)
            target_scalers[tc] = sc
            print(f"  MinMax target {tc:38s}: "
                  f"[{sc.data_min_[0]:.4f}, {sc.data_max_[0]:.4f}]")

        def _transform(df):
            out = df.copy()
            for col, sc in feature_scalers.items():
                if col not in out.columns:
                    continue
                vals = out[[col]].astype(float)
                mask = vals.notna().values.ravel()
                if mask.any():
                    out.loc[vals.index[mask], col] = sc.transform(vals[mask]).flatten()
            for tc, sc in target_scalers.items():
                if tc not in out.columns:
                    continue
                vals = out[[tc]].astype(float)
                mask = vals.notna().values.ravel()
                if mask.any():
                    out.loc[vals.index[mask], tc] = sc.transform(vals[mask])[:, 0]
            return out

        train_sc = _transform(train_df)
        val_sc   = _transform(val_df)
        test_sc  = _transform(test_df) if test_df is not None else None

        scaler_payload = {
            "feature_scalers":    feature_scalers,
            "feature_list":       list(feature_scalers.keys()),
            "target_scalers":     target_scalers,
            "target_list":        list(target_scalers.keys()),
            "age_cols":           age_cols,
            "engineered_cyclical": engineered_cyclical,
        }
        with open(scaler_file,   "wb") as f: pickle.dump(scaler_payload, f)
        with open(p_scaler_file, "wb") as f: pickle.dump(target_scalers, f)
        print(f"\nSaved feature scalers → {scaler_file}")
        print(f"Saved target  scalers → {p_scaler_file}")

        for label, split_df in [("TRAIN", train_sc), ("VAL", val_sc),
                                 ("TEST", test_sc)]:
            if split_df is not None and not split_df.empty:
                self.print_stats(split_df, feature_cols + target_cols,
                                 f"{label} (post-transform)")
                self.print_stats(split_df, engineered_cyclical,
                                 f"{label} cyclical (unscaled)")

        return train_sc, val_sc, test_sc, scaler_payload, target_scalers

    def count_target_rows(self, df: pd.DataFrame, split_name: str) -> dict:
        if df.empty:
            return {}
        counts    = {}
        total     = len(df)
        print(f"\n{split_name.upper()} SET (Total rows: {total:,}):")
        for tc in self.target_cols:
            if tc in df.columns:
                nn  = int(df[tc].notna().sum())
                pct = (total - nn) / total * 100
                counts[tc] = nn
                print(f"  {tc}: {nn:,} non-null  {total-nn:,} null ({pct:.1f}%)")
            else:
                print(f"  {tc}: COLUMN NOT FOUND")
                counts[tc] = 0
        total_vals = total * len(self.target_cols)
        avail      = sum(counts.values())
        print(f"  Overall coverage: {avail/total_vals*100:.1f}% ({avail:,}/{total_vals:,})")
        return counts

    def process_pipeline(self, input_csv: Path, create_val: bool = False):
        """
        Full pipeline:
            load → time features → age features (+ pre-install masking)
            → retain columns → split → scale → save → EDA reports → corr plots
        """
        df = self.load_input_csv(input_csv)
        df = self.add_time_features(df)
        df = self.add_age_features(df, MODULE_CONFIG)

        for group, cols in [
            ("sensor_cols",         self.sensor_cols),
            ("env_cols",            self.env_cols),
            ("engineered_cyclical", self.engineered_cyclical),
            ("age_cols",            self.age_cols),
            ("target_cols",         self.target_cols),
        ]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                print(f"WARNING: missing in {group}: {missing}")

        df = self.retain_only_features_and_target(df)

        # Split
        if create_val:
            train_df, val_df, test_df = self.temporal_fixed_date_split(df)
        else:
            train_df, test_df = self.temporal_fixed_date_split_train_test(df)
            val_df = pd.DataFrame(columns=train_df.columns)

        # Row counts
        train_counts = self.count_target_rows(train_df, "train")
        val_counts   = self.count_target_rows(val_df,   "validation")
        test_counts  = self.count_target_rows(test_df,  "test")

        # Summary table
        print("\n" + "=" * 78)
        print("SUMMARY — NON-NULL ROWS PER TARGET")
        print(f"{'Target':<30} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
        print("-" * 78)
        all_targets = sorted(
            set(train_counts) | set(val_counts) | set(test_counts)
        )
        for tc in all_targets:
            tr = train_counts.get(tc, 0)
            va = val_counts.get(tc, 0)
            te = test_counts.get(tc, 0)
            print(f"{tc:<30} {tr:>10,} {va:>10,} {te:>10,} {tr+va+te:>10,}")
        print("=" * 78)

        # Scale
        train_sc, val_sc, test_sc, scaler_payload, target_scalers = (
            self.scale_features_and_save(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                sensor_cols=self.sensor_cols,
                env_cols=self.env_cols,
                age_cols=self.age_cols,
                engineered_cyclical=self.engineered_cyclical,
                target_cols=self.target_cols,
                scaler_file=self.scaler_file,
                p_scaler_file=self.p_scaler_file,
            )
        )

        # Save parquets
        train_sc.to_parquet(self.out_dir / "train_scaled.parquet")
        val_sc.to_parquet(  self.out_dir / "val_scaled.parquet")
        if test_sc is not None:
            test_sc.to_parquet(self.out_dir / "test_scaled.parquet")
        print(f"\nSaved scaled splits → {self.out_dir}")

        # Column info
        column_info = {
            "target_columns":      self.target_cols,
            "feature_columns":     [c for c in train_sc.columns if c not in self.target_cols],
            "sensor_columns":      self.sensor_cols,
            "env_columns":         self.env_cols,
            "age_columns":         self.age_cols,
            "engineered_cyclical": self.engineered_cyclical,
        }
        with open(self.out_dir / "column_info.pkl", "wb") as f:
            pickle.dump(column_info, f)
        print(f"Saved column info → {self.out_dir / 'column_info.pkl'}")

        return train_sc, val_sc, test_sc, scaler_payload, target_scalers