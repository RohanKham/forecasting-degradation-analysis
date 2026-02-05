#Version: 1.0 
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

class PVValidationPipeline:
    """
    Class to clean and validate PV data from Influxdb.
    """

    def __init__(self, df: pd.DataFrame, validation_modules: List[str], results_dir: str | Path = "result_data_validation",):
        self.df = df.copy()
        self.validation_modules = validation_modules
        
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "validation_plot"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.scaling_ref_module = "Perovskite_1_1"
        self.cross_module_ref = "Sanyo_5_1"

        self._init_metadata()
        self._prepare_time_index()
        self._print_dataset_summary()

    def _init_metadata(self) -> None:
            self.panel_name_fixes = {
                "Atersa_1-1": "Atersa_1_1",
                "Atersa_2-1": "Atersa_2_1",
                "Atersa_3-1": "Atersa_3_1",
                "Atersa_4-1": "Atersa_4_1",
                "Atersa_5-1": "Atersa_5_1",
                "Atersa_6-1": "Atersa_6_1",
            
                "Sanyo_1-1": "Sanyo_1_1",
                "Sanyo_2-1": "Sanyo_2_1",
                "Sanyo_3-1": "Sanyo_3_1",
                "Sanyo_4-1": "Sanyo_4_1",
                "Sanyo_5-1": "Sanyo_5_1",

                "Solon_1-1": "Solon_1_1",
                "Solon_1-2": "Solon_1_2",
                "Solon_2-1": "Solon_2_1",
                "Solon_2-2": "Solon_2_2",
                "Solon_3-1": "Solon_3_1",
                "Solon_3-2": "Solon_3_2",
                "Solon_4-2": "Solon_4_2",

                "Sun_Power_1-1": "Sun_Power_1_1",
                "Sun_Power_2-1": "Sun_Power_2_1",
                "Sun_Power_3-1": "Sun_Power_3_1",
                "Sun_Power_4-1": "Sun_Power_4_1",
                "Sun_Power_5-1": "Sun_Power_5_1",

                "Perovskite_1": "Perovskite_1_1", 
                "Perovskite_2": "Perovskite_1_2",
            }

            self.install_dates = {
                "Atersa_1_1": "2024-07-26 00:00:00",
                "Atersa_2_1": "2024-07-26 00:00:00",
                "Atersa_3_1": "2024-07-26 00:00:00",
                "Atersa_4_1": "2024-07-26 00:00:00",
                "Atersa_5_1": "2024-07-26 00:00:00",
                "Atersa_6_1": "2024-07-26 00:00:00",

                "Sanyo_1_1" : "2024-07-26 00:00:00",
                "Sanyo_2_1" : "2024-07-26 00:00:00",
                "Sanyo_3_1" : "2024-07-26 00:00:00",
                "Sanyo_4_1" : "2024-07-26 00:00:00",
                "Sanyo_5_1" : "2024-07-26 00:00:00",

                "Solon_1_1" : "2024-07-26 00:00:00",
                "Solon_1_2" : "2024-07-26 00:00:00",
                "Solon_2_1" : "2024-07-26 00:00:00",
                "Solon_2_2" : "2024-07-26 00:00:00",
                "Solon_3_1" : "2024-07-26 00:00:00",
                "Solon_3_2" : "2024-07-26 00:00:00",
                "Solon_4_2" : "2024-07-26 00:00:00",

                "Sun_Power_1_1": "2024-07-26 00:00:00",
                "Sun_Power_2_1": "2024-07-26 00:00:00",
                "Sun_Power_3_1": "2024-07-26 00:00:00",
                "Sun_Power_4_1": "2024-07-26 00:00:00",
                "Sun_Power_5_1": "2024-07-26 00:00:00",

                "Perovskite_1_1": "2024-11-28 00:00:00",
                "Perovskite_1_2": "2024-11-28 00:00:00", 
                "Perovskite_1_3": "2025-03-29 12:00:00",
                "Perovskite_2_1": "2025-03-28 15:00:00", 
                "Perovskite_2_2": "2025-03-28 15:00:00", 
                "Perovskite_2_3": "2025-03-28 15:00:00", 
                "Perovskite_3_1": "2025-07-18 14:00:00",
                "Perovskite_3_2": "2025-07-18 10:00:00",
                "Perovskite_3_3": "2025-07-18 10:00:00",
                "Perovskite_4_1": "2025-07-20 14:00:00",
                "Perovskite_4_2": "2025-07-20 10:00:00",
                "Perovskite_4_3": "2025-07-20 10:00:00",
            }

            self.name_mapping_dict = {
                "Atersa_1_1":(1, "atersa", "si"),
                "Atersa_2_1":(2, "atersa", "si"),
                "Atersa_3_1":(3, "atersa", "si"),
                "Atersa_4_1":(4, "atersa", "si"),
                "Atersa_5_1":(5, "atersa", "si"),
                "Atersa_6_1":(6, "atersa", "si"),
                
                "Sanyo_1_1":(1, "sanyo", "si"),
                "Sanyo_2_1":(2, "sanyo", "si"),
                "Sanyo_3_1":(3, "sanyo", "si"),
                "Sanyo_4_1":(4, "sanyo", "si"),
                "Sanyo_5_1":(5, "sanyo", "si"),

                "Solon_1_1":(1, "solon", "si"),
                "Solon_1_2":(2, "solon", "si"),
                "Solon_2_1":(3, "solon", "si"),
                "Solon_2_2":(4, "solon", "si"),
                "Solon_3_1":(5, "solon", "si"),
                "Solon_3_2":(6, "solon", "si"),
                "Solon_4_2":(7, "solon", "si"),

                "Sun_Power_1_1":(1, "sun_power", "si"),
                "Sun_Power_2_1":(2, "sun_power", "si"),
                "Sun_Power_3_1":(3, "sun_power", "si"),
                "Sun_Power_4_1":(4, "sun_power", "si"),
                "Sun_Power_5_1":(5, "sun_power", "si"),

                "Perovskite_1_1": (1, "perovskite", "psc"),
                "Perovskite_1_2": (2, "perovskite", "psc"),
                "Perovskite_1_3": (3, "perovskite", "psc"),
                "Perovskite_2_1": (4, "perovskite", "psc"),
                "Perovskite_2_2": (5, "perovskite", "psc"),
                "Perovskite_2_3": (6, "perovskite", "psc"),
                "Perovskite_3_1": (7, "perovskite", "psc"),
                "Perovskite_3_2": (8, "perovskite", "psc"),
                "Perovskite_3_3": (9, "perovskite", "psc"),
                "Perovskite_4_1": (10, "perovskite", "psc"),
                "Perovskite_4_2": (11, "perovskite", "psc"),
                "Perovskite_4_3": (12, "perovskite", "psc"),
            }

    def _prepare_time_index(self) -> None:
        self.df["_time"] = pd.to_datetime(self.df["_time"], utc=True, errors="coerce")
        self.df = self.df.dropna(subset=["_time"])

    def _print_dataset_summary(self) -> None:
        print("-" * 100)
        print(f"Loaded dataset: {len(self.df):,} rows")
        print("Columns:", list(self.df.columns))
        print("Modules:", sorted(self.df["Name"].unique()))
        print("-" * 100)
        
    def _save_plot(self, fig: plt.Figure, name: str) -> None:
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {path}")

    def clean_module_names(self) -> None:
        print("\nSTEP 1: Cleaning module names")
        unique_before = self.df["Name"].unique()
        print(f"\nUnique module names before cleaning ({len(unique_before)}): {unique_before}")
        print(f"DataFrame size before cleaning: {self.df.shape}")
        self.df["Name"] = self.df["Name"].astype(str).replace(self.panel_name_fixes)
        unique_after = self.df["Name"].unique()
        print(f"\nUnique module names after cleaning ({len(unique_after)}): {unique_after}")
        print(f"DataFrame size after cleaning: {self.df.shape}")

    #remove faulty data before installation date
    def align_resample(self) -> None:
        print("\nSTEP 2: Removing pre-install & duplicate data")
        print(f"Total rows before: {len(self.df):,}")
        filtered = []

        for module, sub in self.df.groupby("Name"):
            if module not in self.install_dates:
                continue

            install_dt = pd.to_datetime(self.install_dates[module], utc=True)
            sub = (
                sub.drop_duplicates(subset="_time")
                .set_index("_time")
                .sort_index()
                .loc[install_dt:]
                .reset_index()
            )
            filtered.append(sub)

        self.df = pd.concat(filtered, ignore_index=True)
        print(f"Total rows after: {len(self.df):,}")

    #create copies of Sanyo_5_1 and Perovskite_1_1 for Irr gap filling
    def create_reference_copies(self)-> None:
        print("\nCreating reference module copies for Irr gap filling")
        
        self.original_scaling_ref = self.df[self.df["Name"] == self.scaling_ref_module][["_time", "Irr"]].copy()
        self.original_cross_ref   = self.df[self.df["Name"] == self.cross_module_ref][["_time", "Irr"]].copy()

        print(f"Scaling ref copy ({self.scaling_ref_module}): {len(self.original_scaling_ref)} rows | Non-NaN Irr values: {self.original_scaling_ref['Irr'].notna().sum()}")
        print(f"Cross ref copy ({self.cross_module_ref}): {len(self.original_cross_ref)} rows | Non-NaN Irr values: {self.original_cross_ref['Irr'].notna().sum()}")

    #drop modules other than validation modules, even the ref modules as their copies are already made with columns _time, Irr columns 
    def drop_non_validation_modules(self):
        print("\nSTEP 3: Drop modules not mentioned in validation modules")
        print(f"DataFrame size before dropping unwanted modules: {self.df.shape}")
        self.df = self.df[self.df["Name"].isin(self.validation_modules)]
        print(f"DataFrame size after dropping unwanted modules: {self.df.shape}")
        print("Modules kept:")
        for module, g in self.df.groupby("Name"):
            n_unique_ts = g["_time"].nunique()
            print(f"{module}: {n_unique_ts:,} timestamps")

    """
        Fill Irradiance gaps in Perovskite_1_1 and apply to all modules:
        1. Load, clean and merge refernce modules.
        2. Load DWD Irr data.
        3. Calculate scaling factors.
        4. Fill gaps within the merged reference dataframe. 
        5. Copy the processed Irr column to all validation modules (exact timestamp match only orelse nan)
        6. Fill per-module missing Irr values using scaled DWD
    """    
    def fill_irradiance_gaps(self, dwd_file, cross_module_ref=None, scaling_ref_module=None):        
        print("\nSTEP 4: Fill Irradiance gaps")
        print(f"df size before: {self.df.shape}")
        
        if cross_module_ref:
            self.cross_module_ref = cross_module_ref
        if scaling_ref_module:
            self.scaling_ref_module = scaling_ref_module

        def load_and_clean_reference_module(module_name, original_data):
            """Load and clean a reference module's data"""
            data = original_data.copy()
            if data is None or data.empty:
                print(f"Reference module '{module_name}' is empty or not available")
                return None
            
            original_count = len(data)
            data["_time"] = pd.to_datetime(data["_time"], utc=True)
            data = data.set_index("_time").sort_index()
            original_na = data["Irr"].isna().sum()
            
            # Identify outliers
            outlier_mask = (data["Irr"] <= 0) | (data["Irr"] > 1300)          
            data_clean = data.copy()
            # Mark outliers as NaN
            data_clean.loc[outlier_mask, "Irr"] = np.nan  
            final_na = data_clean["Irr"].isna().sum()
            new_outliers_marked = final_na - original_na
            
            print(f"Reference module '{module_name}': "
                f"Rows: {original_count}, "
                f"Marked {new_outliers_marked} outliers as NaN, "
                f"Total NaN: {final_na}")
            
            return data_clean[["Irr"]]

        def save_dwd_scaling_factors(doy_period_factors, hourly_fallback):
            """Save DWD irradiance scaling factors to JSON file"""
            scaling_file = self.results_dir / "dwd_irradiance_scaling_factors.json"

            out = {
                "dayofyear_period": {
                    f"{doy:03d}-{period}": {
                        "factor": float(v["factor"]),
                        "rmse": None if np.isnan(v["rmse"]) else float(v["rmse"]),
                        "mape": None if np.isnan(v["mape"]) else float(v["mape"]),
                    }
                    for (doy, period), v in doy_period_factors.items()
                },
                "period_fallback": {
                    period: float(f) for period, f in hourly_fallback.items()
                }
            }

            with open(scaling_file, "w") as f:
                json.dump(out, f, indent=2)

            print(f"Saved DWD scaling factors to: {scaling_file}")
            return scaling_file

        def plot_month_period_scaling_factors():
            """Plot monthly period scaling factors"""
            plot_data = pd.DataFrame([
                {
                    "DayOfYear": doy,
                    "Month": pd.Timestamp(f"2024-01-01") + pd.Timedelta(days=doy-1),
                    "Period": period,
                    "Factor": v["factor"]
                }
                for (doy, period), v in doy_period_factors.items()
            ])
            plot_data["Month"] = plot_data["Month"].dt.month 

            periods = ["morning", "midday", "evening"]
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

            for i, period in enumerate(periods):
                ax = axes[i]
                subset = plot_data[plot_data["Period"] == period]
                summary = subset.groupby("Month")["Factor"].agg(["min", "max", "mean"]).reset_index()

                for _, row in summary.iterrows():
                    ax.vlines(
                        x=row["Month"],
                        ymin=row["min"],
                        ymax=row["max"],
                        color="black",
                        linewidth=3,
                        alpha=0.6
                    )
                ax.plot(
                    summary["Month"],
                    summary["mean"],
                    color="black",
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=8,
                    label=f"{period.capitalize()} Mean Factor"
                )

                ax.set_ylabel("Scaling Factor")
                ax.set_title(f"{period.capitalize()} Scaling Factor per Month")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                ax.set_xticks(range(1, 13))
                ax.set_xlim(1, 12)

            axes[-1].set_xlabel("Month")

            plt.tight_layout()
            self._save_plot(fig, "monthly_period_scaling_factors")
            plt.close(fig)

        print("\nLoading and merging reference modules.")        
        sanyo_data = None
        perovskite_data = None
        
        # Check if Sanyo_5_1 exists
        if hasattr(self, 'original_cross_ref') and self.original_cross_ref is not None:
            sanyo_data = load_and_clean_reference_module("Sanyo_5_1", self.original_cross_ref)
        
        # Check if Perovskite_1_1 exists
        if hasattr(self, 'original_scaling_ref') and self.original_scaling_ref is not None:
            perovskite_data = load_and_clean_reference_module("Perovskite_1_1", self.original_scaling_ref)
        
        # Merge reference modules
        if sanyo_data is None and perovskite_data is None:
            raise ValueError("Neither Sanyo_5_1 nor Perovskite_1_1 is available. Cannot proceed with gap filling.")
        
        if sanyo_data is not None and perovskite_data is not None:
            print("\nMerging Sanyo_5_1 and Perovskite_1_1 reference data...")
            # Concatenate both dataframes
            merged_refs = pd.concat([
                sanyo_data.rename(columns={"Irr": "Irr_sanyo"}),
                perovskite_data.rename(columns={"Irr": "Irr_perovskite"})
            ], axis=1)
            
            # For overlapping timestamps, take the mean
            merged_refs["Irr_ref"] = merged_refs[["Irr_sanyo", "Irr_perovskite"]].mean(axis=1, skipna=True)
            
            overlapping = merged_refs[["Irr_sanyo", "Irr_perovskite"]].notna().all(axis=1).sum()
            sanyo_only = merged_refs["Irr_sanyo"].notna().sum() - overlapping
            perovskite_only = merged_refs["Irr_perovskite"].notna().sum() - overlapping
            total_ref_points = merged_refs["Irr_ref"].notna().sum()
            
            print(f"Merged reference data: {overlapping} overlapping (averaged), "
                f"{sanyo_only} Sanyo-only, {perovskite_only} Perovskite-only, "
                f"Total: {total_ref_points} valid reference points")
            
            scaling_ref_data_clean = merged_refs[["Irr_ref"]].copy()
            
            # Keep individual modules for plotting
            sanyo_ref_clean = sanyo_data.rename(columns={"Irr": f"Irr_Sanyo_5_1"})
            perovskite_ref_clean = perovskite_data.rename(columns={"Irr": f"Irr_Perovskite_1_1"})
            
        elif sanyo_data is not None:
            print("\nUsing only Sanyo_5_1 as reference (Perovskite_1_1 not available)")
            scaling_ref_data_clean = sanyo_data.rename(columns={"Irr": "Irr_ref"})
            sanyo_ref_clean = sanyo_data.rename(columns={"Irr": f"Irr_Sanyo_5_1"})
            perovskite_ref_clean = None
        else:
            print("\nUsing only Perovskite_1_1 as reference (Sanyo_5_1 not available)")
            scaling_ref_data_clean = perovskite_data.rename(columns={"Irr": "Irr_ref"})
            sanyo_ref_clean = None
            perovskite_ref_clean = perovskite_data.rename(columns={"Irr": f"Irr_Perovskite_1_1"})
        
        print(f"Final scaling reference data shape: {scaling_ref_data_clean.shape}, "
            f"Non-NaN count: {scaling_ref_data_clean['Irr_ref'].notna().sum()}")

        print("\nLoading DWD data file")
        dwd = pd.read_csv(dwd_file)
        dwd.columns = dwd.columns.str.strip()
        if "timestamp" not in dwd.columns:
            raise ValueError("DWD file must contain 'timestamp' column")
        dwd["timestamp"] = pd.to_datetime(dwd["timestamp"], errors="coerce", utc=True)
        dwd = dwd.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        if "global_radiation" not in dwd.columns:
            raise ValueError("DWD file must contain 'global_radiation' column")
        dwd = dwd.rename(columns={"global_radiation": "Irr_dwd"})

        #check and if not resample dwd irr to 10min frequency
        mean_step = dwd.index.to_series().diff().median()
        if not (pd.Timedelta("9min") <= mean_step <= pd.Timedelta("11min")):
            print("Resampling DWD to 10 min")
            dwd = dwd.resample("10min").mean()

        # Align DWD data with scaling reference time range
        scaling_start = scaling_ref_data_clean.index.min()
        scaling_end = scaling_ref_data_clean.index.max()
        dwd = dwd.loc[scaling_start:scaling_end]
        print(f"DWD data shape after alignment: {dwd.shape}, Non-NaN count: {dwd['Irr_dwd'].notna().sum()}")
        
        # Merge scaling reference with DWD
        merged_clean = pd.concat([scaling_ref_data_clean, dwd], axis=1)
        print(f"Merged (ref + DWD) shape: {merged_clean.shape}")

        print("\nCalculating DWD scaling factors")
        doy_period_factors = {}
        hourly_fallback = {}

        periods = {
            "morning": range(4, 11),  # 4:00 to 10:59
            "midday": range(11, 17),  # 11:00 to 16:59
            "evening": range(17, 22)  # 17:00 to 21:59
        }

        for doy in range(1, 366):
            for period_name, hours in periods.items():
                mask = (
                    (merged_clean.index.dayofyear == doy) &
                    (merged_clean.index.hour.isin(hours)) &
                    (merged_clean["Irr_ref"] > 10) &
                    (merged_clean["Irr_dwd"] > 10) &
                    (merged_clean["Irr_ref"] < 1300)
                )
                if mask.sum() >= 20:
                    subset = merged_clean.loc[mask]
                    ratio = subset["Irr_ref"] / subset["Irr_dwd"]
                    ratio = ratio[(ratio > 0.0) & (ratio < 3.0)]
                    if len(ratio) >= 5:
                        factor = ratio.median()
                        scaled = subset["Irr_dwd"] * factor
                        rmse = np.sqrt(((subset["Irr_ref"] - scaled) ** 2).mean())
                        mape = (np.abs(subset["Irr_ref"] - scaled) / subset["Irr_ref"]).mean() * 100
                    else:
                        factor, rmse, mape = 0.0, np.nan, np.nan
                else:
                    factor, rmse, mape = 0.0, np.nan, np.nan

                doy_period_factors[(doy, period_name)] = {
                    "factor": factor,
                    "rmse": rmse,
                    "mape": mape
                }
        
        # Calculate hourly fallback factors
        for period_name, hours in periods.items():
            mask = (
                (merged_clean.index.hour.isin(hours)) &
                (merged_clean["Irr_ref"] > 10) &
                (merged_clean["Irr_dwd"] > 10)
            )
            if mask.sum() >= 20:
                ratio = merged_clean.loc[mask, "Irr_ref"] / merged_clean.loc[mask, "Irr_dwd"]
                ratio = ratio[(ratio > 0.0) & (ratio < 3.0)]
                factor = ratio.median() if len(ratio) >= 30 else 0.0
            else:
                factor = 0.0
            hourly_fallback[period_name] = factor

        print(f"Calculated {len(doy_period_factors)} day-of-year period factors")        
        # Save and plot scaling factors
        self.dwd_scaling_file = save_dwd_scaling_factors(doy_period_factors, hourly_fallback)
        plot_month_period_scaling_factors()

        # Define scaling application function
        def apply_doy_period_scaling(ts, dwd_val):
            """Apply day-of-year and period-based scaling to DWD value"""
            if pd.isna(dwd_val):
                return np.nan

            doy = ts.dayofyear
            hour = ts.hour

            period_name = None
            for pname, hrs in periods.items():
                if hour in hrs:
                    period_name = pname
                    break

            if period_name is None:
                return np.nan

            key = (doy, period_name)
            if key in doy_period_factors:
                factor = doy_period_factors[key]["factor"]
            else:
                factor = hourly_fallback.get(period_name, 0.0)

            return dwd_val * factor

        print("\nFilling gaps within merged reference data")
        scaling_ref_final = scaling_ref_data_clean.copy()
        initial_gaps = scaling_ref_final["Irr_ref"].isna().sum()
        print(f"Initial gaps in merged reference: {initial_gaps}")
        
        if initial_gaps > 0:
            missing_idx = scaling_ref_final[scaling_ref_final["Irr_ref"].isna()].index
            dwd_missing = dwd.reindex(missing_idx)
            
            filled_count = 0
            for ts in dwd_missing.dropna().index:
                scaled_val = apply_doy_period_scaling(ts, dwd_missing.at[ts, "Irr_dwd"])
                if not pd.isna(scaled_val):
                    scaling_ref_final.loc[ts, "Irr_ref"] = scaled_val
                    filled_count += 1
            
            remaining_gaps = scaling_ref_final["Irr_ref"].isna().sum()
            print(f"Filled {filled_count} gaps using scaled DWD, {remaining_gaps} gaps remaining")
        else:
            print("No gaps to fill in merged reference")
            filled_count = 0
            remaining_gaps = 0

        # Remove remaining NaN values
        scaling_ref_final = scaling_ref_final[scaling_ref_final["Irr_ref"].notna()]
        
        total_possible = len(scaling_ref_data_clean)
        final_coverage = len(scaling_ref_final) / total_possible * 100 if total_possible > 0 else 0
        
        print(f"Merged reference final stats: Total filled {filled_count}, "
            f"Final coverage: {final_coverage:.1f}%, "
            f"Total valid points: {len(scaling_ref_final)}")

        print("\nCopying processed Irr to all validation modules")

        irr_reference_df = scaling_ref_final[["Irr_ref"]].reset_index()
        irr_reference_df.columns = ["_time", "Irr_processed"]

        all_processed = []
        modules_processed = []

        for module, sub in self.df.groupby("Name"):
            if module not in self.validation_modules:
                continue
                
            sub = sub.copy()
            sub["_time"] = pd.to_datetime(sub["_time"], utc=True)
            
            # Merge on exact timestamp only 
            before_merge = len(sub)
            sub_merged = pd.merge(
                sub,
                irr_reference_df,
                on="_time",
                how="left"  
            )
            
            # Mismatches remain NaN
            sub_merged["Irr"] = sub_merged["Irr_processed"]
            sub_merged = sub_merged.drop(columns=["Irr_processed"])
            
            exact_matches = sub_merged["Irr"].notna().sum()
            mismatches = sub_merged["Irr"].isna().sum()
            match_percentage = (exact_matches / len(sub_merged)) * 100
            
            modules_processed.append(module)
            print(f"{module}: {exact_matches}/{len(sub_merged)} exact timestamp matches ({match_percentage:.1f}%) - {mismatches} gaps for L4")
            all_processed.append(sub_merged)

        # Combine all modules
        if all_processed:
            self.df = pd.concat(all_processed, ignore_index=True)
            print(f"Combined df shape after exact timestamp matching: {self.df.shape}")
        else:
            print("WARNING: No validation modules were processed!")

        print("\nFilling remaining per-module gaps using scaled DWD")

        # Pre-compute all scaled DWD values at once
        print("Pre-computing scaled DWD values...")
        dwd_scaled = dwd.copy()
        dwd_scaled["Irr_scaled"] = dwd_scaled.apply(
            lambda row: apply_doy_period_scaling(row.name, row["Irr_dwd"]),
            axis=1
        )
        print(f"Pre-computed {dwd_scaled['Irr_scaled'].notna().sum()} scaled DWD values")

        l4_filled_total = 0
        modules_l4_stats = []

        for module in modules_processed:
            module_mask = self.df["Name"] == module
            module_na_before = self.df.loc[module_mask, "Irr"].isna().sum()
            
            if module_na_before > 0:
                # Get indices where Irr is missing for this module
                missing_mask = module_mask & self.df["Irr"].isna()
                
                # Create a temporary dataframe with missing timestamps
                missing_df = self.df.loc[missing_mask, ["_time"]].copy()
                missing_df["_time"] = pd.to_datetime(missing_df["_time"], utc=True)
                
                # Merge with pre-computed scaled DWD values
                filled = pd.merge(
                    missing_df,
                    dwd_scaled[["Irr_scaled"]].reset_index(),
                    left_on="_time",
                    right_on="timestamp",
                    how="left"
                )
                
                # Update only non-NaN scaled values
                valid_fill_mask = filled["Irr_scaled"].notna()
                l4_filled_module = valid_fill_mask.sum()
                
                if l4_filled_module > 0:
                    # Get the indices in the original dataframe
                    fill_indices = self.df.index[missing_mask][valid_fill_mask]
                    self.df.loc[fill_indices, "Irr"] = filled.loc[valid_fill_mask, "Irr_scaled"].values
                
                module_na_after = self.df.loc[module_mask, "Irr"].isna().sum()
                l4_filled_total += l4_filled_module
                
                modules_l4_stats.append((module, module_na_before, l4_filled_module, module_na_after))

        for module, na_before, l4_filled, na_after in modules_l4_stats:
            if na_before > 0:
                coverage_improvement = (l4_filled / na_before * 100) if na_before > 0 else 0
                print(f"{module}: {na_before} gaps → {l4_filled} filled → {na_after} remaining ({coverage_improvement:.1f}% filled)")
            else:
                print(f"{module}: No gaps to fill")

        print(f"Total L4 gaps filled: {l4_filled_total}")
        
        # Remove any remaining unfillable gaps
        before_removal = len(self.df)
        self.df = self.df[self.df["Irr"].notna()]
        removed_rows = before_removal - len(self.df)
        print(f"Removed {removed_rows} rows with unfillable Irr gaps")
        
        # Final statistics
        print("\nPer-module timestamp and Irr count after gap filling:")
        for module, g in self.df.groupby("Name"):
            n_rows = len(g)
            n_unique_ts = g["_time"].nunique()
            n_irr = g["Irr"].notna().sum()
            print(
                f"{module}: "
                f"rows={n_rows:,}, "
                f"unique timestamps={n_unique_ts:,}, "
                f"Irr count={n_irr:,}"
            )

        print(f"\ndf size after: {self.df.shape}")
        
        # Plot comparisons
        print("\nGenerating comparison plots...")
        # For plotting, prepare the reference data
        if sanyo_ref_clean is not None and perovskite_ref_clean is not None:
            # Use the first available for compatibility with plot function
            plot_ref = sanyo_ref_clean
        elif sanyo_ref_clean is not None:
            plot_ref = sanyo_ref_clean
        else:
            plot_ref = perovskite_ref_clean
        
        self.plot_irradiance_gap_filling_overview(
            scaling_ref_final=scaling_ref_final,
            dwd=dwd,
            sanyo_ref_clean=plot_ref
        )
        
        return self.df

    #NEW: Plotting function
    def plot_irradiance_gap_filling_overview(
        self,
        scaling_ref_final: pd.DataFrame,
        dwd: pd.DataFrame,
        sanyo_ref_clean: pd.Series | None,
    ):
        """
        Overview plots for irradiance gap filling and DWD scaling analysis.
        Uses raw DOY-hour scaling factors (Irr_ref / Irr_dwd).
        """

        sns.set_style("whitegrid")
        merged = pd.concat([scaling_ref_final, dwd], axis=1)
        merged = merged.dropna(subset=["Irr_ref", "Irr_dwd"])
        merged["hour"] = merged.index.hour
        merged = merged[(merged["hour"] >= 4) & (merged["hour"] < 22)]
        merged = merged[merged["Irr_dwd"] > 0]
        merged["year"] = merged.index.year
        merged["doy"] = merged.index.dayofyear
        merged["month"] = merged.index.month

        # Raw scaling factor (NO period aggregation)
        merged["raw_scaling_factor"] = merged["Irr_ref"] / merged["Irr_dwd"]
        merged = merged[
            (merged["raw_scaling_factor"] > 0) &
            (merged["raw_scaling_factor"] < 3)
        ]
        SEASON_MAP = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        merged["season"] = merged["month"].map(SEASON_MAP)
        fig = plt.figure(figsize=(22, 18))

        # 1. DWD vs Scaling reference
        ax1 = plt.subplot(3, 4, 1)
        ax1.scatter(merged["Irr_dwd"], merged["Irr_ref"], s=4, alpha=0.4)
        max_val = merged["Irr_dwd"].max()
        ax1.plot([0, max_val], [0, max_val], "r--")
        ax1.set_xlabel("DWD Irradiance [W/m²]")
        ax1.set_ylabel(f"{self.scaling_ref_module} Irradiance [W/m²]")
        ax1.set_title("DWD vs Scaling Reference")
        ax1.grid(True, alpha=0.3)

        # 2. Scaling reference vs Cross-module reference
        ax2 = plt.subplot(3, 4, 2)
        if sanyo_ref_clean is not None:
            ref_cross = pd.concat(
                [scaling_ref_final["Irr_ref"], sanyo_ref_clean],
                axis=1
            ).dropna()
            ax2.scatter(ref_cross.iloc[:, 0], ref_cross.iloc[:, 1], s=4, alpha=0.4)
            max_val2 = ref_cross.max().max()
            ax2.plot([0, max_val2], [0, max_val2], "r--")
            ax2.set_xlabel("Scaling Ref Irr [W/m²]")
            ax2.set_ylabel(f"{self.cross_module_ref} Irr [W/m²]")
            ax2.set_title("Scaling Ref vs Cross-module")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No cross-module reference",
                    ha="center", va="center")
            ax2.axis("off")

        # 3. Raw scaling factor histogram
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(merged["raw_scaling_factor"], bins=60, edgecolor="black", alpha=0.7)
        ax3.axvline(merged["raw_scaling_factor"].median(),
                    color="red", linestyle="--",
                    label=f"Median = {merged['raw_scaling_factor'].median():.3f}")
        ax3.set_xlabel("Raw scaling factor (Irr_ref / Irr_dwd)")
        ax3.set_ylabel("Count")
        ax3.set_title("Scaling Factor Distribution")
        ax3.legend()

        # 4. Scaling factor vs Irr (colored by hour)
        ax4 = plt.subplot(3, 4, 4)
        sc = ax4.scatter(
            merged["Irr_dwd"],
            merged["raw_scaling_factor"],
            c=merged["hour"],
            s=4,
            alpha=0.4,
            cmap="viridis"
        )
        ax4.set_xlabel("DWD Irradiance [W/m²]")
        ax4.set_ylabel("Scaling factor")
        ax4.set_title("Scaling Factor vs Irradiance")
        ax4.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax4, label="Hour of day")

        # 5. Hourly scaling factor boxplot (raw)
        ax5 = plt.subplot(3, 4, 5)
        hour_data = [
            merged.loc[merged["hour"] == h, "raw_scaling_factor"].values
            for h in range(24)
            if not merged.loc[merged["hour"] == h].empty
        ]
        hour_labels = [
            h for h in range(24)
            if not merged.loc[merged["hour"] == h].empty
        ]
        ax5.boxplot(hour_data, labels=hour_labels, showfliers=True)
        ax5.set_xlabel("Hour")
        ax5.set_ylabel("Scaling factor")
        ax5.set_title("Hourly Scaling Factor (raw)")
        ax5.grid(True, alpha=0.3)

        # 6. Monthly scaling factor boxplot (raw)
        ax6 = plt.subplot(3, 4, 6)
        month_data = [
            merged.loc[merged["month"] == m, "raw_scaling_factor"].values
            for m in range(1, 13)
            if not merged.loc[merged["month"] == m].empty
        ]
        month_labels = [
            m for m in range(1, 13)
            if not merged.loc[merged["month"] == m].empty
        ]
        ax6.boxplot(month_data, labels=month_labels, showfliers=True)
        ax6.set_xlabel("Month")
        ax6.set_ylabel("Scaling factor")
        ax6.set_title("Monthly Scaling Factor (raw)")
        ax6.grid(True, alpha=0.3)

        # 7. Seasonal scaling factor trend
        ax7 = plt.subplot(3, 4, 7)
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        season_data = [
            merged.loc[merged["season"] == s, "raw_scaling_factor"].values
            for s in season_order
        ]
        ax7.boxplot(season_data, labels=season_order, showfliers=True)
        ax7.set_ylabel("Scaling factor")
        ax7.set_title("Seasonal Scaling Factor Trend")
        ax7.grid(True, alpha=0.3)

        # 8. Scaling factor time evolution (sampled)
        ax8 = plt.subplot(3, 4, 8)
        sample = merged.iloc[::10]
        sc2 = ax8.scatter(
            sample.index,
            sample["raw_scaling_factor"],
            c=sample["hour"],
            s=4,
            alpha=0.5,
            cmap="viridis"
        )
        ax8.set_xlabel("Time")
        ax8.set_ylabel("Scaling factor")
        ax8.set_title("Scaling Factor Over Time (sampled)")
        ax8.tick_params(axis="x", rotation=45)
        plt.colorbar(sc2, ax=ax8, label="Hour of day")

        # 9. Residuals using median scaling
        ax9 = plt.subplot(3, 4, 9)
        median_factor = merged["raw_scaling_factor"].median()
        predicted = merged["Irr_dwd"] * median_factor
        residuals = merged["Irr_ref"] - predicted
        ax9.scatter(predicted, residuals, s=4, alpha=0.4)
        ax9.axhline(0, color="red", linestyle="--")
        ax9.set_xlabel("Predicted Irr [W/m²]")
        ax9.set_ylabel("Residual [W/m²]")
        ax9.set_title("Residuals (Median Scaling)")
        ax9.grid(True, alpha=0.3)

        # 10. Summary statistics
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis("off")

        stats_text = [
            f"Scaling reference: {self.scaling_ref_module}",
            f"Cross reference: {self.cross_module_ref if sanyo_ref_clean is not None else 'None'}",
            f"Years covered: {merged['year'].min()}–{merged['year'].max()}",
            f"Total points: {len(merged):,}",
            "",
            f"Median scaling factor: {merged['raw_scaling_factor'].median():.3f}",
            f"Mean scaling factor: {merged['raw_scaling_factor'].mean():.3f}",
            f"Std scaling factor: {merged['raw_scaling_factor'].std():.3f}",
            "",
            f"Min: {merged['raw_scaling_factor'].min():.3f}",
            f"Max: {merged['raw_scaling_factor'].max():.3f}",
        ]

        ax10.text(
            0.02, 0.98, "\n".join(stats_text),
            transform=ax10.transAxes,
            va="top",
            family="monospace",
            fontsize=9
        )

        plt.suptitle(
            f"Irradiance Gap Filling & Scaling Analysis – {self.scaling_ref_module}",
            fontsize=16,
            y=1.02
        )
        plt.tight_layout()
        self._save_plot(fig, "irradiance_gap_filling_overview")
        plt.close(fig)

    def night_detector(
        self,
        mode: int = 3,
        night_start: int = 22,
        night_end: int = 4,
        freq: str = "10min",
    ):
        """
        Night data handling with three modes:
        mode 1: remove_night_rows - Drop all rows between night_start and night_end
        mode 2: add_night_rows_with_nan - Ensure full night timestamps exist per module
                                        - Set P, I, U, Irr, Temp = NaN
        mode 3: add_night_rows_with_zero - Ensure full night timestamps exist per module
                                        - Set P, I, U, Irr, Temp = 0.0
        """

        print("\nSTEP 5: Night data handling")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")
        cols = ["P", "I", "U", "Irr", "Temp"]
        before_rows = len(self.df)

        self.df["_time"] = pd.to_datetime(self.df["_time"], utc=True)
        self.df["_hour"] = self.df["_time"].dt.hour

        night_mask = (
            (self.df["_hour"] >= night_start) |
            (self.df["_hour"] <= night_end)
        )

        if mode == 1:
            removed = night_mask.sum()
            self.df = self.df.loc[~night_mask].copy()

            print(f"Mode 1: Remove night rows ({night_start}:00-{night_end}:00)")
            print(f"Rows removed: {removed:,}")
            print(f"Rows remaining: {len(self.df):,}")

        elif mode in (2, 3):
            value = np.nan if mode == 2 else 0.0
            mode_name = "NaN" if mode == 2 else "Zero"

            print(f"Mode {mode}: Add night rows with {mode_name} ({night_start}:00-{night_end}:00)")

            all_modules = self.df["Name"].unique()

            full_time_index = pd.date_range(
                start=self.df["_time"].min().floor("D"),
                end=self.df["_time"].max().ceil("D"),
                freq=freq,
                tz="UTC"
            )

            night_times = [
                t for t in full_time_index
                if (t.hour >= night_start or t.hour <= night_end)
            ]

            new_rows = []

            for module in all_modules:
                module_times = set(self.df.loc[self.df["Name"] == module, "_time"])

                for ts in night_times:
                    if ts not in module_times:
                        new_rows.append({
                            "_time": ts,
                            "Name": module,
                            "P": value,
                            "I": value,
                            "U": value,
                            "Irr": value,
                            "Temp": value,
                        })

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                self.df = pd.concat([self.df, new_df], ignore_index=True)
                self.df = self.df.sort_values(["_time", "Name"]).reset_index(drop=True)

            added = len(new_rows)

            print(f"Rows added: {added:,}")
            print(f"Total rows after operation: {len(self.df):,}")

        else:
            raise ValueError("Invalid mode. Use mode = 1, 2, or 3.")

        self.df = self.df.drop(columns=["_hour"], errors="ignore")

        delta = len(self.df) - before_rows
        sign = "+" if delta >= 0 else ""
        print(f"Net row change: {sign}{delta:,}")

        return self.df

    #NEW: Add columns based on name_mapping_dict
    def map_module_metadata(self):
        """
        Adds columns:
        - module_id (eg. 1,2,3, ...) 
        - module_type (eg. atersa, sanyo, solon, ...)
        - category (eg. si/psc)
        """
        print("\nSTEP 6: Mapping module metadata from name_mapping_dict")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")

        # initialize columns
        self.df["module_id"] = np.nan
        self.df["module_type"] = np.nan
        self.df["category"] = np.nan

        missing_modules = []

        for module, idx in self.df.groupby("Name").groups.items():
            if module not in self.name_mapping_dict:
                missing_modules.append(module)
                continue

            module_id, module_type, category = self.name_mapping_dict[module]

            self.df.loc[idx, "module_id"] = int(module_id)
            self.df.loc[idx, "module_type"] = str(module_type)
            self.df.loc[idx, "category"] = str(category)

            print(
                f"{module:30s} -> "
                f"id={module_id}, type={module_type}, category={category} "
                f"({len(idx):,} rows)"
            )

        # Report missing mappings
        if missing_modules:
            print("\nWARNING: Modules missing in name_mapping_dict")
            print("-" * 40)
            for m in missing_modules:
                print(m)

        self.df["module_id"] = self.df["module_id"].astype("Int64")

        print("\nSUMMARY")
        print("-" * 40)
        print(self.df[["module_type", "category"]].value_counts(dropna=False))

    #NEW
    def plot_module_metadata_summary(self, out_png):
        """
        Plot summary of module metadata distribution
        """
        print("\nPlotting module metadata summary")

        df = self.df.copy()
        df["date"] = df["_time"].dt.date

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("Module Metadata Distribution Summary", fontsize=16)

        summaries = [
            ("category", "Category"),
            ("module_type", "Module Type"),
            ("module_id", "Module ID"),
        ]

        for row, (col, title) in enumerate(summaries):
            # rows
            row_counts = df[col].value_counts().sort_index()
            axes[row, 0].bar(row_counts.index.astype(str), row_counts.values)
            axes[row, 0].set_title(f"{title} – Number of Rows")
            axes[row, 0].set_ylabel("Rows")
            axes[row, 0].tick_params(axis="x", rotation=45)

            # days
            day_counts = (
                df.groupby(col)["date"]
                .nunique()
                .sort_index()
            )
            axes[row, 1].bar(day_counts.index.astype(str), day_counts.values)
            axes[row, 1].set_title(f"{title} – Number of Days")
            axes[row, 1].set_ylabel("Days")
            axes[row, 1].tick_params(axis="x", rotation=45)

        for ax in axes.flat:
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_png, dpi=150)
        plt.close(fig)

        print(f"Saved metadata summary plot: {out_png}")

    #NEW : range_validation based on module_type column (eg. atersa, sanyo, solon, ...)
    def range_validation(self):
        """
        Validation is performed per module_type
        """

        print("\nSTEP 7: Physical range validation (daytime only)")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")

        night_mask = (
            ((self.df["_time"].dt.hour >= 22) | (self.df["_time"].dt.hour <= 4)) & (self.df["Irr"] < 1)
        )

        bounds_by_module_type = {
            "atersa": {
                "P": (0, 170),
                "I": (0, 6),
                "U": (10, 50),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
            "sanyo": {
                "P": (0, 200),
                "I": (0, 6),
                "U": (10, 50),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
            "solon": {
                "P": (0, 250),
                "I": (0, 9),
                "U": (10, 40),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
            "sun_power": {
                "P": (0, 250),
                "I": (0, 6),
                "U": (10, 50),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
            "perovskite": {
                "P": (0, 110),
                "I": (0, 1.4),
                "U": (40, 100),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
        }

        params = ["P", "I", "U", "Irr", "Temp"]

        # initialize flags
        for p in params:
            self.df[f"invalid_physical_{p}"] = 0

        # daytime data only
        df_day = self.df.loc[~night_mask]

        print(f"Total rows   : {len(self.df):,}")
        print(f"Daytime rows : {len(df_day):,}")
        print(f"Night rows   : {night_mask.sum():,}")

        # validation per module_type
        for module_type, sub in df_day.groupby("module_type"):
            if module_type not in bounds_by_module_type:
                print(f"\nWARNING: No bounds defined for module_type = {module_type}")
                continue

            bounds = bounds_by_module_type[module_type]

            print(f"\n{module_type} - daytime rows: {len(sub):,}")
            print("Param | Invalid | Min(day) | Max(day)")
            print("-" * 40)

            for p in params:
                lo, hi = bounds[p]
                bad = (sub[p] < lo) | (sub[p] > hi)

                self.df.loc[sub.index[bad], f"invalid_physical_{p}"] = 1

                valid_vals = sub.loc[~bad, p]
                p_min = valid_vals.min() if not valid_vals.empty else float("nan")
                p_max = valid_vals.max() if not valid_vals.empty else float("nan")

                print(f"{p:5} | {bad.sum():7} | {p_min:8.2f} | {p_max:8.2f}")

        print("\nSUMMARY (daytime only)")
        print("-" * 40)
        for p in params:
            count = self.df.loc[~night_mask, f"invalid_physical_{p}"].sum()
            pct = count / len(df_day) * 100 if len(df_day) > 0 else 0
            print(f"{p:5}: {count:7} invalid ({pct:5.2f}%)")

    def detect_statistical_outliers(self, param_cols=["P", "I", "U", "Irr", "Temp"], z_thresh=3.0):
        print("\nSTEP 8: Z-Score based Outlier detection")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")
        night_mask = (
            ((self.df["_time"].dt.hour >= 22) | (self.df["_time"].dt.hour <= 4)) & (self.df["Irr"] < 1)
        )

        total_night_rows = night_mask.sum()
        total_day_rows = len(self.df) - total_night_rows
        
        for col in param_cols:
            self.df[f"invalid_outlier_{col}"] = 0
        
        print(f"Detecting outliers with Z-score threshold ±{z_thresh}")
        
        for module, sub in self.df.groupby("Name"):
            if len(sub) < 10:
                continue
            
            night_sub_mask = night_mask.loc[sub.index]
            sub_day = sub.loc[~night_sub_mask]
            
            print(f"\n{module} — total rows: {len(sub):,} | day: {len(sub_day):,} | night: {len(sub) - len(sub_day):,}")
            print("-" * 80)
            print("Param    | Mean    | Std Dev | Outliers | % Outliers | Z-score Range")
            print("-" * 80)
            
            for col in param_cols:
                if col not in sub_day.columns or sub_day[col].empty:
                    continue
                
                mean = sub_day[col].mean()
                std = sub_day[col].std()
                if std < 1e-9:  
                    continue
                
                z_scores = (sub_day[col] - mean) / std
                outliers_mask = z_scores.abs() > z_thresh
                outlier_indices = sub_day.index[outliers_mask]
                
                self.df.loc[outlier_indices, f"invalid_outlier_{col}"] = 1
                
                outlier_count = outliers_mask.sum()
                outlier_pct = (outlier_count / len(sub_day)) * 100 if len(sub_day) > 0 else 0
                
                print(f"{col:7} | {mean:7.2f} | {std:7.2f} | {outlier_count:8} | {outlier_pct:10.1f}% | [{z_scores.min():6.2f}, {z_scores.max():6.2f}]")
        
        # Summary of all outliers
        print("\n" + "-" * 80)
        print("Summary (Daytime only)")
        print("-" * 80)
        print(f"Total rows        : {len(self.df):,}")
        print(f"Daytime rows      : {total_day_rows:,}")
        print(f"Nighttime rows    : {total_night_rows:,}")
        total_outliers = sum(self.df[f"invalid_outlier_{col}"].sum() for col in param_cols)
        print(f"Total outliers    : {total_outliers:,}")
        print("-" * 80)
        print("Param    | Total Outliers | % of Daytime Data")
        print("-" * 80)
        for col in param_cols:
            col_outliers = self.df[f"invalid_outlier_{col}"].sum()
            pct = (col_outliers / total_day_rows) * 100 if total_day_rows > 0 else 0
            print(f"{col:7} | {col_outliers:14,} | {pct:17.1f}%")

    #NEW
    def detect_correlation_anomalies(self, mark_as_nan: bool = True) -> list[dict]:
        """
        Detect correlation anomalies between Irr and P by comparing relative changes.
        """
        print("\nSTEP 9: Correlation Anomaliy Detection")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")
        min_proportion = 0.05       #minimum allowed proportion of P_change / Irr_change
        min_irr_threshold = 10.0    #minimum Irr threshold to consider in comparison

        anomalies = []

        if not {'Irr', 'P', '_time'}.issubset(self.df.columns):
            print(f"WARNING: Required columns 'Irr', 'P', or '_time' not found.")
            return anomalies

        print(f"Starting correlation anomaly detection...")

        df = self.df.sort_values('_time').reset_index()

        valid_mask = (df['Irr'] > 0) & (df['P'] > 0)
        valid_indices = df.index[valid_mask].tolist()

        if len(valid_indices) < 2:
            print(f"[WARNING: Not enough valid data points for anomaly detection.")
            return anomalies

        for i in range(1, len(valid_indices)):
            prev_i = valid_indices[i-1]
            curr_i = valid_indices[i]

            prev_irr = df.at[prev_i, 'Irr']
            curr_irr = df.at[curr_i, 'Irr']
            prev_p = df.at[prev_i, 'P']
            curr_p = df.at[curr_i, 'P']

            # Skip if both Irr values below threshold
            if prev_irr < min_irr_threshold and curr_irr < min_irr_threshold:
                continue

            # Avoid division by zero
            if prev_irr == 0 or prev_p == 0:
                continue

            irr_change = (curr_irr - prev_irr) / prev_irr
            p_change = (curr_p - prev_p) / prev_p

            if abs(irr_change) < 0.01:
                continue

            proportion = abs(p_change) / abs(irr_change) if irr_change != 0 else 1.0

            if proportion < min_proportion:
                anomaly = {
                    "timestamp": df.at[curr_i, '_time'],
                    "type": "irr_p_correlation_mismatch",
                    "irr_change_pct": irr_change * 100,
                    "p_change_pct": p_change * 100,
                    "proportion_pct": proportion * 100,
                    "min_proportion_pct": min_proportion * 100,
                    "irr_value": curr_irr,
                    "p_value": curr_p,
                    "prev_irr": prev_irr,
                    "prev_p": prev_p,
                    "row_index": df.at[curr_i, 'index'],
                    "description": f"P changed only {proportion*100:.1f}% relative to Irr change "
                                f"(min required: {min_proportion*100:.1f}%)"
                }
                anomalies.append(anomaly)

                idx = anomaly["row_index"]
                if mark_as_nan:
                    self.df.at[idx, 'P'] = np.nan
                    self.df.at[idx, 'Irr'] = np.nan
                else:
                    self.df.at[idx, 'P'] = 0.0
                    self.df.at[idx, 'Irr'] = 0.0

        print(f"\nDetected {len(anomalies)} correlation anomalies.")

    def generate_feature_nan_masks(self):
        print("\nSTEP 10: Generate NaN column per column")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")
        feature_columns = ["P", "Irr", "I", "U", "Temp"]
        created = 0
        nan_summary = {}  

        for col in feature_columns:
            if col not in self.df.columns:
                print(f"Warning: feature '{col}' not present, skipping mask creation.")
                continue

            mask_col = f"{col}_nan"

            #create mask column (1 = NaN, 0 = valid)
            self.df[mask_col] = self.df[col].isna().astype("uint8")
            nan_count = self.df[col].isna().sum()
            nan_summary[col] = nan_count

            created += 1

        print("-" * 100)
        print(f"{'Feature':<25} {'NaN Count':>10}")
        print("-" * 100)
        for col, cnt in nan_summary.items():
            print(f"{col:<25} {cnt:>10}")
        print("-" * 100)

    #Not using
    def fill_P_U_I(self):
        print("\nSTEP 9 : Fill P, U, I using P = U * I when possible")
        df = self.df.copy()
        def build_masks(var):
            invalid = (
                (df.get(f"invalid_physical_{var}", 0) == 1) |
                (df.get(f"invalid_statistical_{var}", 0) == 1) |
                (df.get(f"{var}_nan", 0) == 1)
            )
            valid = ~invalid
            return invalid, valid

        mask_P_invalid, mask_P_valid = build_masks("P")
        mask_U_invalid, mask_U_valid = build_masks("U")
        mask_I_invalid, mask_I_valid = build_masks("I")
        results = {"P_filled": 0, "U_filled": 0, "I_filled": 0}

        # 1.fill P when P invalid & U,I valid
        mask_fill_P = mask_P_invalid & mask_U_valid & mask_I_valid
        print(f"Rows eligible to fill P: {mask_fill_P.sum()}")

        if mask_fill_P.any():
            new_P = df.loc[mask_fill_P, "U"] * df.loc[mask_fill_P, "I"]
            valid_mask = ~new_P.isna()

            idx = new_P.index[valid_mask]
            df.loc[idx, "P"] = new_P[idx]

            # Clear only P flags
            for col in ["invalid_physical_P", "invalid_statistical_P", "P_nan"]:
                if col in df.columns:
                    df.loc[idx, col] = 0

            results["P_filled"] = len(idx)
            print(f"Filled P for {len(idx)} rows.")

        # 2.fill U when U invalid & P,I valid
        mask_fill_U = mask_U_invalid & mask_P_valid & mask_I_valid
        print(f"Rows eligible to fill U: {mask_fill_U.sum()}")

        if mask_fill_U.any():
            denom = df.loc[mask_fill_U, "I"].replace(0, np.nan)
            new_U = df.loc[mask_fill_U, "P"] / denom

            valid_mask = ~new_U.isna()
            idx = new_U.index[valid_mask]

            df.loc[idx, "U"] = new_U[idx]

            # Clear only U flags
            for col in ["invalid_physical_U", "invalid_statistical_U", "U_nan"]:
                if col in df.columns:
                    df.loc[idx, col] = 0

            results["U_filled"] = len(idx)
            print(f"Filled U for {len(idx)} rows.")

        # 3.fILL I when I invalid & P,U valid
        mask_fill_I = mask_I_invalid & mask_P_valid & mask_U_valid
        print(f"Rows eligible to fill I: {mask_fill_I.sum()}")

        if mask_fill_I.any():
            denom = df.loc[mask_fill_I, "U"].replace(0, np.nan)
            new_I = df.loc[mask_fill_I, "P"] / denom

            valid_mask = ~new_I.isna()
            idx = new_I.index[valid_mask]

            df.loc[idx, "I"] = new_I[idx]

            # Clear only I flags
            for col in ["invalid_physical_I", "invalid_statistical_I", "I_nan"]:
                if col in df.columns:
                    df.loc[idx, col] = 0

            results["I_filled"] = len(idx)
            print(f"Filled I for {len(idx)} rows.")

        self.df = df

    def combine_masks(self):
        print("\nCombining all flags into a single flag: invalid_any")
        print(f"Start date: {self.df['_time'].min()}, End date: {self.df['_time'].max()}")
        mask_cols = [
            'invalid_physical_P', 'invalid_physical_I', 'invalid_physical_U', 'invalid_physical_Irr', 'invalid_physical_Temp', 
            'invalid_outlier_P', 'invalid_outlier_I', 'invalid_outlier_U', 'invalid_outlier_Irr', 'invalid_outlier_Temp',
            'P_nan', 'I_nan', 'U_nan', 'Irr_nan', 'Temp_nan'
        ]
        self.df["invalid_any"] = self.df[mask_cols].max(axis=1)
        print(f"Total invalid rows: {self.df['invalid_any'].sum()}")

    def plot_histogram_per_panel(self):
        print("\nPlotting histogram of P per panel (day/night bins)")

        modules = sorted(self.df["Name"].unique())
        num_modules = len(modules)
        ncols = 4
        nrows = int(np.ceil(num_modules / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).reshape(-1)

        for i, module in enumerate(modules):
            ax = axes[i]

            # Full P data for module (for NaN, zero, min, max)
            mod_data = self.df[self.df["Name"] == module]["P"]
            nan_count = mod_data.isna().sum()
            zero_count = (mod_data == 0).sum()
            p_min = mod_data.min(skipna=True)
            p_max = mod_data.max(skipna=True)

            if mod_data.dropna().empty:
                ax.text(0.5, 0.5, f"{module}\nAll values zero/NaN",
                        ha="center", va="center", fontsize=10, color="red")
                ax.axis("off")
                continue

            max_bin = (int(np.ceil(p_max / 10)) + 1) * 10  
            bins = np.arange(0, max_bin + 10, 10)

            valid_mask = (
                (self.df["Name"] == module) &
                (self.df["invalid_physical_P"] == 0) &
                (self.df["invalid_outlier_P"] == 0) &
                (self.df["P_nan"] == 0)
            )
            valid_data = self.df.loc[valid_mask, ["P", "_time"]].copy()

            if valid_data.empty:
                ax.text(0.5, 0.5, f"{module}\nNo valid data",
                        ha="center", va="center", fontsize=10, color="red")
                ax.axis("off")
                continue

            # Split day and night data
            valid_data["is_night"] = ((valid_data["_time"].dt.hour >= 22) | (valid_data["_time"].dt.hour <= 4))
            day_vals = valid_data.loc[~valid_data["is_night"], "P"].dropna()
            night_vals = valid_data.loc[valid_data["is_night"], "P"].dropna()

            day_hist, _ = np.histogram(day_vals, bins=bins)
            night_hist, _ = np.histogram(night_vals, bins=bins)

            width = 4
            bin_centers = (bins[:-1] + bins[1:]) / 2

            ax.bar(bin_centers - width/2, day_hist, width=width, color="tab:blue", alpha=0.7, label="Day")
            ax.bar(bin_centers + width/2, night_hist, width=width, color="tab:orange", alpha=0.7, label="Night")

            ax.set_title(
                f"{module}\n"
                f"NaN: {nan_count} | Zero: {zero_count}\n"
                f"Min: {p_min:.2f} | Max: {p_max:.2f}",
                fontsize=9
            )
            ax.set_xlabel("P")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.set_xticks(bins)
            ax.set_xlim(bins[0], bins[-1])

        # Turn off extra axes
        for j in range(len(modules), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Histogram of P per Panel (Day vs Night)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        self._save_plot(fig, "histogram_p_per_panel_day_night")
        plt.close(fig)

    def plot_scatter_per_panel_P_Irr(self):
        print("\nPlotting P vs Irr scatter plot per panel")
        modules = sorted(self.df["Name"].unique())
        ncols = 4
        nrows = int(np.ceil(len(modules) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten()

        for i, module in enumerate(modules):
            ax = axes[i]

            mod_df = self.df[self.df["Name"] == module]

            valid_mask = (
                (mod_df["invalid_physical_P"] == 0) &
                (mod_df["invalid_outlier_P"] == 0) &
                (mod_df["P_nan"] == 0) &
                (mod_df["invalid_physical_Irr"] == 0) &
                (mod_df["invalid_outlier_Irr"] == 0) &
                (mod_df["Irr_nan"] == 0)
            )
            valid_data = mod_df.loc[valid_mask, ["P", "Irr", "_time"]].dropna()

            if valid_data.empty:
                ax.text(0.5, 0.5, f"{module}\nNo valid data",
                        ha="center", va="center", fontsize=10, color="red")
                ax.axis("off")
                continue

            valid_data["is_night"] = ((valid_data["_time"].dt.hour >= 22) | (valid_data["_time"].dt.hour <= 4))

            day_data = valid_data.loc[~valid_data["is_night"]]
            night_data = valid_data.loc[valid_data["is_night"]]

            ax.scatter(day_data["Irr"], day_data["P"], s=8, alpha=0.7, label="Day", color="tab:blue")
            ax.scatter(night_data["Irr"], night_data["P"], s=8, alpha=0.7, label="Night", color="red")

            nan_count_irr = mod_df["Irr"].isna().sum()
            zero_count_irr = (mod_df["Irr"] == 0).sum()
            nan_count_P = mod_df["P"].isna().sum()
            zero_count_P = (mod_df["P"] == 0).sum()
            p_min, p_max = mod_df["P"].min(skipna=True), mod_df["P"].max(skipna=True)
            irr_min, irr_max = mod_df["Irr"].min(skipna=True), mod_df["Irr"].max(skipna=True)

            ax.set_title(
                f"{module}\n"
                f"Irr NaN: {nan_count_irr} | Irr Zero: {zero_count_irr}\n"
                f"P NaN: {nan_count_P} | P Zero: {zero_count_P}\n"
                f"P[min,max]: {p_min:.2f} / {p_max:.2f}\n"
                f"Irr[min,max]: {irr_min:.2f} / {irr_max:.2f}",
                fontsize=9
            )
            ax.set_xlabel("Irr", fontsize=8)
            ax.set_ylabel("P", fontsize=8)
            ax.legend(fontsize=7)

        for j in range(len(modules), len(axes)):
            axes[j].axis("off")

        plt.suptitle("P-Irr Correlation Scatter per Panel (Night = red)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        self._save_plot(fig, "scatter_per_panel_P_Irr")
        plt.close(fig)

    #NEW
    def normalise_power_per_module(
        self,
        df: pd.DataFrame | None = None,
        power_col: str = "P",
        module_col: str = "Name",
        out_col: str = "P_normalised",
        quantile: float = 0.95,
        json_name: str = "power_normalisation_factors.json",
    ):
        """
        For each unique module:
        - Compute P_q95
        - Create P_normalised = P / P_q95
        - Save scaling factors + min/max to JSON
        """

        print("\nSTEP 11: Power normalisation per module")

        if df is None:
            if not hasattr(self, "df") or self.df is None:
                raise ValueError("WARNING: No DataFrame provided.")
            df = self.df
        else:
            df = df.copy()

        if power_col not in df.columns:
            raise ValueError(f"WARNING: Column '{power_col}' not found in DataFrame")
        if module_col not in df.columns:
            raise ValueError(f"WARNING: Column '{module_col}' not found in DataFrame")

        neg_mask = df[power_col] < 0
        n_neg = neg_mask.sum()

        if n_neg > 0:
            print(f"Clamping {n_neg} negative '{power_col}' values to 0")
            df.loc[neg_mask, power_col] = 0.0
        else:
            print("No negative power values found")

        scaling_stats = {}

        for module, g in df.groupby(module_col):
            power = g[power_col].dropna()

            if power.empty:
                print(f"{module}: no valid power values, skipping")
                continue

            p_q = power.quantile(quantile)

            if p_q <= 0 or np.isnan(p_q):
                print(f"{module}: invalid quantile value ({p_q}), skipping")
                continue

            min_p = power.min()
            max_p = power.max()

            # Apply normalisation
            module_mask = df[module_col] == module
            df.loc[module_mask, out_col] = (df.loc[module_mask, power_col] / p_q).clip(lower=0.0, upper=1.75) #1.2 for si

            # Normalised stats
            p_norm = df.loc[module_mask, out_col].dropna()
            min_p_norm = p_norm.min() if not p_norm.empty else np.nan
            max_p_norm = p_norm.max() if not p_norm.empty else np.nan

            scaling_stats[module] = {
                "quantile": quantile,
                "p_q": float(p_q),
                "min_P": float(min_p),
                "max_P": float(max_p),
                "min_P_normalised": float(min_p_norm),
                "max_P_normalised": float(max_p_norm),
                "rows_total": int(len(g)),
                "rows_non_nan": int(power.notna().sum()),
            }

            print(
                f"{module}: "
                f"P_q{int(quantile*100)}={p_q:.3f}, "
                f"P[min,max]=({min_p:.3f},{max_p:.3f}), "
                f"P_norm[min,max]=({min_p_norm:.3f},{max_p_norm:.3f}), "
                f"rows={len(g)}"
            )

        # Save JSON
        scaling_file = self.results_dir / json_name
        with open(scaling_file, "w") as f:
            json.dump(scaling_stats, f, indent=2)

        print(f"\nSaved power normalisation factors to: {scaling_file}")

        self.df = df
        return df
    
    #Name change: compute_p_normalized to compute_module_average    
    def compute_module_average(self):
        print("\nCompute per column average from all modules")
        timestamp_groups = self.df.groupby('_time')
        normalized_data = []
        for timestamp, group in timestamp_groups:
            row_data = {'_time': timestamp}
            
            for col in ['P','P_normalised', 'I', 'U', 'Temp', 'Irr']: #add P_normalised here if that is target
                values = group[col].dropna()
                if len(values) == 0:
                    row_data[col] = np.nan
                    continue
                
                # Exclude zeros if non-zero values exist
                non_zero_values = values[values != 0.0]
                zero_values = values[values == 0.0]
                
                if len(non_zero_values) > 0 and len(zero_values) > 0:
                    values = non_zero_values  # Ignore zeros
                
                # Outlier detection if 3 or more values remain
                if len(values) >= 3:
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        z_scores = abs((values - mean_val) / std_val)
                        # Keep values with z-score <= 2.5
                        filtered_values = values[z_scores <= 2.5]
                        if len(filtered_values) > 0:
                            values = filtered_values
                
                # Final aggregation with median
                row_data[col] = values.median() if len(values) > 0 else np.nan
            
            row_data['modules_available'] = len(group)
            row_data['modules_total'] = len(self.validation_modules)
            
            normalized_data.append(row_data)
        
        # Create average dataset
        averaged_df = pd.DataFrame(normalized_data)
        
        # Sort by timestamp
        averaged_df = averaged_df.sort_values('_time').reset_index(drop=True)
        print(f"Created normalized dataset: {len(averaged_df)} timestamps")
        
        # Save normalized dataset
        averaged_file = self.results_dir / "pv_normalized.csv"
        averaged_df.to_csv(averaged_file, index=False)
        print(f"Saved normalized data to: {averaged_file}")
        
        return averaged_df
    
    def plot_normalized_parameters_monthly(self, averaged_df):
        exclude_ranges_psc = [
            ("2024-12-06", "2024-12-14"),
            ("2024-12-30", "2024-12-31"),
            ("2025-01-02", "2025-01-03"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-25", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-18", "2025-11-26"),
        ]

        exclude_ranges_si = [
            ("2024-12-06", "2024-12-14"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-16", "2025-11-26"),
            ("2025-12-21", "2025-12-24"),
        ]

        print("\nPlotting normalized P, I, U, Temp per month")

        if averaged_df.empty:
            print("No data in normalized dataframe")
            return

        averaged_df = averaged_df.copy()
        averaged_df['_time'] = pd.to_datetime(averaged_df['_time'], utc=True)

        if 'category' in averaged_df.columns:
            cats = averaged_df['category'].dropna().str.lower().unique()
            if len(cats) == 1 and cats[0] == 'si':
                active_ranges = exclude_ranges_si
                print("Using SI exclude ranges")
            else:
                active_ranges = exclude_ranges_psc
                print("Using PSC exclude ranges")
        else:
            # no category column → use both
            active_ranges = exclude_ranges_psc + exclude_ranges_si
            print("No category column → using BOTH PSC + SI exclude ranges")

        exclude_intervals = [
            (pd.Timestamp(start).tz_localize('UTC'),
            pd.Timestamp(end).tz_localize('UTC'))
            for start, end in active_ranges
        ]

        parameters = ['P', 'I', 'U', 'Temp', 'Irr']
        months = sorted(averaged_df['_time'].dt.tz_convert(None).dt.to_period('M').unique())
        n_months = len(months)
        ncols = min(4, n_months)
        nrows = (n_months + ncols - 1) // ncols

        for param in parameters:
            if param not in averaged_df.columns:
                print(f"Parameter '{param}' not found in normalized data, skipping")
                continue

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            if n_months == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for idx, month_period in enumerate(months):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                month_start = month_period.start_time.tz_localize('UTC')
                month_end = month_period.end_time.tz_localize('UTC')

                month_data = averaged_df[(averaged_df['_time'] >= month_start) & (averaged_df['_time'] <= month_end)].copy().sort_values('_time')

                if month_data.empty:
                    ax.text(0.5, 0.5, f"No data\n{month_period}",
                            ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f"{month_period}", fontsize=10)
                    ax.axis('off')
                    continue

                valid_data = month_data[month_data[param].notna()]
                if not valid_data.empty:
                    ax.plot(valid_data['_time'], valid_data[param], color='blue', linewidth=1, alpha=0.8, marker='.', markersize=2)

                for start, end in exclude_intervals:
                    if end >= month_start and start <= month_end:
                        highlight_start = max(start, month_start)
                        highlight_end = min(end, month_end)
                        ax.axvspan(highlight_start, highlight_end, color='red', alpha=0.3)

                ax.set_title(f"{month_period}", fontsize=10, fontweight='bold')
                ax.set_ylabel(param, fontsize=9)
                ax.grid(True, alpha=0.3)

                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(byweekday=0))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

                if not valid_data.empty:
                    stats_text = (
                        f"Min: {valid_data[param].min():.1f}\n"
                        f"Max: {valid_data[param].max():.1f}\n"
                        f"Points: {len(valid_data)}"
                    )
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            for idx in range(len(months), len(axes)):
                axes[idx].axis('off')

            start_date = averaged_df['_time'].min()
            end_date = averaged_df['_time'].max()

            fig.suptitle(
                f"Monthly {param} - Normalized Dataset\n"
                f"Time Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
                f"Red areas indicate exclude date ranges",
                fontsize=12, fontweight='bold'
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_plot(fig, f"monthly_{param}_normalized")
            plt.close(fig)

        print(f"\nAll monthly parameter plots saved to: {self.results_dir}/")

    def plot_normalized_parameters_detailed(self, averaged_df):
        exclude_ranges_psc = [
            ("2024-12-06", "2024-12-14"),
            ("2024-12-30", "2024-12-31"),
            ("2025-01-02", "2024-01-03"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-18", "2025-11-23"),
        ]

        exclude_ranges_si = [
            ("2024-12-06", "2024-12-14"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-16", "2025-11-26"),
            ("2025-12-21", "2025-12-24"),
        ]

        if averaged_df.empty:
            print("No data in normalized dataframe")
            return

        averaged_df = averaged_df.copy()
        averaged_df['_time'] = pd.to_datetime(averaged_df['_time'], utc=True)

        if 'category' in averaged_df.columns:
            cats = averaged_df['category'].dropna().str.lower().unique()
            if len(cats) == 1 and cats[0] == 'si':
                active_ranges = exclude_ranges_si
                print("Using SI exclude ranges")
            else:
                active_ranges = exclude_ranges_psc
                print("Using PSC exclude ranges")
        else:
            active_ranges = exclude_ranges_psc + exclude_ranges_si
            print("No category column → using BOTH PSC + SI exclude ranges")

        exclude_intervals = [(pd.Timestamp(start).tz_localize('UTC'), pd.Timestamp(end).tz_localize('UTC')) for start, end in active_ranges]

        parameters = ['P', 'I', 'U', 'Temp', 'Irr']

        for param in parameters:
            if param not in averaged_df.columns:
                continue

            print(f"\nCreating detailed timeline for: {param}")

            fig, ax = plt.subplots(figsize=(26, 6))

            plot_data = averaged_df[['_time', param]].sort_values('_time')
            valid_data = plot_data[plot_data[param].notna()]

            if not valid_data.empty:
                ax.plot(valid_data['_time'], valid_data[param], linewidth=1, alpha=0.7, color='blue',
                        marker='.', markersize=1, label=f'{param} (n={len(valid_data)})')

            for start, end in exclude_intervals:
                ax.axvspan(start, end, alpha=0.2, color='red')

            ax.set_title(f"{param} - Complete Timeline with Exclude Date Highlighting\n"
                f"Red areas indicate exclude date ranges",
                fontsize=14, fontweight='bold')
            ax.set_ylabel(param, fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            ax.legend(loc='upper right', fontsize=10)

            stats_text = (
                f"Data Points: {len(valid_data):,}\n"
                f"Time Range: {plot_data['_time'].min().strftime('%Y-%m-%d')} to "
                f"{plot_data['_time'].max().strftime('%Y-%m-%d')}\n"
                f"Coverage: {len(valid_data) / len(plot_data) * 100:.1f}%"
            )

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            self._save_plot(fig, f"detailed_timeline_{param}_normalized")
            plt.close(fig)

            print(f"Detailed {param} timeline: {len(valid_data):,} points")
    
    def run(self, dwd_file, flag_invalid=False):
        """
        Main pipeline execution sequence.

        Args:
            dwd_file (str): Path to DWD irradiance csv file.
            flag_invalid (bool): 
                True  : Keep invalid rows but flag them (for model masking).
                False : Drop all invalid rows.
        """
        print(f"\nValidation modules: {self.validation_modules}")
        print(f"\nReference modules: {self.cross_module_ref} (cross-ref), {self.scaling_ref_module} (scaling)")

        #cleans names
        self.clean_module_names()
        #remove data before installation dates, duplicates, convert to UTC
        self.align_resample()
        #create copies of scaling_ref_module and cross_module_ref
        self.create_reference_copies()
        #drop modules other than validation_moduels 
        self.drop_non_validation_modules()
        #fill Irr column with scaling_ref_module + cross_module_ref + DWD data and save scaling factors in JSON, 
        #File path: result_data_validation/dwd_irradiance_scaling_factors.json
        #Plot Irr comparison between park sensors and DWD irr data 
        #File path: result_data_validation/Irradiance_comparison_side_by_side.png
        self.fill_irradiance_gaps(dwd_file, cross_module_ref="Sanyo_5_1", scaling_ref_module="Perovskite_1_1")
        #clean night time data 
        self.night_detector()
        #Add columns based on name_mapping_dict
        self.map_module_metadata()
        self.plot_module_metadata_summary(out_png=self.results_dir / "module_metadata_summary.png")
        #apply range validation as per module type and flag them as invalid_physical_*
        self.range_validation()
        #filter based on z-score and flag them as invalid_statistical_*
        self.detect_statistical_outliers(z_thresh=3.0)
        #NEW
        self.detect_correlation_anomalies(mark_as_nan=False)
        #filter nan values and flag them as *_nan
        self.generate_feature_nan_masks()
        #fill P, I, U based on P=U*I relation
        #self.fill_P_U_I()
        #combine all flags as invalid_any
        self.combine_masks()    
        #Drop or keep invalids based on flag_invalid
        if not flag_invalid:
            print("\nDropping invalid rows")
            before_total = len(self.df)
            panel_stats_before = self.df.groupby("Name").size()
            invalid_stats_before = self.df.groupby("Name")["invalid_any"].sum()
            
            print(f"Before dropping total rows: {before_total}")
            print("\nPer-panel (Before):")
            for panel in sorted(self.df["Name"].unique()):
                panel_total = panel_stats_before.get(panel, 0)
                panel_invalid = invalid_stats_before.get(panel, 0)
                panel_valid = panel_total - panel_invalid
                valid_pct = (panel_valid / panel_total * 100) if panel_total > 0 else 0
                print(f"{panel:15s}: {int(panel_valid):5d} / {int(panel_total):5d} valid ({valid_pct:5.1f}%)")
            
            #drop
            self.df = self.df[self.df["invalid_any"] == 0]

            after_total = len(self.df)
            panel_stats_after = self.df.groupby("Name").size()
            
            print(f"\nAfter dropping total rows: {after_total}")
            print(f"Rows dropped: {before_total - after_total} ({(before_total - after_total)/before_total*100:.1f}% of total)")
            print(f"Overall data survived: {after_total/before_total*100:.1f}%")
            
        else:
            print("\nKeeping invalid rows, using flags for masking.")
            print(f"Total rows: {len(self.df)}")
            print(f"Invalid rows: {self.df['invalid_any'].sum()} ({(self.df['invalid_any'].sum()/len(self.df)*100):.1f}%)")
            
            invalid_by_panel = self.df.groupby("Name")["invalid_any"].agg(['sum', 'count'])
            invalid_by_panel['pct_invalid'] = (invalid_by_panel['sum'] / invalid_by_panel['count'] * 100)
            
            print("\nPer-panel invalid statistics:")
            for panel, row in invalid_by_panel.iterrows():
                print(f"  {panel:15s}: {int(row['sum']):3d} / {int(row['count']):5d} invalid ({row['pct_invalid']:5.1f}%)")

        #Plots 
        self.plot_histogram_per_panel()
        self.plot_scatter_per_panel_P_Irr()
        #Normalise power per module using p95 
        self.normalise_power_per_module()
        #compute average per column and create new df 
        averaged_df =self.compute_module_average()
        self.plot_normalized_parameters_monthly(averaged_df)
        self.plot_normalized_parameters_detailed(averaged_df)

        #save final outputs
        cleaned_file = self.results_dir / "pv_cleaned_masked.csv"
        self.df.to_csv(cleaned_file, index=False)
        
        print(f"\nCleaned per-module dataset saved: {cleaned_file}")
        print(f"Rows: {len(self.df):,}, Columns: {len(self.df.columns)}")
        print(f"Modules: {sorted(self.df['Name'].unique())}")
        
        # Save normalized dataset
        averaged_file = self.results_dir / "pv_normalized.csv"
        averaged_df.to_csv(averaged_file, index=False)
        print(f"\nNormalized dataset saved: {averaged_file}")
        print(f"Rows: {len(averaged_df):,}, Columns: {len(averaged_df.columns)}")
                
        return self.df, averaged_df
