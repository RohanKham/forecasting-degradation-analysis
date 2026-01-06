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
                "Perovskite_1": "Perovskite_1_1", 
                "Perovskite_2": "Perovskite_1_2",
            }

            self.install_dates = {
                "Sanyo_1_1" : "2024-07-26 08:00:00",
                "Sanyo_2_1" : "2024-07-26 08:00:00", 
                "Sanyo_3_1" : "2024-07-26 08:00:00",
                "Sanyo_4_1" : "2024-07-26 08:00:00",
                "Sanyo_5_1" : "2024-07-26 08:00:00",
                "Solon_1_1" : "2024-07-26 08:00:00",
                "Solon_1_2" : "2024-07-26 08:00:00", 
                "Solon_2_1" : "2024-07-26 08:00:00",
                "Solon_2_2" : "2024-07-26 08:00:00",
                "Solon_3_1" : "2024-07-26 08:00:00",
                "Solon_3_2" : "2024-07-26 08:00:00",
                "Solon_4_2" : "2024-07-26 08:00:00",
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
        self.df["Name"] = self.df["Name"].astype(str).replace(self.panel_name_fixes)

    #remove faulty data before installation date
    def align_resample(self) -> None:
        print("\nSTEP 2: Removing pre-install & duplicate data")
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
        self.df = self.df[self.df["Name"].isin(self.validation_modules)]
        print("Modules kept:")
        for module, g in self.df.groupby("Name"):
            n_unique_ts = g["_time"].nunique()
            print(f"{module}: {n_unique_ts:,} timestamps")

    """
        Fill Irradiance gaps in Perovskite_1_1 and apply to all modules:
        1. Clean both reference modules 
        2. Process scaling reference module (Perovskite_1_1) using:
        - It's own cleaned sensor data
        - Cleaned cross-module reference (Sanyo_5_1) 
        - Scaled DWD data
        3. Copy the processed Irr column to all validation modules (exact timestamp match only)
        4. Fill remaining gaps in other modules using scaled DWD (L4)
    """
    def fill_irradiance_gaps(self, dwd_file, cross_module_ref=None, scaling_ref_module=None):        
        print("\nSTEP 4 : Fill Irradiance gaps")

        if cross_module_ref:
            self.cross_module_ref = cross_module_ref
        if scaling_ref_module:
            self.scaling_ref_module = scaling_ref_module

        def load_and_clean_cross_module_ref():
            data = self.original_cross_ref.copy()
            if data is None or data.empty:
                print(f"Cross-module reference '{self.cross_module_ref}' is empty")
                return None
            original_count = len(data)
            data["_time"] = pd.to_datetime(data["_time"], utc=True)
            data = data.set_index("_time").sort_index()
            original_na = data["Irr"].isna().sum()
            #identify outliers
            outlier_mask = (data["Irr"] <= 0) | (data["Irr"] > 1300)          
            data_clean = data.copy()
            #mark outlier as Nan
            data_clean.loc[outlier_mask, "Irr"] = np.nan  
            final_na = data_clean["Irr"].isna().sum()
            new_outliers_marked = final_na - original_na
            print(f"\nCross-module ref '{self.cross_module_ref}': "
                  f"Rows: {original_count}, "
                  f"Marked {new_outliers_marked} outliers as NaN, "
                  f"Total NaN: {final_na}")
            
            return data_clean["Irr"].rename(f"Irr_{self.cross_module_ref}")

        def load_and_clean_scaling_ref():
            data = self.original_scaling_ref.copy()
            if data is None or data.empty:
                raise ValueError(f"Scaling reference '{self.scaling_ref_module}' is not available!")
            original_count = len(data)
            data["_time"] = pd.to_datetime(data["_time"], utc=True)
            data = data.set_index("_time").sort_index()
            original_na = data["Irr"].isna().sum()           
            outlier_mask = (data["Irr"] <= 0) | (data["Irr"] > 1300)            
            data_clean = data.copy()
            data_clean.loc[outlier_mask, "Irr"] = np.nan  
            final_na = data_clean["Irr"].isna().sum()
            new_outliers_marked = final_na - original_na   
            print(f"Scaling ref '{self.scaling_ref_module}': "
                  f"Rows: {original_count}, "
                  f"Marked {new_outliers_marked} outliers as NaN, "
                  f"Total NaN: {final_na}")
            
            return data_clean.rename(columns={"Irr": "Irr_ref"})
        
        #save DWD irradiance scaling factors to JSON file
        def save_dwd_scaling_factors(doy_period_factors, hourly_fallback):
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

        #load both references
        sanyo_ref_clean = load_and_clean_cross_module_ref()
        scaling_ref_data_clean = load_and_clean_scaling_ref()

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

        scaling_start = scaling_ref_data_clean.index.min()
        scaling_end = scaling_ref_data_clean.index.max()
        dwd = dwd.loc[scaling_start:scaling_end]  
        
        merged_clean = pd.concat([scaling_ref_data_clean, dwd], axis=1)
        
        #doy scaling 
        doy_period_factors = {}
        hourly_fallback = {}

        periods = {
            "morning": range(4, 11),  
            "midday": range(11, 17),  
            "evening": range(17, 22)  
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
        
        plot_month_period_scaling_factors()

        #hr fallback
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


        self.dwd_scaling_file = save_dwd_scaling_factors(doy_period_factors, hourly_fallback)

        def apply_doy_period_scaling(ts, dwd_val):
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

        print(f"\nFilling gaps in scaling reference: {self.scaling_ref_module}")
        scaling_ref_final = scaling_ref_data_clean.copy()
        remaining_na = scaling_ref_final["Irr_ref"].isna().sum()
        print(f"Initial gaps in cleaned scaling ref: {remaining_na}")
        
        l2_filled = 0
        l3_filled = 0

        #1.Cross-module reference filling
        if sanyo_ref_clean is not None and remaining_na > 0:
            missing_idx = scaling_ref_final[scaling_ref_final["Irr_ref"].isna()].index
            sanyo_fill = sanyo_ref_clean.reindex(missing_idx)
            valid_fill = sanyo_fill.dropna()
            l2_filled = len(valid_fill)
            scaling_ref_final.loc[valid_fill.index, "Irr_ref"] = valid_fill
            remaining_na = scaling_ref_final["Irr_ref"].isna().sum()

        #2.Scaled DWD filling
        if remaining_na > 0:
            missing_idx = scaling_ref_final[scaling_ref_final["Irr_ref"].isna()].index
            dwd_missing = dwd.reindex(missing_idx)
            
            for ts in dwd_missing.dropna().index:
                scaled_val = apply_doy_period_scaling(ts, dwd_missing.at[ts, "Irr_dwd"])
                if not pd.isna(scaled_val):
                    scaling_ref_final.loc[ts, "Irr_ref"] = scaled_val
                    l3_filled += 1
            
            remaining_na = scaling_ref_final["Irr_ref"].isna().sum()


        scaling_ref_final = scaling_ref_final[scaling_ref_final["Irr_ref"].notna()]
        
        total_filled = l2_filled + l3_filled
        total_possible = len(scaling_ref_data_clean)
        final_coverage = len(scaling_ref_final) / total_possible * 100 if total_possible > 0 else 0
        
        print(f"{self.scaling_ref_module}: Total filled {total_filled}, "
            f"Final coverage: {final_coverage:.1f}% "
            f"[L2(cross-module): {l2_filled}, L3(DWD): {l3_filled}]")

        print(f"\nCopying processed Irr data to all validation modules")

        irr_reference_df = scaling_ref_final[["Irr_ref"]].reset_index()
        irr_reference_df.columns = ["_time", "Irr_processed"]

        all_processed = []
        modules_processed = []

        for module, sub in self.df.groupby("Name"):
            if module not in self.validation_modules:
                continue
                
            sub = sub.copy()
            sub["_time"] = pd.to_datetime(sub["_time"], utc=True)
            
            # merge on exact timestamp only 
            before_merge = len(sub)
            sub_merged = pd.merge(
                sub,
                irr_reference_df,
                on="_time",
                how="left"  
            )
            
            #mismatches remain NaN
            sub_merged["Irr"] = sub_merged["Irr_processed"]
            sub_merged = sub_merged.drop(columns=["Irr_processed"])
            
            exact_matches = sub_merged["Irr"].notna().sum()
            mismatches = sub_merged["Irr"].isna().sum()
            match_percentage = (exact_matches / len(sub_merged)) * 100
            
            modules_processed.append(module)
            print(f"{module}: {exact_matches}/{len(sub_merged)} exact timestamp matches ({match_percentage:.1f}%) - {mismatches} gaps for L4")
            all_processed.append(sub_merged)

        #combine all modules
        self.df = pd.concat(all_processed, ignore_index=True)
        
        print(f"\nL4: Filling all remaining gaps in modules using scaled DWD")
        
        l4_filled_total = 0
        modules_l4_stats = []
        
        for module in modules_processed:
            module_mask = self.df["Name"] == module
            module_na_before = self.df.loc[module_mask, "Irr"].isna().sum()
            
            if module_na_before > 0:
                #Get timestamps with missing Irr for this module
                missing_times = self.df.loc[module_mask & self.df["Irr"].isna(), "_time"]
                
                l4_filled_module = 0
                for ts in missing_times:
                    ts_dt = pd.to_datetime(ts, utc=True)
                    #Get DWD value for this timestamp
                    dwd_val = dwd.reindex([ts_dt])["Irr_dwd"].iloc[0] if ts_dt in dwd.index else np.nan
                    
                    if not pd.isna(dwd_val):
                        #Apply hourly scaling
                        scaled_val = apply_doy_period_scaling(ts_dt, dwd_val)
                        if not pd.isna(scaled_val):
                            #Update the value in the dataframe
                            self.df.loc[(self.df["_time"] == ts) & (self.df["Name"] == module), "Irr"] = scaled_val
                            l4_filled_module += 1
                
                module_na_after = self.df.loc[module_mask, "Irr"].isna().sum()
                l4_filled_total += l4_filled_module
                
                modules_l4_stats.append((module, module_na_before, l4_filled_module, module_na_after))
        
        for module, na_before, l4_filled, na_after in modules_l4_stats:
            if na_before > 0:
                coverage_improvement = (l4_filled / na_before * 100) if na_before > 0 else 0
                print(f"{module}: {na_before} gaps : {l4_filled} filled : {na_after} remaining ({coverage_improvement:.1f}% filled)")
            else:
                print(f"{module}: No gaps to fill")
        
        print(f"Total L4 gaps filled: {l4_filled_total}")
        
        #remove any remaining unfillable gaps
        self.df = self.df[self.df["Irr"].notna()]
        
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

        #Plot comparisons between DWD and scaling_ref_module, cross_module_ref
        self.plot_irradiance_comparison(scaling_ref_final, dwd, sanyo_ref_clean)
        
        return self.df

    def plot_irradiance_comparison(self, scaling_ref_final, dwd, sanyo_ref_clean):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #Plot 1: DWD vs Scaling Reference
        merged_plot = pd.concat([scaling_ref_final, dwd], axis=1).dropna(subset=["Irr_ref", "Irr_dwd"])

        ax = axes[0]
        ax.scatter(merged_plot["Irr_dwd"], merged_plot["Irr_ref"], s=10, alpha=0.5, c='blue')
        max_val = merged_plot["Irr_dwd"].max()
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_xlabel("DWD Global Radiation [W/m²]")
        ax.set_ylabel(f"{self.scaling_ref_module} Irradiance [W/m²]")
        ax.set_title("DWD vs Final Scaling Reference Module")
        ax.grid(True)

        #Plot 2: DWD vs Cross-Module Reference 
        ax = axes[1]

        if sanyo_ref_clean is not None:
            def ensure_df(obj):
                return obj.to_frame() if isinstance(obj, pd.Series) else obj
            sanyo_ref_df = ensure_df(sanyo_ref_clean)
            dwd_df = ensure_df(dwd)
            scaling_ref_df = ensure_df(scaling_ref_final)

            sanyo_ref_df.index = sanyo_ref_df.index.round("10min")
            dwd_df.index = dwd_df.index.round("10min")
            overlap = sanyo_ref_df.index.intersection(dwd_df.index)
            merged_cross = sanyo_ref_df.join(dwd_df, how="inner")
            
            ax.scatter(
                merged_cross["Irr_dwd"],
                merged_cross[f"Irr_{self.cross_module_ref}"],
                s=10, alpha=0.5, c='green'
            )
            max_val2 = merged_cross["Irr_dwd"].max()
            ax.plot([0, max_val2], [0, max_val2], 'r--')

            ax.set_xlabel("DWD Global Radiation [W/m²]")
            ax.set_ylabel(f"{self.cross_module_ref} Irradiance [W/m²]")
            ax.set_title("DWD vs Cross-Module Reference")
            ax.grid(True)

        else:
            ax.text(0.5, 0.5,"No Cross-Module Reference available",
                    ha='center', va='center', fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        self._save_plot(fig,"Irradiance_comparison_side_by_side")
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

        print("\nSTEP 5 : Night data handling")

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

    def range_validation(self):
        print("\nSTEP 6 : Physical range validation (daytime only)")

        night_mask = (
            ((self.df["_time"].dt.hour >= 22) | (self.df["_time"].dt.hour <= 4)) & (self.df["Irr"] < 1)
        )

        bounds_by_type = {
            "Si": {
                "P": (0, 220),
                "I": (0.1, 7),
                "U": (0, 60),
                "Irr": (0, 1300),
                "Temp": (-20, 100),
            },
            "PSC": {
                "P": (0, 110),
                "I": (0.1, 1.4),
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

        for module, sub in df_day.groupby("Name"):
            if "Sanyo" in module or "Solon" in module:
                mtype = "Si"
            elif "Perovskite" in module:
                mtype = "PSC"
            else:
                mtype = "Si"

            print(f"\n{module} ({mtype}) - daytime rows: {len(sub):,}")
            print("Param | Invalid | Min(day) | Max(day)")
            print("-" * 40)

            for p in params:
                lo, hi = bounds_by_type[mtype][p]
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
        print("\nSTEP 7 : Z-Score based Outlier detection")
        
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

    def generate_feature_nan_masks(self):
        print("\nSTEP 8 : Generate NaN column per column")
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

    def compute_p_normalized(self):
        print("\nCompute P_normalized")
        timestamp_groups = self.df.groupby('_time')
        normalized_data = []
        for timestamp, group in timestamp_groups:
            row_data = {'_time': timestamp}            
            for col in ['P', 'I', 'U', 'Temp', 'Irr']:
                available_vals = group[col].dropna()
                if len(available_vals) > 0:
                    row_data[col] = available_vals.median() #should be mean or median?
                else:
                    row_data[col] = np.nan
            
            row_data['modules_available'] = len(group)
            row_data['modules_total'] = len(self.validation_modules)
            
            normalized_data.append(row_data)
        
        #create normalized dataset
        normalized_df = pd.DataFrame(normalized_data)
        
        #sort by timestamp
        normalized_df = normalized_df.sort_values('_time').reset_index(drop=True)
        print(f"Created normalized dataset: {len(normalized_df)} timestamps")
        
        #Save normalized dataset
        normalized_file = self.results_dir / "pv_normalized.csv"
        normalized_df.to_csv(normalized_file, index=False)
        print(f"Saved normalized data to: {normalized_file}")
        
        return normalized_df
    
    def plot_normalized_parameters_monthly(self, normalized_df):
        exclude_ranges = [
            ("2024-12-06", "2024-12-14"),
            ("2024-12-30", "2024-12-31"),
            ("2025-01-02", "2024-01-03"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-18", "2025-11-23"),
        ]
        print("\nPlotting normalized P, I, U, Temp per month")

        if normalized_df.empty:
            print("No data in normalized dataframe")
            return

        normalized_df = normalized_df.copy()
        normalized_df['_time'] = pd.to_datetime(normalized_df['_time'], utc=True)
        parameters = ['P', 'I', 'U', 'Temp', 'Irr']
        exclude_intervals = [(pd.Timestamp(start).tz_localize('UTC'), pd.Timestamp(end).tz_localize('UTC')) for start, end in exclude_ranges]
        months = sorted(normalized_df['_time'].dt.tz_convert(None).dt.to_period('M').unique())
        n_months = len(months)
        ncols = min(4, n_months)
        nrows = (n_months + ncols - 1) // ncols

        for param in parameters:
            if param not in normalized_df.columns:
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

                month_data = normalized_df[(normalized_df['_time'] >= month_start) & (normalized_df['_time'] <= month_end)].copy()
                month_data = month_data.sort_values('_time')

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
                    stats_text = f"Min: {valid_data[param].min():.1f}\nMax: {valid_data[param].max():.1f}\nPoints: {len(valid_data)}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            for idx in range(len(months), len(axes)):
                axes[idx].axis('off')

            start_date = normalized_df['_time'].min()
            end_date = normalized_df['_time'].max()

            fig.suptitle(f"Monthly {param} - Normalized Dataset\n"
                        f"Time Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
                        f"Red areas indicate exclude date ranges",
                        fontsize=12, fontweight='bold')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self._save_plot(fig, f"monthly_{param}_normalized")
            plt.close(fig)

        print(f"\nAll monthly parameter plots saved to: {self.results_dir}/")

    def plot_normalized_parameters_detailed(self, normalized_df):
        exclude_ranges = [
            ("2024-12-06", "2024-12-14"),
            ("2024-12-30", "2024-12-31"),
            ("2025-01-02", "2024-01-03"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-18", "2025-11-23"),
        ]
        if normalized_df.empty:
            print("No data in normalized dataframe")
            return

        normalized_df = normalized_df.copy()
        normalized_df['_time'] = pd.to_datetime(normalized_df['_time'], utc=True)

        parameters = ['P', 'I', 'U', 'Temp', 'Irr']

        exclude_intervals = [(pd.Timestamp(start).tz_localize('UTC'), pd.Timestamp(end).tz_localize('UTC')) for start, end in exclude_ranges]

        for param in parameters:
            if param not in normalized_df.columns:
                continue

            print(f"\nCreating detailed timeline for: {param}")

            fig, ax = plt.subplots(figsize=(26, 6))

            plot_data = normalized_df[['_time', param]].sort_values('_time')
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

            stats_text = (f"Data Points: {len(valid_data):,}\n"
                        f"Time Range: {plot_data['_time'].min().strftime('%Y-%m-%d')} to {plot_data['_time'].max().strftime('%Y-%m-%d')}\n"
                        f"Coverage: {len(valid_data) / len(plot_data) * 100:.1f}%")

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
                True  → Keep invalid rows but flag them (for model masking).
                False → Drop all invalid rows.
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
        self.fill_irradiance_gaps(dwd_file, cross_module_ref="Perovskite_1_1", scaling_ref_module="Sanyo_5_1")
        #clean night time data 
        self.night_detector()
        #apply range validation as per module type and flag them as invalid_physical_*
        self.range_validation()
        #filter based on z-score and flag them as invalid_statistical_*
        self.detect_statistical_outliers(z_thresh=2.0)
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
        
        #compute P_normalized and create new df 
        normalized_df =self.compute_p_normalized()
        self.plot_normalized_parameters_monthly(normalized_df)
        self.plot_normalized_parameters_detailed(normalized_df)

        #save final outputs
        cleaned_file = self.results_dir / "pv_cleaned_masked.csv"
        self.df.to_csv(cleaned_file, index=False)
        
        print(f"\nCleaned per-module dataset saved: {cleaned_file}")
        print(f"Rows: {len(self.df):,}, Columns: {len(self.df.columns)}")
        print(f"Modules: {sorted(self.df['Name'].unique())}")
        
        # Save normalized dataset
        normalized_file = self.results_dir / "pv_normalized.csv"
        normalized_df.to_csv(normalized_file, index=False)
        print(f"\nNormalized dataset saved: {normalized_file}")
        print(f"Rows: {len(normalized_df):,}, Columns: {len(normalized_df.columns)}")
                
        return self.df, normalized_df
