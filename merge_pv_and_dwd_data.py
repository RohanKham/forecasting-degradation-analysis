import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

class PVDWDDataMerger:
    def __init__(self, dwd_data_path: str, scaling_factors_path: str, dwd_columns: Optional[List[str]] = None):
        """
            Class to merge PV system data with DWD weather data using DWD time as backbone.
        """
        self.dwd_data_path = dwd_data_path
        self.scaling_factors_path = scaling_factors_path
        self.dwd_columns = dwd_columns
        self.dwd_df = None
        self.scaling_factors = None
        
    def check_timezone_utc(self, df: pd.DataFrame, df_name: str) -> bool:
        if df.index.tz is None:
            print(f"{df_name}: No timezone info - assuming UTC")
            return True
        elif str(df.index.tz) == 'UTC':
            return True
        else:
            print(f"{df_name}: NOT UTC ({df.index.tz})")
            return False
    
    def load_scaling_factors(self) -> bool:
        try:
            if not os.path.exists(self.scaling_factors_path):
                print(f"Scaling factors file not found: {self.scaling_factors_path}")
                return False
            with open(self.scaling_factors_path, 'r') as f:
                self.scaling_factors = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Failed to load scaling factors: {e}")
            return False
    
    def apply_doy_period_scaling(self, timestamp: pd.Timestamp, dwd_irradiance: float) -> float:
        if pd.isna(dwd_irradiance) or dwd_irradiance <= 0:
            return 0.0

        doy = timestamp.dayofyear
        hour = timestamp.hour

        periods = {
            "morning": range(4, 11),  
            "midday": range(11, 17),  
            "evening": range(17, 22),
        }

        period_name = None
        for pname, hrs in periods.items():
            if hour in hrs:
                period_name = pname
                break

        if period_name is None:
            return dwd_irradiance

        key = f"{doy:03d}-{period_name}"
        factor_entry = self.scaling_factors.get("dayofyear_period", {}).get(key)

        if factor_entry and "factor" in factor_entry:
            factor = float(factor_entry["factor"])
            if factor > 0:
                return dwd_irradiance * factor

        fallback_factor = self.scaling_factors.get("period_fallback", {}).get(period_name, 0.0)
        if fallback_factor > 0:
            return dwd_irradiance * fallback_factor

        return dwd_irradiance
    
    def load_and_parse_dwd(self) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists(self.dwd_data_path):
                print(f"DWD data file not found: {self.dwd_data_path}")
                return None
            
            dwd_df = pd.read_csv(self.dwd_data_path)
            
            if 'timestamp' not in dwd_df.columns:
                print("No timestamp column found in DWD data")
                return None
            
            timestamp_col = 'timestamp'
            
            dwd_df[timestamp_col] = pd.to_datetime(dwd_df[timestamp_col], errors='coerce')
            
            nat_count = dwd_df[timestamp_col].isna().sum()
            if nat_count > 0:
                print(f"Found {nat_count} invalid timestamps (NaT). Removing them.")
                dwd_df = dwd_df.dropna(subset=[timestamp_col])
            
            initial_count = len(dwd_df)
            dwd_df = dwd_df.drop_duplicates(subset=[timestamp_col])  
            duplicates_removed = initial_count - len(dwd_df)

            if len(dwd_df) == 0:
                print("No valid DWD data remaining.")
                return None

            dwd_df.set_index(timestamp_col, inplace=True)
            dwd_df = dwd_df.sort_index()

            print(f"DWD: Loaded {len(dwd_df)} records")
            print(f"Time range: {dwd_df.index.min()} to {dwd_df.index.max()}")
            print(f"Removed {duplicates_removed} duplicate timestamps")
            
            is_dwd_utc = self.check_timezone_utc(dwd_df, "DWD")
            if not is_dwd_utc:
                print("DWD data is not UTC. Stopping.")
                return None

            default_weather_columns = ["temp_C", "humidity", "precip_mm", "precip_indicator", "cloud_cover", "global_radiation"]
            
            if self.dwd_columns is None:
                self.dwd_columns = default_weather_columns
                print(f"Default weather columns: {self.dwd_columns}")
            else:
                print(f"Requested columns: {self.dwd_columns}")

            existing_requested = [c for c in self.dwd_columns if c in dwd_df.columns]
            missing_columns = [c for c in self.dwd_columns if c not in dwd_df.columns]
            
            if missing_columns:
                print(f"Requested columns not found: {missing_columns}")
            
            columns_to_keep = existing_requested 
            dwd_df = dwd_df[columns_to_keep]
                            
            expected_index = pd.date_range(
                start=dwd_df.index.min(),
                end=dwd_df.index.max(),
                freq="10min"
            )

            missing_timestamps = expected_index.difference(dwd_df.index)
            extra_timestamps = dwd_df.index.difference(expected_index)

            if missing_timestamps.empty and extra_timestamps.empty:
                print("DWD timestamps are perfectly continuous at 10-minute intervals.")
            else:
                print(f"Missing timestamps: {len(missing_timestamps)}")
                print(f"Extra timestamps: {len(extra_timestamps)}")
                
                if not missing_timestamps.empty:
                    print(f"First 10 missing timestamps:")
                    for ts in missing_timestamps[:10]:
                        print(f"  {ts}")
                
                if not extra_timestamps.empty:
                    print(f"First 10 extra timestamps:")
                    for ts in extra_timestamps[:10]:
                        print(f"  {ts}")
            
            if not missing_timestamps.empty:
                print(f"Filling {len(missing_timestamps)} missing timestamps with NaN...")
                dwd_df = dwd_df.reindex(expected_index)
            
            if not extra_timestamps.empty:
                print(f"Removing {len(extra_timestamps)} extra timestamps...")
                dwd_df = dwd_df[dwd_df.index.isin(expected_index)]
            
            return dwd_df

        except Exception as e:
            print(f"Error in load_and_parse_dwd: {e}")
            return None

    def load_and_parse_pv(self, pv_data_path: str) -> Optional[pd.DataFrame]:
        try:          
            if not os.path.exists(pv_data_path):
                print(f"PV data file not found: {pv_data_path}")
                return None
            
            pv_df = pd.read_csv(pv_data_path)
            
            if '_time' in pv_df.columns:
                timestamp_col = '_time'
            else:
                print("No timestamp column found in PV data")
                return None
            
            pv_df[timestamp_col] = pd.to_datetime(pv_df[timestamp_col])
            
            # Remove duplicates
            initial_count = len(pv_df)
            pv_df = pv_df.drop_duplicates(subset=[timestamp_col])
            duplicates_removed = initial_count - len(pv_df)
            
            
            print(f"PV: Loaded {len(pv_df)} records")
            print(f"Time range: {pv_df[timestamp_col].min()} to {pv_df[timestamp_col].max()}")
            print(f"Removed {duplicates_removed} duplicate timestamps")

            pv_df_temp = pv_df.set_index(timestamp_col)
            is_pv_utc = self.check_timezone_utc(pv_df_temp, "PV")
            pv_df = pv_df_temp.reset_index()
            
            if not is_pv_utc:
                print("PV data is not UTC. Stopping.")
                return None
            
            return pv_df
            
        except Exception as e:
            print(f"Failed to load PV data: {e}")
            return None
    
    def align_pv_to_backbone(self, pv_df: pd.DataFrame, dwd_df: pd.DataFrame) -> pd.DataFrame:
        if '_time' in pv_df.columns:
            timestamp_col = '_time'
        else:
            timestamp_col = 'timestamp'
        
        pv_df_clean = pv_df.copy()
        pv_df_clean[timestamp_col] = pd.to_datetime(pv_df_clean[timestamp_col])        
        pv_sorted = pv_df_clean.sort_values(timestamp_col)
        dwd_reset = dwd_df.reset_index()
        dwd_sorted = dwd_reset.sort_values('timestamp')
        
        print(f"DWD NaN timestamps: {dwd_sorted['timestamp'].isna().sum()}")
        print(f"PV NaN timestamps: {pv_sorted[timestamp_col].isna().sum()}")
        
        # Remove NaN timestamps if any exist
        if dwd_sorted['timestamp'].isna().any():
            print(f"Found {dwd_sorted['timestamp'].isna().sum()} NaN timestamps in DWD data")
            dwd_sorted = dwd_sorted.dropna(subset=['timestamp'])
        
        if pv_sorted[timestamp_col].isna().any():
            print(f"Found {pv_sorted[timestamp_col].isna().sum()} NaN timestamps in PV data")
            pv_sorted = pv_sorted.dropna(subset=[timestamp_col])
        
        # Check if DataFrames are empty after removing NaN
        if len(dwd_sorted) == 0:
            print("DWD DataFrame is empty after removing NaN timestamps")
            return pd.DataFrame()
        
        if len(pv_sorted) == 0:
            print("PV DataFrame is empty after removing NaN timestamps")
            result = dwd_sorted.set_index('timestamp')
            pv_columns = [col for col in pv_df_clean.columns if col != timestamp_col]
            for col in pv_columns:
                result[col] = np.nan
            return result
        
        # Perform left join merge_asof
        try:
            merged = pd.merge_asof(
                dwd_sorted,  # Left: DWD backbone (all rows kept)
                pv_sorted,   # Right: PV data
                left_on='timestamp',
                right_on=timestamp_col,
                direction='nearest',
                tolerance=pd.Timedelta('2min')
            )
            
            if timestamp_col in merged.columns and timestamp_col != 'timestamp':
                merged = merged.drop(columns=[timestamp_col])
            
            merged.set_index('timestamp', inplace=True)
            
            print(f"Merged records: {len(merged)} (all DWD timestamps preserved)")
            print(f"Time range: {merged.index.min()} to {merged.index.max()}")
            
            return merged
            
        except Exception as e:
            print(f"ERROR in merge_asof: {e}")
            print(f"DWD shape: {dwd_sorted.shape}, PV shape: {pv_sorted.shape}")
            print(f"DWD timestamp range: {dwd_sorted['timestamp'].min()} to {dwd_sorted['timestamp'].max()}")
            print(f"PV timestamp range: {pv_sorted[timestamp_col].min()} to {pv_sorted[timestamp_col].max()}")
            
            # Fallback: Return DWD data with NaN PV columns
            result = dwd_sorted.set_index('timestamp')
            pv_columns = [col for col in pv_df_clean.columns if col != timestamp_col]
            for col in pv_columns:
                result[col] = np.nan
            return result
        
    def fill_missing_irradiance(self, result_df: pd.DataFrame) -> pd.DataFrame:
        if result_df.empty:
            print("No data to process")
            return result_df
        
        if self.scaling_factors is None:
            print("No scaling factors loaded")
            return result_df
        
        # Check for irradiance column (might be 'Irr' or 'global_radiation') in PV
        irradiance_col = None
        for col in ['Irr']:
            if col in result_df.columns:
                irradiance_col = col
                break
        
        if irradiance_col is None:
            print("No irradiance column found")
            return result_df
        
        # Check for DWD irradiance column
        dwd_irradiance_col = None
        for col in ['global_radiation']:
            if col in result_df.columns:
                dwd_irradiance_col = col
                break
        
        if dwd_irradiance_col is None:
            print(f"No DWD irradiance column found for scaling")
            return result_df
        
        # Count missing irradiance before filling
        missing_before = result_df[irradiance_col].isna().sum()
        
        if missing_before == 0:
            print(f"No missing irradiance values to fill")
            return result_df
        
        print(f"Filling {missing_before} missing {irradiance_col} values")
        
        # Fill missing irradiance values
        filled_count = 0
        for idx, row in result_df.iterrows():
            if pd.isna(row[irradiance_col]) and not pd.isna(row[dwd_irradiance_col]):
                scaled_value = self.apply_doy_period_scaling(idx, row[dwd_irradiance_col])
                result_df.at[idx, irradiance_col] = scaled_value
                filled_count += 1
        
        # Count after filling
        missing_after = result_df[irradiance_col].isna().sum()
        
        print(f"Filled {filled_count} irradiance values using DWD scaling")
        print(f"{missing_after} irradiance values still missing")
        
        return result_df
    
    def mark_maintenance_periods(self, df: pd.DataFrame, exclude_ranges):
        """
        Sets PV columns (P, I, U, Temp) to NaN for maintenance periods.
        """
        result_df = df.copy()

        if not isinstance(result_df.index, pd.DatetimeIndex):
            print("DataFrame must have datetime index")
            return result_df

        # Initialize bad_day
        if "bad_day" not in result_df.columns:
            result_df["bad_day"] = 0

        pv_cols = ["P", "I", "U", "Temp"]
        available_pv_cols = [c for c in pv_cols if c in result_df.columns]

        print(f"PV columns affected: {available_pv_cols}")

        # Normalize ranges to UTC timestamps
        ranges = [
            (pd.to_datetime(start).tz_localize("UTC"),
            pd.to_datetime(end).tz_localize("UTC"))
            for start, end in exclude_ranges
        ]

        for start, end in ranges:
            mask = (result_df.index >= start) & (result_df.index <= end)

            if mask.any():
                result_df.loc[mask, "bad_day"] = 1
                result_df.loc[mask, available_pv_cols] = np.nan

                print(f"Maintenance {start} to {end}")
                print(f"{mask.sum()} rows flagged, PV set to NaN")

        return result_df
    
    def mark_low_quality_days_by_p_count(self, df: pd.DataFrame, threshold: int = 70):
        """
        Sets PV columns (P, I, U, Temp) to NaN for days with P count less than threshold.
        """
        result_df = df.copy()
        if "bad_day" not in result_df.columns:
            print("bad_day column missing")
            return result_df

        pv_cols = ["P", "I", "U", "Temp"]
        available_pv_cols = [c for c in pv_cols if c in result_df.columns]
        good_df = result_df[result_df["bad_day"] == 0]

        if good_df.empty:
            print("No good days available for quality check.")
            return result_df

        valid_p_per_day = (
            good_df.groupby(good_df.index.date)["P"]
            .apply(lambda x: x.notna().sum())
        )

        if valid_p_per_day.empty:
            print("No valid P data found.")
            return result_df

        min_valid = valid_p_per_day.min()
        max_valid = valid_p_per_day.max()
        avg_valid = valid_p_per_day.mean()

        print(
            f"Good days valid 'P' count per day:"
            f"min={min_valid}, max={max_valid}, avg={avg_valid:.2f}"
        )

        low_valid_days = valid_p_per_day[valid_p_per_day < threshold].index
        print(f"Days with fewer than {threshold} valid 'P' rows: {len(low_valid_days)}")

        for day in low_valid_days:
            mask = result_df.index.date == day
            result_df.loc[mask, "bad_day"] = 1
            result_df.loc[mask, available_pv_cols] = np.nan

        total_bad_rows = result_df["bad_day"].sum()
        print(f"Total bad rows after quality filtering: {total_bad_rows}")

        return result_df
    
    def interpolate_pv_daylight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate PV values when:
        - bad_day == 0
        - daytime (04:00-22:00)
        - within the same day
        - gaps <= 1 hour
        - after interpolation: out of range values => 0
        - smooth ramp down to zero for P, I, Temp at end of day
        - if Irr (irradiance) <= 10, set P, I, U to zero
        """

        if df.empty:
            return df

        BOUNDS = {
            "P": (0, 110),
            "I": (0, 1.4),
            "U": (40, 100),
            "Temp": (-20, 100),
        }

        pv_cols = [c for c in BOUNDS if c in df.columns]
        if not pv_cols:
            return df

        result = df.copy().sort_index()
        is_day = (result.index.hour >= 4) & (result.index.hour < 22)
        good_day = result["bad_day"] == 0

        for col in pv_cols:
            day_mask = is_day & good_day
            daylight_series = result.loc[day_mask, col]

            if daylight_series.isna().sum() == 0:
                continue

            interpolated_days = []

            for _, day_group in daylight_series.groupby(daylight_series.index.date):
                s = day_group.copy()

                valid_ts = s.dropna().index
                for i in range(1, len(valid_ts)):
                    if (valid_ts[i] - valid_ts[i - 1]).total_seconds() > 3600:
                        gap_idx = s.loc[valid_ts[i - 1]:valid_ts[i]].index
                        if len(gap_idx) > 2:
                            s.loc[gap_idx[1:-1]] = np.nan

                s = s.interpolate(method="linear", limit_direction="both", limit_area="inside")
                if col in ["P", "I", "Temp"]:
                    last_valid_idx = s.last_valid_index()
                    if last_valid_idx is not None:
                        tail_idx = s.loc[last_valid_idx:].index
                        if len(tail_idx) > 1:
                            s.loc[tail_idx] = np.linspace(
                                s.loc[last_valid_idx],
                                0,
                                len(tail_idx)
                            )

                lo, hi = BOUNDS[col]
                s[(s < lo) | (s > hi)] = 0

                interpolated_days.append(s)

            if interpolated_days:
                merged = pd.concat(interpolated_days)
                result.loc[merged.index, col] = merged

        if "Irr" in result.columns:
            low_irr_mask = result["Irr"] <= 10
            for col in ["P", "I", "U"]:
                if col in result.columns:
                    result.loc[low_irr_mask, col] = 0

        return result
    
    def finalize_nan_cleanup(self, df: pd.DataFrame, set_nan_to_zero: bool = False) -> pd.DataFrame:
        """
        Prints NaN and non-NaN counts per column and optionally fills remaining NaNs.
        """
        nan_counts = df.isna().sum()
        non_nan_counts = df.notna().sum()
        total_rows = len(df)
        total_nans = nan_counts.sum()
        
        print(f"Total rows: {total_rows}")
        print(f"Total NaN values: {total_nans}")
        
        for col in df.columns:
            nan_count = nan_counts[col]
            non_nan_count = non_nan_counts[col]
            
            nan_pct = (nan_count / total_rows * 100) if total_rows > 0 else 0
            
            if nan_count > 0:
                print(f"{col}: {nan_count} NaN ({nan_pct:.1f}%) | {non_nan_count} non-NaN")
            else:
                print(f"{col}: {nan_count} NaN | {non_nan_count} non-NaN")
        
        if total_nans == 0:
            print("\nNo NaNs found in any column.")
            return df
        
        if set_nan_to_zero:
            print(f"\nSetting {total_nans} NaN values to 0.")
            df_filled = df.fillna(0)
            return df_filled
        else:
            print(f"\nKeeping {total_nans} NaN values (not replacing with 0).")
            return df

    def drop_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop all columns NOT listed in local keep_columns.
        """
        keep_columns = ["P", "I", "U", "Temp", "Irr", "bad_day", "humidity", "temp_C", "precip_mm", "precip_indicator", "cloud_cover"]

        existing_keep = [c for c in keep_columns if c in df.columns]
        to_drop = [c for c in df.columns if c not in existing_keep]

        print(f"\nKeeping: {existing_keep}")
        print(f"Dropping: {to_drop}")

        return df[existing_keep].copy()
    
    def plot_monthly_timeseries(
        self,
        df: pd.DataFrame,
        output_dir: str,
        pv_cols=('P', 'I', 'U', 'Temp', 'Irr', 'bad_day', 'humidity', 'temp_C', 'precip_mm', 'precip_indicator', 'cloud_cover'),
    ):
        if df.empty:
            print("Empty DataFrame.")
            return

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        if "bad_day" not in df.columns:
            raise ValueError("DataFrame must contain 'bad_day' column")

        os.makedirs(output_dir, exist_ok=True)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["month"] = df["timestamp"].dt.to_period("M")

        cols = [c for c in pv_cols if c in df.columns]
        if not cols:
            print("No PV columns found for plotting")
            return

        print(f"Plotting columns: {cols}")

        for month, month_df in df.groupby("month"):
            fig, axes = plt.subplots(
                len(cols), 1, figsize=(14, 3 * len(cols)), sharex=True
            )

            if len(cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, cols):

                # Line + small dots
                ax.plot(
                    month_df["timestamp"],
                    month_df[col],
                    linewidth=1,
                    marker="o",
                    markersize=2,
                    alpha=0.8
                )

                # Highlight bad days
                bad_days = month_df.loc[month_df["bad_day"] == 1, "timestamp"].dt.date.unique()
                for d in bad_days:
                    start = pd.Timestamp(d)
                    end = start + pd.Timedelta(days=1)
                    ax.axvspan(start, end, color="red", alpha=0.15)

                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)

            # X-axis formatting: daily ticks, 90° rotation
            axes[-1].xaxis.set_major_locator(mdates.DayLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
            axes[-1].tick_params(axis="x", rotation=90, labelsize=8)

            fig.suptitle(f"Monthly Time Series for {month}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = os.path.join(output_dir, f"timeseries_{month}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved {out_path}")

    def plot_correlation_scatter(self, df: pd.DataFrame, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()]

        if "bad_day" in df.columns:
            before = len(df)
            df = df[df["bad_day"] == 0]
            print(f"Excluded bad days: {before - len(df)} rows")
        else:
            print("Warning: 'bad_day' column not found")

        is_day = (df["timestamp"].dt.hour >= 4) & (df["timestamp"].dt.hour < 22)
        day_df = df[is_day]

        print(f"Excluded night-time rows: {(~is_day).sum()}")
        print(f"Remaining daytime rows: {len(day_df)}")

        if day_df.empty:
            print("No daytime data left, skipping correlation plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "PV System Correlations (Bad Days & Night Excluded)",
            fontsize=16,
            fontweight="bold",
        )
        axes = axes.flatten()

        def scatter_plot(ax, x, y, c, xlabel, ylabel, title):
            valid = day_df[[x, y, c]].dropna()
            if valid.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                return

            x_min, x_max = valid[x].min(), valid[x].max()
            y_min, y_max = valid[y].min(), valid[y].max()

            sc = ax.scatter(
                valid[x],
                valid[y],
                c=valid[c],
                cmap="viridis",
                s=12,
                alpha=0.6,
            )

            if "Irr" in valid.columns:
                low_irr = valid[valid["Irr"] < 10]
                if not low_irr.empty:
                    ax.scatter(
                        low_irr[x],
                        low_irr[y],
                        color="red",
                        s=10,
                        alpha=0.9,
                        label="Irr < 10"
                    )
                    ax.legend(loc="best")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            corr = valid[x].corr(valid[y])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}\n(r = {corr:.3f})")
            ax.grid(True, alpha=0.3)

            plt.colorbar(sc, ax=ax, label=c)

        plots = [
            (["Irr", "P", "Temp"], "Irr", "P", "Temp",
            "Irradiance", "Power",
            "Power vs Irradiance (colored by Temperature)"),

            (["U", "Temp", "Irr"], "U", "Temp", "Irr",
            "Voltage", "Panel Temperature",
            "Voltage vs Temperature (colored by Irradiance)"),

            (["Irr", "I", "Temp"], "Irr", "I", "Temp",
            "Irradiance", "Current",
            "Current vs Irradiance (colored by Temperature)"),

            (["Irr", "Temp", "P"], "Irr", "Temp", "P",
            "Irradiance", "Panel Temperature",
            "Irradiance vs Temperature (colored by Power)"),
        ]

        for ax, (cols, x, y, c, xl, yl, title) in zip(axes, plots):
            if all(col in day_df.columns for col in cols):
                scatter_plot(ax, x, y, c, xl, yl, title)
            else:
                ax.set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(output_dir, "correlation_scatter.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

        self._plot_correlation_heatmap(day_df, output_dir)

    def _plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: str):
        cols = ["P", "I", "U", "Temp", "Irr", "humidity", "temp_C", "precip_mm", "precip_indicator", "cloud_cover",]
        numeric_cols = [c for c in cols if c in df.columns]
        df = df[numeric_cols].dropna()

        if len(df) < 10:
            print("Not enough valid rows for correlation heatmap")
            return

        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            "Correlation Matrix (Bad Days & Night Excluded)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        out_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    
    def plot_day_per_month(
        self,
        df: pd.DataFrame,
        output_dir: str,
        day_list: list[str] | None = None,
        p_col: str = "P",
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

        if df.empty:
            print("Empty DataFrame.")
            return

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        if "bad_day" not in df.columns:
            raise ValueError("DataFrame must contain 'bad_day' column")

        if not {p_col, irr_col}.issubset(df.columns):
            print(f"Required columns missing: {p_col}, {irr_col}")
            return

        os.makedirs(output_dir, exist_ok=True)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
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

            day_df = day_df.dropna(subset=[p_col, irr_col])
            if day_df.empty:
                print(f"[SKIP] {day} has no valid data after NaN drop")
                continue

            times = day_df.index
            month = times[0].to_period("M")

            fig, ax1 = plt.subplots(figsize=(10, 4))

            ax1.plot(times, day_df[p_col], label=p_col, linewidth=2, marker="o", markersize=3)
            ax1.set_xlabel("Time (hour)")
            ax1.set_ylabel(p_col)
            ax1.set_title(f"{month} | {day}")
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

            ax2 = ax1.twinx()
            ax2.plot(times, day_df[irr_col], label=irr_col, linestyle="--", alpha=0.6)
            ax2.set_ylabel(irr_col)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=2, fontsize=8)

            plt.tight_layout()
            fname = os.path.join(output_dir, f"day_{day}.png")
            plt.savefig(fname, dpi=150)
            plt.close()

            print(f"Saved {fname}")

    def process_data(self, pv_data_path: str, output_path: str, create_plots: bool = True) -> bool:       
        # Load scaling factors
        if not self.load_scaling_factors():
            return False
        
        # Load PV data
        print(f"\nSTEP 1: Loading PV data")
        print("-" * 100)
        pv_df = self.load_and_parse_pv(pv_data_path)
        if pv_df is None:
            return False
        
        # Get PV time range
        pv_start = pv_df['_time'].min()
        pv_end = pv_df['_time'].max()
        
        # Load DWD data
        print(f"\nSTEP 2: Loading DWD data")
        print("-" * 100)
        dwd_df = self.load_and_parse_dwd()
        if dwd_df is None:
            return False
        
        # Filter DWD to PV time range
        print(f"\nSTEP 3: Filtering DWD to PV time range")
        print("-" * 100)
        dwd_filtered = dwd_df[
            (dwd_df.index >= pv_start) & 
            (dwd_df.index <= pv_end)
        ].copy()
        
        print(f"Original DWD records: {len(dwd_df)}")
        print(f"Filtered DWD records: {len(dwd_filtered)}")
        print(f"PV time range: {pv_start} to {pv_end}")
        
        if len(dwd_filtered) == 0:
            print("No DWD data within PV time range")
            return False
        
        # Merge PV into DWD backbone
        print(f"\nSTEP 4: Merging PV into DWD backbone")
        print("-" * 100)
        merged_df = self.align_pv_to_backbone(pv_df, dwd_filtered)
        if merged_df.empty:
            print("No PV data could be merged with DWD backbone")
            return False
        
        # Fill missing irradiance
        print(f"\nSTEP 6: Filling missing irradiance")
        print("-" * 100)
        irr_filled_df = self.fill_missing_irradiance(merged_df)

        # Mask maintenance periods as bad days
        print(f"\nSTEP 7: Masking maintenance periods as bad days")
        print("-" * 100)
        exclude_ranges = [
            ("2024-12-06", "2024-12-14"),
            ("2024-12-30", "2024-12-31"),
            ("2025-01-02", "2024-01-03"),
            ("2025-05-02", "2025-05-13"),
            ("2025-06-26", "2025-06-30"),
            ("2025-09-17", "2025-09-26"),
            ("2025-11-18", "2025-11-23"),
        ]   
        meaintenance_period_df = self.mark_maintenance_periods(irr_filled_df, exclude_ranges)
        
        # Mask days with P count less than threshold, already night time is set to zero so the threshold is already 36. 
        print(f"\nSTEP 8: Masking days with low P count")
        print("-" * 100)
        high_quality_df = self.mark_low_quality_days_by_p_count(meaintenance_period_df, threshold=70)

        # Interpolate PV data
        print(f"\nSTEP 9: Interpolating PV data")
        print("-" * 100)
        interpolated_df = self.interpolate_pv_daylight(high_quality_df)

        # Remove unwanted columns
        result_df = self.drop_unwanted_columns(interpolated_df)

        # Set all nans to zero or as it is
        print(f"\nSTEP 10: Nan handling")
        print("-" * 100)
        result_df = self.finalize_nan_cleanup(result_df, set_nan_to_zero=False) 
        
        if create_plots:
                plot_df = result_df.reset_index()
                plot_dir = os.path.join(os.path.dirname(output_path), 'plots')
                print(f"\nSTEP 11: Creating visualizations")
                print("-" * 100)
                self.plot_monthly_timeseries(plot_df, plot_dir)
                self.plot_correlation_scatter(plot_df, plot_dir)
                self.plot_day_per_month(plot_df, plot_dir)
                print(f"\nPlots saved to: {plot_dir}")

        try:
            final_df_reset = result_df.reset_index()
            if "timestamp" in final_df_reset.columns:
                final_df_reset = final_df_reset.rename(columns={"timestamp": "_time"})
            final_df_reset = final_df_reset.sort_values("_time")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV
            final_df_reset.to_csv(output_path, index=False)
            
            print(f"\nData saved to: {output_path}")
            print(f"Total records: {len(final_df_reset)}")
            print(f"Time range: {final_df_reset['_time'].min()} to {final_df_reset['_time'].max()}")
            print(f"Columns: {list(final_df_reset.columns)}")
            return True
            
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False