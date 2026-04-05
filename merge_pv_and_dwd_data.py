#Version: 2.0
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, List, Dict

class PVDWDDataMerger:
    def __init__(self, dwd_data_path: str, scaling_factors_path: str, dwd_columns: Optional[List[str]] = None):
        """
        Class to merge PV system data with DWD weather data using DWD time as backbone.
        Supports both category-averaged and per-module data structures.
        """
        self.dwd_data_path = dwd_data_path
        self.scaling_factors_path = scaling_factors_path
        self.dwd_columns = dwd_columns
        self.dwd_df = None
        self.scaling_factors = None
        self.data_structure = None  # 'category' or 'pv_per_module' - determined from filename
        self.column_mapping = None  # Dynamic column mapping based on structure
        self.pv_categories = []  # Will be detected from data
        self.modules = []  # List of module identifiers for pv_per_module structure
        self.base_pv_columns = ["P", "P_normalised", "I", "U", "Temp"]
        
    def detect_data_structure_from_filename(self, pv_data_path: str) -> str:
        """Determine data structure based on filename."""
        filename = os.path.basename(pv_data_path).lower()
        
        if "pv_averaged_by_category" in filename:
            print(f"Filename '{filename}' indicates 'category' mode")
            return 'category'
        elif "pv_per_module" in filename:
            print(f"Filename '{filename}' indicates 'per_module' mode")
            return 'per_module'
        else:
            print(f"WARNING : Cannot detect data structure from filename, defaulting to 'category'")
            return 'category'
    
    def build_column_mapping(self, df):
        """Build column mapping based on detected data structure."""
        
        if self.data_structure == 'category':
            self.pv_categories = ['si', 'psc']
            
            # mapping for each category
            mapping = {}
            known_params = ['P', 'P_normalised', 'I', 'U', 'Temp']
            
            for category in self.pv_categories:
                mapping[category] = {}
                for param in known_params:
                    col_name = f"{param}_{category}"
                    if col_name in df.columns:
                        mapping[category][param] = col_name
            
            found_cols = []
            for category in self.pv_categories:
                for param in known_params:
                    if param in mapping[category]:
                        found_cols.append(f"{param}_{category}")
            
            print(f"Found PV columns: {found_cols}")
            return mapping
            
        else:  # per_module
            #extract module information
            mapping = {}
            module_types = set()
            module_list = []
            
            for col in df.columns:
                if '_' in col:
                    parts = col.split('_')
                    if len(parts) >= 3:
                        param = parts[0]  #P, I, U, Temp
                        module_type = parts[1]  #atersa, sanyo,...
                        module_id = parts[2]  #1, 2, 3,..
                        
                        module_key = f"{module_type}_{module_id}"
                        module_types.add(module_type)
                        
                        if module_key not in mapping:
                            mapping[module_key] = {
                                'type': module_type,
                                'id': module_id,
                                'category': self._get_category_for_module_type(module_type)
                            }
                            module_list.append(module_key)
                        
                        mapping[module_key][param] = col
            
            self.modules = sorted(module_list)
            
            #extract categories from module types
            self.pv_categories = sorted(set(mapping[m]['category'] for m in mapping))
            print(f"Detected modules: {self.modules}")
            print(f"Module types: {sorted(module_types)}")
            print(f"Derived categories: {self.pv_categories}")
            
            return mapping
    
    def _get_category_for_module_type(self, module_type):
        """Map module types to categories."""
        si_types = ['atersa', 'sanyo', 'solon', 'sun_power']
        psc_types = ['perovskite']
        
        if module_type in si_types:
            return 'si'
        elif module_type in psc_types:
            return 'psc'
        else:
            print(f"Warning: Unknown module type '{module_type}', defaulting to 'si'")
            return 'si'  #default to si
    
    def _get_category_bounds(self):
        """Get bounds for category-averaged data."""
        return {
            # si category bounds
            "P_si": (0, 250),
            "I_si": (0, 9),
            "U_si": (10, 60),
            "P_normalised_si": (0.0, 1.2),
            "Temp_si": (-20, 100),
            
            # psc category bounds
            "P_psc": (0, 110),
            "I_psc": (0, 1.4),
            "U_psc": (40, 100),
            "P_normalised_psc": (0.0, 1.75),
            "Temp_psc": (-20, 100),
        }
    
    #Need to extend per module type
    def _get_per_module_bounds(self):
        """Get bounds for per-module data."""
        bounds = {}
        
        if self.column_mapping is None:
            return bounds
            
        for module_key, module_info in self.column_mapping.items():
            category = module_info.get('category', 'si')
            
            #dpply category-specific bounds
            if category == 'si':
                if 'P' in module_info:
                    bounds[module_info['P']] = (0, 250)
                if 'I' in module_info:
                    bounds[module_info['I']] = (0, 9)
                if 'U' in module_info:
                    bounds[module_info['U']] = (10, 60)
                if 'P_normalised' in module_info:
                    bounds[module_info['P_normalised']] = (0.0, 1.2)
                if 'Temp' in module_info:
                    bounds[module_info['Temp']] = (-20, 100)
                    
            elif category == 'psc':
                if 'P' in module_info:
                    bounds[module_info['P']] = (0, 110)
                if 'I' in module_info:
                    bounds[module_info['I']] = (0, 1.4)
                if 'U' in module_info:
                    bounds[module_info['U']] = (40, 100)
                if 'P_normalised' in module_info:
                    bounds[module_info['P_normalised']] = (0.0, 1.75)
                if 'Temp' in module_info:
                    bounds[module_info['Temp']] = (-20, 100)
        
        return bounds
    
    def get_pv_columns_for_category(self, category: str) -> Dict[str, str]:
        """Get column names for a specific PV category."""
        if self.data_structure == 'category':
            if category in self.column_mapping:
                return self.column_mapping[category]
            else:
                return {}
        else:
            #for per-module structure, collect all columns for modules of this category
            result = {}
            for module_key, module_info in self.column_mapping.items():
                if module_info.get('category') == category:
                    for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                        if param in module_info:
                            if param not in result:
                                result[param] = []
                            result[param].append(module_info[param])
            return result
    
    def get_pv_columns_for_module(self, module_key: str) -> Dict[str, str]:
        """Get column names for a specific module."""
        if self.data_structure != 'per_module':
            return {}
        
        if module_key in self.column_mapping:
            return {k: v for k, v in self.column_mapping[module_key].items() 
                   if k in ['P', 'I', 'U', 'Temp', 'P_normalised']}
        return {}
    
    def get_all_pv_columns(self):
        """Get all PV columns regardless of structure."""
        all_cols = []
        
        if self.data_structure == 'category':
            for category in self.pv_categories:
                category_cols = self.get_pv_columns_for_category(category)
                all_cols.extend(category_cols.values())
        else:
            for module_key, module_info in self.column_mapping.items():
                for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                    if param in module_info:
                        all_cols.append(module_info[param])
        
        return list(set(all_cols))
    
    def create_specific_bad_day_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create category-specific or module-specific bad_day columns based on data structure.
        For category structure: creates bad_day_si, bad_day_psc
        For module structure: creates bad_day_atersa_1, bad_day_sanyo_2,...
        """
        result_df = df.copy()
        bad_day_cols = [col for col in result_df.columns if col.startswith('bad_day')]
        for col in bad_day_cols:
            result_df = result_df.drop(columns=[col])
        
        if self.data_structure == 'category':
            print("\nCreating category-specific bad_day columns...")
            
            for category in self.pv_categories:
                col_name = f"bad_day_{category}"
                result_df[col_name] = 0
                print(f"  Created {col_name}")
            
        else:
            print("\nCreating module-specific bad_day columns...")
            
            for module_key in self.modules:
                col_name = f"bad_day_{module_key}"
                result_df[col_name] = 0
                print(f"  Created {col_name}")
        
        return result_df
    
    def check_timezone_utc(self, df: pd.DataFrame, df_name: str) -> bool:
        """Verify DataFrame has UTC timezone."""
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
                print(f"WARNING : Scaling factors file not found: {self.scaling_factors_path}")
                return False
            with open(self.scaling_factors_path, 'r') as f:
                self.scaling_factors = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"WARNING : Failed to load scaling factors: {e}")
            return False
    
    def apply_doy_period_scaling(self, timestamp: pd.Timestamp, dwd_irradiance: float) -> float:
        """Apply day-of-year and period-based scaling to DWD irradiance."""
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
            #return zero if no scaling factor
            return 0.0

        # Try specific day-period factor
        key = f"{doy:03d}-{period_name}"
        factor_entry = self.scaling_factors.get("dayofyear_period", {}).get(key)

        if factor_entry and "factor" in factor_entry:
            factor = float(factor_entry["factor"])
            if factor > 0:
                return dwd_irradiance * factor

        # Fallback to period average
        fallback_factor = self.scaling_factors.get("period_fallback", {}).get(period_name, 0.0)
        if fallback_factor > 0:
            return dwd_irradiance * fallback_factor

        #return zero if no fallback scaling factor
        return 0.0
    
    def load_and_parse_dwd(self) -> Optional[pd.DataFrame]:
        """Load and parse DWD weather data."""
        try:
            if not os.path.exists(self.dwd_data_path):
                print(f"WARNING : DWD data file not found: {self.dwd_data_path}")
                return None
            
            dwd_df = pd.read_csv(self.dwd_data_path)
            
            if 'timestamp' not in dwd_df.columns:
                print("WARNING : No timestamp column in DWD data")
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
            print(f"WARNING : Error loading DWD data: {e}")
            return None

    def load_and_parse_pv(self, pv_data_path: str) -> Optional[pd.DataFrame]:
        """Load and parse PV data, determining structure from filename."""
        try:
            if not os.path.exists(pv_data_path):
                print(f"WARNING : PV data file not found: {pv_data_path}")
                return None
            
            #etermine data structure from filename
            self.data_structure = self.detect_data_structure_from_filename(pv_data_path)
            print(f"Data structure determined: {self.data_structure}")
            
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

            # Build column mapping based on determined structure
            self.column_mapping = self.build_column_mapping(pv_df)
            
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
            print("WARNING : DWD DataFrame is empty after removing NaN timestamps")
            return pd.DataFrame()
        
        if len(pv_sorted) == 0:
            print("WARNING : PV DataFrame is empty after removing NaN timestamps")
            result = dwd_sorted.set_index('timestamp')
            pv_columns = [col for col in pv_df_clean.columns if col != timestamp_col]
            for col in pv_columns:
                result[col] = np.nan
            return result
        
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
            print(f"WARNING: ERROR in merge_asof: {e}")
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
        """Fill missing irradiance values using DWD scaling."""
        if result_df.empty:
            print("No data to process")
            return result_df
        
        if self.scaling_factors is None:
            print("No scaling factors loaded")
            return result_df
        
        irradiance_col = 'Irr'
        if irradiance_col not in result_df.columns:
            print(f"No '{irradiance_col}' column found")
            return result_df
        
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
        
        #fill missing irradiance values
        filled_count = 0
        for idx, row in result_df.iterrows():
            if pd.isna(row[irradiance_col]) and not pd.isna(row[dwd_irradiance_col]):
                scaled_value = self.apply_doy_period_scaling(idx, row[dwd_irradiance_col])
                result_df.at[idx, irradiance_col] = scaled_value
                filled_count += 1
        
        #count after filling
        missing_after = result_df[irradiance_col].isna().sum()
        print(f"Filled {filled_count} irradiance values using DWD scaling")
        print(f"{missing_after} irradiance values still missing")
        
        return result_df
    
    def mark_maintenance_periods(self, df: pd.DataFrame):
        """Mark maintenance periods as bad days."""
        result_df = df.copy()

        if not isinstance(result_df.index, pd.DatetimeIndex):
            print("WARNING : DataFrame must have datetime index")
            return result_df

        maintenance_periods = {
            'si': [
                ("2024-12-06", "2024-12-14"),
                ("2025-05-02", "2025-05-13"),
                ("2025-06-25", "2025-06-30"),
                ("2025-09-17", "2025-09-26"),
                ("2025-11-18", "2025-11-26"),
            ],
            'psc': [
                ("2024-12-06", "2024-12-14"),
                ("2024-12-30", "2024-12-31"),
                ("2025-01-02", "2025-01-03"),
                ("2025-05-02", "2025-05-13"),
                ("2025-06-25", "2025-06-30"),
                ("2025-09-17", "2025-09-26"),
                ("2025-11-18", "2025-11-26"),
            ]
        }

        for category in self.pv_categories:
            if category not in maintenance_periods:
                continue
                
            #get PV columns for by category
            if self.data_structure == 'category':
                category_cols = self.get_pv_columns_for_category(category)
                pv_cols_to_nan = [
                    category_cols.get('P', f'P_{category}'),
                    category_cols.get('I', f'I_{category}'),
                    category_cols.get('U', f'U_{category}'),
                    category_cols.get('Temp', f'Temp_{category}'),
                ]
                # Add P_normalised if it exists
                if 'P_normalised' in category_cols:
                    pv_cols_to_nan.append(category_cols['P_normalised'])
            else:  # per_module
                pv_cols_to_nan = []
                for module_key, module_info in self.column_mapping.items():
                    if module_info.get('category') == category:
                        for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                            if param in module_info:
                                pv_cols_to_nan.append(module_info[param])
            
            # Filter to columns that actually exist in the dataframe
            available_pv_cols = [c for c in pv_cols_to_nan if c in result_df.columns]
            
            if not available_pv_cols:
                continue
                
            print(f"\nCategory '{category}' - PV columns affected: {available_pv_cols[:5]}...")  # Show first 5

            #process maintenance periods for the category
            for start_str, end_str in maintenance_periods[category]:
                try:
                    start_dt = pd.to_datetime(start_str)
                    if start_dt.tz is None:
                        start_dt = start_dt.tz_localize("UTC")
                    
                    end_dt = pd.to_datetime(end_str)
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize("UTC")
                    
                    mask = (result_df.index >= start_dt) & (result_df.index <= end_dt)
                    
                    if mask.any():
                        if self.data_structure == 'category':
                            category_bad_col = f"bad_day_{category}"
                            if category_bad_col in result_df.columns:
                                result_df.loc[mask, category_bad_col] = 1
                        
                        else:  # per_module
                            for module_key, module_info in self.column_mapping.items():
                                if module_info.get('category') == category:
                                    module_bad_col = f"bad_day_{module_key}"
                                    if module_bad_col in result_df.columns:
                                        result_df.loc[mask, module_bad_col] = 1
                        
                        #Set PV values to NaN
                        result_df.loc[mask, available_pv_cols] = np.nan
                        
                        print(f"Maintenance {start_str} to {end_str}")
                        print(f"{mask.sum()} rows flagged, PV set to NaN")
                        
                except Exception as e:
                    print(f"Error parsing date range {start_str} to {end_str}: {e}")

        return result_df
    
    def mark_before_installation_bad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark data before installation as bad."""
        result_df = df.copy()

        if not isinstance(result_df.index, pd.DatetimeIndex):
            return result_df

        installation_dates = {
            'si': pd.Timestamp("2024-07-26").tz_localize("UTC"),
            'psc': pd.Timestamp("2024-11-28").tz_localize("UTC"),
        }

        total_rows = len(result_df)

        for category in self.pv_categories:
            if category not in installation_dates:
                continue
                
            install_date = installation_dates[category]
            
            # Get columns for this category
            if self.data_structure == 'category':
                category_cols = self.get_pv_columns_for_category(category)
                pv_cols_to_nan = [
                    category_cols.get('P', f'P_{category}'),
                    category_cols.get('I', f'I_{category}'),
                    category_cols.get('U', f'U_{category}'),
                    category_cols.get('Temp', f'Temp_{category}'),
                ]
                #add P_normalised if it exists
                if 'P_normalised' in category_cols:
                    pv_cols_to_nan.append(category_cols['P_normalised'])
            else:  #per_module
                pv_cols_to_nan = []
                for module_key, module_info in self.column_mapping.items():
                    if module_info.get('category') == category:
                        for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                            if param in module_info:
                                pv_cols_to_nan.append(module_info[param])
            
            #filter to columns that actually exist in the dataframe
            available_pv_cols = [c for c in pv_cols_to_nan if c in result_df.columns]
            
            if not available_pv_cols:
                continue
            
            #mark data before installation as bad
            mask = result_df.index < install_date
            
            if mask.any():
                rows_affected = mask.sum()                
                if self.data_structure == 'category':
                    category_bad_col = f"bad_day_{category}"
                    if category_bad_col in result_df.columns:
                        result_df.loc[mask, category_bad_col] = 1                
                else:  # per_module
                    for module_key, module_info in self.column_mapping.items():
                        if module_info.get('category') == category:
                            module_bad_col = f"bad_day_{module_key}"
                            if module_bad_col in result_df.columns:
                                result_df.loc[mask, module_bad_col] = 1
                
                #set PV values to NaN
                result_df.loc[mask, available_pv_cols] = np.nan
                
                print(f"\nCategory '{category}' (installed on {install_date.date()}):")
                print(f"{rows_affected} rows before installation")
                print(f"Marked specific bad_day columns = 1")

        print(f"\nPer-category summary:")
        for category in self.pv_categories:
            if category in installation_dates:
                install_date = installation_dates[category]
                rows_before_install = (result_df.index < install_date).sum()
                print(f"{category}: installed {install_date.date()}, {rows_before_install} rows before installation")

        return result_df
    
    def mark_low_quality_days_by_p_count(self, df: pd.DataFrame, threshold: int = 70):
        """Mark days with low P count as bad."""
        result_df = df.copy()

        # Get P columns
        if self.data_structure == 'category':
            p_columns = [f'P_{cat}' for cat in self.pv_categories if f'P_{cat}' in result_df.columns]
        else:  # per_module
            p_columns = []
            module_p_mapping = {}
            for module_key, module_info in self.column_mapping.items():
                if 'P' in module_info and module_info['P'] in result_df.columns:
                    p_col = module_info['P']
                    p_columns.append(p_col)
                    module_p_mapping[p_col] = module_key

        if not p_columns:
            print("No P columns found for any category/module")
            return result_df

        for p_col in p_columns:
            if p_col not in result_df.columns:
                continue
            valid_p_per_day = (
                result_df.groupby(result_df.index.date)[p_col]
                .apply(lambda x: x.notna().sum())
            )
            if valid_p_per_day.empty:
                print(f"No valid data found for {p_col}.")
                continue

            # Get scalar values
            min_valid = float(valid_p_per_day.min())
            max_valid = float(valid_p_per_day.max())
            avg_valid = float(valid_p_per_day.mean())

            print(f"{p_col} (valid count per day): min={min_valid:.0f}, max={max_valid:.0f}, avg={avg_valid:.2f}")

            low_valid_days = valid_p_per_day[valid_p_per_day < threshold].index
            low_days_count = len(low_valid_days)
            print(f"Days with fewer than {threshold} valid '{p_col}' rows: {low_days_count}")

            if low_days_count == 0:
                continue

            if self.data_structure == 'category':
                for category in self.pv_categories:
                    if p_col == f'P_{category}':
                        category_cols = self.get_pv_columns_for_category(category)
                        pv_cols_to_nan = []
                        for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                            if param in category_cols:
                                pv_cols_to_nan.append(category_cols[param])
                        
                        available_cols = [c for c in pv_cols_to_nan if c in result_df.columns]
                        
                        for day in low_valid_days:
                            mask = result_df.index.date == day
                            
                            #update category-specific bad_day
                            category_bad_col = f"bad_day_{category}"
                            if category_bad_col in result_df.columns:
                                result_df.loc[mask, category_bad_col] = 1
                            
                            #set PV values to NaN
                            if available_cols:
                                result_df.loc[mask, available_cols] = np.nan
                        
                        print(f"Category '{category}': marked {low_days_count} days as bad")
                        break
            
            else:  # per_module
                module_key = module_p_mapping.get(p_col)
                if not module_key:
                    continue                    
                module_cols = self.get_pv_columns_for_module(module_key)
                available_cols = list(module_cols.values())                
                module_info = self.column_mapping[module_key]
                category = module_info.get('category', 'unknown')
                
                for day in low_valid_days:
                    mask = result_df.index.date == day
                    module_bad_col = f"bad_day_{module_key}"
                    if module_bad_col in result_df.columns:
                        result_df.loc[mask, module_bad_col] = 1
                    if available_cols:
                        result_df.loc[mask, available_cols] = np.nan
                
                print(f"Module '{module_key}' (category: {category}): marked {low_days_count} days as bad")

        if self.data_structure == 'category':
            print("\nCategory-specific bad rows after low P count filtering:")
            for category in self.pv_categories:
                category_bad_col = f"bad_day_{category}"
                if category_bad_col in result_df.columns:
                    category_bad_rows = result_df[category_bad_col].sum()
                    print(f"  {category_bad_col}: {category_bad_rows} bad rows")
        else:  # per_module
            print("\nModule-specific bad rows after low P count filtering (first 5 modules):")
            module_count = 0
            for module_key in self.modules:
                module_bad_col = f"bad_day_{module_key}"
                if module_bad_col in result_df.columns:
                    module_bad_rows = result_df[module_bad_col].sum()
                    print(f"  {module_bad_col}: {module_bad_rows} bad rows")
                    module_count += 1
                    if module_count >= 5:
                        break

        return result_df
    
    def enforce_bounds_and_irradiance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce bounds and irradiance rules on good days."""
        if df.empty:
            return df

        result = df.copy()

        if self.data_structure == 'category':
            BOUNDS = self._get_category_bounds()
            # Category based processing
            for category in self.pv_categories:
                category_bad_col = f"bad_day_{category}"  
                if category_bad_col not in result.columns:
                    print(f"Warning: No bad_day column for category '{category}', treating all days as good")
                    good_day_mask = pd.Series(True, index=result.index)
                else:
                    good_day_mask = result[category_bad_col] == 0  
                if not good_day_mask.any():
                    print(f"No good days found for category '{category}', skipping bounds enforcement")
                    continue                
                category_cols = self.get_pv_columns_for_category(category)

                # Apply bounds
                for param in ['P', 'I', 'U', 'Temp', 'P_normalised']:
                    if param in category_cols:
                        col_name = category_cols[param]
                        if col_name in result.columns and col_name in BOUNDS:
                            lo, hi = BOUNDS[col_name]

                            valid_val_mask = result[col_name].notna() & (result[col_name] != 0)
                            out_of_bounds_mask = (valid_val_mask & ((result[col_name] < lo) | (result[col_name] > hi)) & good_day_mask)
                            
                            if out_of_bounds_mask.any():
                                rows_to_zero = out_of_bounds_mask.sum()
                                result.loc[out_of_bounds_mask, col_name] = 0
                                print(f"Category '{category}': Set {rows_to_zero} out-of-bounds values to 0 for column '{col_name}'")
        
        else:  # per_module
            BOUNDS = self._get_per_module_bounds()            
            for module_key in self.modules:
                module_bad_col = f"bad_day_{module_key}"
                if module_bad_col not in result.columns:
                    print(f"Warning: No bad_day column for module '{module_key}', treating all days as good")
                    module_good_mask = pd.Series(True, index=result.index)
                else:
                    module_good_mask = result[module_bad_col] == 0
                
                if not module_good_mask.any():
                    print(f"No good days found for module '{module_key}', skipping bounds enforcement")
                    continue
                
                module_cols = self.get_pv_columns_for_module(module_key)
                
                # Apply bounds
                for param, col_name in module_cols.items():
                    if col_name in result.columns and col_name in BOUNDS:
                        lo, hi = BOUNDS[col_name]
                        
                        valid_val_mask = result[col_name].notna() & (result[col_name] != 0)
                        out_of_bounds_mask = (valid_val_mask & ((result[col_name] < lo) | (result[col_name] > hi)) & module_good_mask)
                        
                        if out_of_bounds_mask.any():
                            rows_to_zero = out_of_bounds_mask.sum()
                            result.loc[out_of_bounds_mask, col_name] = 0
                            print(f"Module '{module_key}': Set {rows_to_zero} out-of-bounds values to 0 for column '{col_name}'")

        return result
    
    def finalize_nan_cleanup(self, df: pd.DataFrame, set_nan_to_zero: bool = False) -> pd.DataFrame:
        """Print statistics and optionally fill NaNs."""
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
        """Keep only required columns."""
        keep_columns = ["Irr"]
        
        #keep all bad_day_* columns for training
        bad_day_cols = [col for col in df.columns if col.startswith('bad_day_')]
        keep_columns.extend(bad_day_cols)
        
        # Add weather columns
        weather_columns = ["humidity", "temp_C", "precip_mm", "precip_indicator", "cloud_cover"]
        keep_columns.extend([c for c in weather_columns if c in df.columns])
        
        # Add PV columns
        if self.data_structure == 'category':
            for category in self.pv_categories:
                category_cols = self.get_pv_columns_for_category(category)
                for param in ["P", "P_normalised", "I", "U", "Temp"]:
                    if param in category_cols and category_cols[param] in df.columns:
                        keep_columns.append(category_cols[param])
        else:  # per_module
            for module_key, module_info in self.column_mapping.items():
                for param in ["P", "P_normalised", "I", "U", "Temp"]:
                    if param in module_info and module_info[param] in df.columns:
                        keep_columns.append(module_info[param])

        existing_keep = [c for c in keep_columns if c in df.columns]
        to_drop = [c for c in df.columns if c not in existing_keep]

        print(f"\nKeeping: {len(existing_keep)} columns")
        print(f"Weather/PV columns: {[c for c in existing_keep if not c.startswith('bad_day_')]}")
        print(f"Bad day columns: {[c for c in existing_keep if c.startswith('bad_day_')]}")
        print(f"Dropping: {len(to_drop)} columns")

        return df[existing_keep].copy()

    def process_data(self, pv_data_path: str, output_path: str, create_plots: bool = True) -> bool:       
        print("\nSTEP 1 : Loading scaling factors")
        print("-"*100)
        if not self.load_scaling_factors():
            return False
        
        print("\nSTEP 2 : Loading PV data")
        print("-"*100)
        pv_df = self.load_and_parse_pv(pv_data_path)
        if pv_df is None:
            return False
        
        pv_start = pv_df['_time'].min()
        pv_end = pv_df['_time'].max()
        
        print("\nSTEP 3 : Loading DWD data")
        print("-"*100)
        dwd_df = self.load_and_parse_dwd()
        if dwd_df is None:
            return False
        
        print("\nSTEP 4 : Filtering DWD to PV time range")
        print("-"*100)
        dwd_filtered = dwd_df[(dwd_df.index >= pv_start) & (dwd_df.index <= pv_end)].copy()
        
        print(f"Original DWD records: {len(dwd_df)}")
        print(f"Filtered DWD records: {len(dwd_filtered)}")
        print(f"PV time range: {pv_start} to {pv_end}")
        
        if len(dwd_filtered) == 0:
            print("WARNING : No DWD data in PV time range")
            return False
        
        print("\nSTEP 5 : Merging PV into DWD backbone")
        print("-"*100)
        merged_df = self.align_pv_to_backbone(pv_df, dwd_filtered)
        if merged_df.empty:
            print("No PV data could be merged with DWD backbone")
            return False
        
        print("\nSTEP 6 : Filling missing irradiance")
        print("-"*100)
        irr_filled_df = self.fill_missing_irradiance(merged_df)
        
        print("\nSTEP 7 : Creating bad_day columns")
        print("-"*100)
        specific_bad_day_df = self.create_specific_bad_day_columns(irr_filled_df)

        print("\nSTEP 8 : Marking before installation")
        print("-"*100)
        before_install_df = self.mark_before_installation_bad(specific_bad_day_df)

        print("\nSTEP 9 : Marking maintenance periods")
        print("-"*100)
        maintenance_period_df = self.mark_maintenance_periods(before_install_df)
        
        print("\nSTEP 10 : Marking low quality days")
        print("-"*100)
        high_quality_df = self.mark_low_quality_days_by_p_count(maintenance_period_df, threshold=50)

        print("\nSTEP 11 : Enforcing bounds and irradiance rules")
        print("-"*100)
        enforced_df = self.enforce_bounds_and_irradiance(high_quality_df)

        print("\nSTEP 12 : Selecting final columns")
        print("-"*100)
        result_df = self.drop_unwanted_columns(enforced_df)

        print(f"\nSTEP 13: Nan handling")
        print("-" * 100)
        result_df = self.finalize_nan_cleanup(result_df, set_nan_to_zero=False) 

        try:
            final_df_reset = result_df.reset_index()
            if "timestamp" in final_df_reset.columns:
                final_df_reset = final_df_reset.rename(columns={"timestamp": "_time"})
            final_df_reset = final_df_reset.sort_values("_time")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            #Save to CSV
            final_df_reset.to_csv(output_path, index=False)
            
            print(f"\nData saved to: {output_path}")
            print(f"Total records: {len(final_df_reset)}")
            print(f"Time range: {final_df_reset['_time'].min()} to {final_df_reset['_time'].max()}")
            print(f"Data structure: {self.data_structure}")
            print(f"Columns ({len(final_df_reset.columns)} total):")
            for i, col in enumerate(final_df_reset.columns):
                if col.startswith('bad_day_'):
                    print(f"  {i+1}. {col} (specific bad day column)")
                else:
                    print(f"  {i+1}. {col}")
            return True
            
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False