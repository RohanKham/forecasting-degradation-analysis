#version 3.0
#Handles training for more than one module
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from get_influxdb_pv_data import InfluxDBDataExporter
from get_dwd_weather_data import DWDDownloader
from data_validation import PVValidationPipeline
from merge_pv_and_dwd_data import PVDWDDataMerger
from create_features_and_scale import FeatureEnggPipeline
from train_lstm_model import run_lstm_training_multi_output

def create_run_folder(base_name="lstm_run"):
    """
    Create a folder with name base_name_yyyy_mm_dd_HHMMSS
    """
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"\nCreated folder: {folder_name}")
    return folder_name

def get_pv_data(start_date, end_date, output_path=None):
    """
    Download data from InfluxDB.
    
    Args:
        start_date (str): Start date YYYY-MM-DD
        end_date (str): End date YYYY-MM-DD
        output_path (str): Optional output CSV file path
    
    Returns:
        pd.DataFrame: Downloaded data
    """
    
    load_dotenv()
    exporter = InfluxDBDataExporter(
        url=os.environ["INFLUXDB_URL"],
        token=os.environ["INFLUXDB_TOKEN"],
        org=os.environ["INFLUXDB_ORG"],
        bucket=os.environ.get("INFLUXDB_BUCKET", "Uni"),
    )

    #check .env file values
    print("\nINFLUXDB_URL:", os.getenv("INFLUXDB_URL"))
    print("INFLUXDB_TOKEN loaded:", bool(os.getenv("INFLUXDB_TOKEN")))
    print("INFLUXDB_ORG:", os.getenv("INFLUXDB_ORG"))
    print("INFLUXDB_BUCKET:", os.getenv("INFLUXDB_BUCKET"))
    
    try:
        all_modules = exporter.get_all_modules()
        df = exporter.query_data(
            module_names=all_modules,
            start_date=start_date,
            end_date=end_date,
            save_to_csv=True,
            output_path=output_path
        )
        return df
    finally:
        exporter.close()

def get_dwd_data(output_csv: str, start_date: str = None, end_date: str = None) -> None:
    """
    Download and clean DWD weather data and save to CSV.
    Addition: Use an existing file which covers the requested date range, instead of downloading.
    """
    if os.path.exists(output_csv):
        print(f"\nFound existing weather file: {output_csv}")
        
        try:
            existing_df = pd.read_csv(output_csv, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(output_csv, nrows=0).columns else None)
            
            if 'timestamp' in existing_df.columns:
                file_start = existing_df['timestamp'].min()
                file_end = existing_df['timestamp'].max()
                
                print(f"Existing data covers: {file_start} to {file_end}")
                
                # Check if requested dates are covered
                if start_date and end_date:
                    requested_start = pd.to_datetime(start_date).tz_localize('UTC')
                    requested_end = pd.to_datetime(end_date).tz_localize('UTC')
                    if file_start.tz is None:
                        file_start = file_start.tz_localize('UTC')
                    if file_end.tz is None:
                        file_end = file_end.tz_localize('UTC')
                    if file_start <= requested_start and file_end >= requested_end:
                        print(f"Existing file already covers requested range ({start_date} to {end_date})")
                        print(f"Using existing file: {output_csv}")
                        return
                    else:
                        print(f"WARNING: Existing file does not fully cover requested range ({start_date} to {end_date})")
                        print(f"Missing: {min(file_start, requested_start)} to {max(file_end, requested_end)}")
                        print(f"Downloading fresh data.")
                else:
                    # No date range specified, just use existing file
                    print(f"Using existing file: {output_csv}")
                    return
            else:
                print(f"WARNING: Could not verify date coverage (no 'timestamp' column)")
                print(f"Downloading fresh data to be safe.")
                
        except Exception as e:
            print(f"WARNING: Could not read existing file: {e}")
            print(f"Downloading fresh data.")
    else:
        print(f"\nWARNING: No existing weather file found at {output_csv}")
        print(f"Downloading fresh data.")

    # Download fresh data
    downloader = DWDDownloader(output_path=output_csv)
    cleaned = downloader.run()

    print("\nWeather data columns:")
    print(cleaned.columns.tolist())
    print("\nWeather data preview:")
    print(cleaned.head())
    
    # Verify the downloaded data covers the requested range
    if start_date and end_date and 'timestamp' in cleaned.columns:
        file_start = cleaned['timestamp'].min()
        file_end = cleaned['timestamp'].max()
        requested_start = pd.to_datetime(start_date).tz_localize('UTC')
        requested_end = pd.to_datetime(end_date).tz_localize('UTC')
        
        if file_start.tz is None:
            file_start = file_start.tz_localize('UTC')
        if file_end.tz is None:
            file_end = file_end.tz_localize('UTC')
        
        print(f"\nDownloaded data covers: {file_start} to {file_end}")
        if file_start <= requested_start and file_end >= requested_end:
            print(f"Downloaded data covers requested range ({start_date} to {end_date})")
        else:
            print(f"WARNING: Downloaded data does not fully cover requested range!")
            print(f"Missing: {min(file_start, requested_start)} to {max(file_end, requested_end)}")

def run_pv_validation_pipeline(df_pv, weather_csv, run_folder):
    """
    Run PV validation pipeline with PV data and weather data CSV.
    Saves cleaned and averaged outputs in run_folder.
    
    Args:
        df_pv (pd.DataFrame): Raw PV data
        weather_csv (str): Path to cleaned DWD weather CSV
        run_folder (str): Folder to save output CSVs
    
    Returns:
        cleaned_df, averaged_df
    """
    # List all the modules you wish to train for 
    validation_modules = ["Perovskite_1_1"]
    flag_invalid = False
    
    print("\nRunning PV Validation Pipeline.")
    pipeline = PVValidationPipeline(df_pv, validation_modules=validation_modules, results_dir=run_folder)
    """
    flag_invalid: 
        True  : Keep invalid rows but flag them (for model masking).
        False : Drop all invalid rows.
    normalise (bool):
        True  : Normalize power using p95 scaling per module.
        False : Keep raw power values.
    average_modules (bool):
        True  : Average by category (si/psc).
        False : Keep per-module columns (module_type + module_id).
    """
    cleaned_df, averaged_df, averaged_csv_path = pipeline.run(dwd_file=weather_csv, flag_invalid=flag_invalid, normalise=False, average_modules=False)
    cleaned_out = os.path.join(run_folder, "pv_cleaned_masked.csv")
    
    print(f"\nValidation done. Files saved by pipeline:")
    print(f"Cleaned PV data: {cleaned_out}")
    print(f"Averaged PV data: {averaged_csv_path}")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(f"Averaged rows: {len(averaged_df):,}")
    
    return cleaned_df, averaged_df, averaged_csv_path

def merge_pv_dwd_data(pv_csv_path, dwd_csv_path, scaling_json_path, run_folder):
    """
    Merge PV and DWD data using the 6-step merger.
    
    Args:
        pv_csv_path (str): Path to averaged PV CSV
        dwd_csv_path (str): Path to cleaned DWD weather CSV
        scaling_json_path (str): Path to scaling factors JSON
        run_folder (str): Folder to save output
    
    Returns:
        str: Path to merged CSV file
    """
    print("Merging PV and DWD Data.")
    
    merged_output_path = os.path.join(run_folder, "merged_pv_dwd_step.csv")
    merger = PVDWDDataMerger(
        dwd_data_path=dwd_csv_path,
        scaling_factors_path=scaling_json_path
    )
    success = merger.process_data(
        pv_data_path=pv_csv_path,
        output_path=merged_output_path
    )
    
    if success:
        print(f"\nMerge completed successfully.")
        merged_df = pd.read_csv(merged_output_path)
        print(merged_df.head())        
        return merged_output_path
    else:
        print(f"\nMerge failed!")
        return None

if __name__ == "__main__":
    #create base folder 
    run_folder = create_run_folder()

    #download pv data from influxdb
    start_date = "2024-11-28" 
    end_date = "2026-06-18"
    
    output_csv = os.path.join(run_folder, f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    df_pv = get_pv_data(start_date, end_date, output_csv)
    
    if not df_pv.empty:
        print(f"\nDownloaded data shape: {df_pv.shape}")
        print(df_pv.head())
    else:
        print("No data downloaded.")
    print(f"\nSaved pv data to: {output_csv}")

    #download dwd weather data or use existing file
    weather_csv = "/Users/rohansanjaykhamkar/Rohan_Khamkar/Stuttgart University/HiWi IPV/raspi_codes/weather_data_10min_cleaned.csv"
    get_dwd_data(weather_csv, start_date=start_date, end_date=end_date)
    print(f"\nWeather data ready: {weather_csv}")

    #validate pv data
    try:
        cleaned_df, averaged_df, averaged_pv_path = run_pv_validation_pipeline(df_pv, weather_csv, run_folder)        
        #check duplicates and NaNs for cleaned_df
        print("\nCleaned DataFrame")
        num_duplicates_cleaned = cleaned_df.duplicated().sum()
        print(f"Number of duplicate rows in cleaned_df: {num_duplicates_cleaned}")
        if num_duplicates_cleaned > 0:
            print("Duplicate rows indices:", cleaned_df[cleaned_df.duplicated()].index.tolist())
        
        nan_counts_cleaned = cleaned_df.isna().sum()
        print("NaN counts per column in cleaned_df:")
        print(nan_counts_cleaned[nan_counts_cleaned > 0])
        #check duplicates and NaNs for averaged_df 
        print("\nAveraged DataFrame")
        num_duplicates_averaged = averaged_df.duplicated().sum()
        print(f"Number of duplicate rows in averaged_df: {num_duplicates_averaged}")
        if num_duplicates_averaged > 0:
            print("Duplicate rows indices:", averaged_df[averaged_df.duplicated()].index.tolist())
        
        nan_counts_averaged = averaged_df.isna().sum()
        print("NaN counts per column in averaged_df:")
        print(nan_counts_averaged[nan_counts_averaged > 0])
        
    except Exception as e:
        print(f"Error running PV validation pipeline: {e}")
        raise
    
    #merge pv data with dwd weather data 
    scaling_json_path = os.path.join(run_folder, "dwd_irradiance_scaling_factors.json")
    merged_csv_path = merge_pv_dwd_data(
        pv_csv_path=averaged_pv_path,
        dwd_csv_path=weather_csv,
        scaling_json_path=scaling_json_path,
        run_folder=run_folder
    )
    
    #feature engineering
    training_data_dir = os.path.join(run_folder, "training_data")
    pipeline = FeatureEnggPipeline(
        out_dir = training_data_dir,
        window = 48,
        horizon = 36
    )
    print("\nFeature engineering and splitting")
    #create_val=True, creates a separate continuous validation df; orelse val is % of train sequences.
    pipeline.process_pipeline(input_csv=merged_csv_path, create_val=False)

    print("\nStarting LSTM Training")
    try:
        training_results = run_lstm_training_multi_output(
            training_data_dir=training_data_dir,
            output_dir=os.path.join(run_folder, "lstm_results"),  
            window=48,
            horizon=36,
            batch_size=32,
            epochs=150,
            lr=0.0005, #PSC:0.0005 | si: 0.001
            patience=15,
            hidden_size=64,
            num_layers=4,
            dropout=0.2,
            target_cols=["P_perovskite_1"],  #lst all the targets here
            use_bad_day=False,   #Include bad_day as input feature
            mask_bad_days=True, #Use bad_day column to mask loss 
            validation_split= 0.15, #if create_val=False, then validation_split=0.15 will select every 7th sequence from traing sequences
            random_seed = 42,
        )
 
        print(f"\nAll results saved in: {run_folder}")
        print(f"Training data: {training_data_dir}")
        print(f"LSTM results: {os.path.join(run_folder, 'lstm_results')}")
        
    except Exception as e:
        print(f"\nERROR in LSTM training: {e}")
        raise
    