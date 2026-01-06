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
from train_lstm_model import run_lstm_training

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

def get_dwd_data(output_csv: str) -> None:
    """
    Download and clean DWD weather data and save to CSV.

    Args:
        output_csv: Path where cleaned weather data will be saved
    """
    downloader = DWDDownloader(output_path=output_csv)
    cleaned = downloader.run()

    print("\nWeather data columns:")
    print(cleaned.columns.tolist())
    print("\nWeather data preview:")
    print(cleaned.head())

def run_pv_validation_pipeline(df_pv, weather_csv, run_folder):
    """
    Run PV validation pipeline with PV data and weather data CSV.
    Saves cleaned and normalized outputs in run_folder.
    
    Args:
        df_pv (pd.DataFrame): Raw PV data
        weather_csv (str): Path to cleaned DWD weather CSV
        run_folder (str): Folder to save output CSVs
    
    Returns:
        cleaned_df, normalized_df
    """
    validation_modules = [
        "Sanyo_1_1", "Sanyo_2_1", "Sanyo_3_1", "Sanyo_4_1", "Sanyo_5_1",
        #"Solon_1_1", "Solon_1_1", "Solon_2_1", "Solon_3_1", "Solon_4_1", "Solon_5_1",
        #"Perovskite_1_1", "Perovskite_1_2", #"Perovskite_1_3", 
        #"Perovskite_2_1", "Perovskite_2_2", "Perovskite_2_3", 
        #"Perovskite_3_1", "Perovskite_3_2", "Perovskite_3_3",
        #"Perovskite_4_1", "Perovskite_4_2", "Perovskite_4_3"
    ]
    flag_invalid = False
    
    print("\nRunning PV Validation Pipeline.")
    pipeline = PVValidationPipeline(df_pv, validation_modules=validation_modules, results_dir=run_folder)
    cleaned_df, normalized_df = pipeline.run(dwd_file=weather_csv, flag_invalid=flag_invalid)
    cleaned_out = os.path.join(run_folder, "pv_cleaned_masked.csv")
    normalized_out = os.path.join(run_folder, "pv_normalized.csv")
    
    print(f"\nValidation done. Files saved by pipeline:")
    print(f"Cleaned PV data: {cleaned_out}")
    print(f"Normalized PV data: {normalized_out}")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(f"Normalized rows: {len(normalized_df):,}")
    
    return cleaned_df, normalized_df

def merge_pv_dwd_data(pv_csv_path, dwd_csv_path, scaling_json_path, run_folder):
    """
    Merge PV and DWD data using the 6-step merger.
    
    Args:
        pv_csv_path (str): Path to normalized PV CSV
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
    end_date = "2025-12-30"
    
    output_csv = os.path.join(run_folder, f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    df_pv = get_pv_data(start_date, end_date, output_csv)
    
    if not df_pv.empty:
        print(f"\nDownloaded data shape: {df_pv.shape}")
        print(df_pv.head())
    else:
        print("No data downloaded.")
    print(f"\nSaved pv data to: {output_csv}")

    #download dwd weather data
    weather_csv = os.path.join(run_folder, "weather_data_10min_cleaned.csv")
    get_dwd_data(weather_csv)
    print(f"\nWeather data saved to: {weather_csv}")

    #validate pv data
    try:
        cleaned_df, normalized_df = run_pv_validation_pipeline(df_pv, weather_csv, run_folder)        
        #check duplicates and NaNs for cleaned_df
        print("\nCleaned DataFrame")
        num_duplicates_cleaned = cleaned_df.duplicated().sum()
        print(f"Number of duplicate rows in cleaned_df: {num_duplicates_cleaned}")
        if num_duplicates_cleaned > 0:
            print("Duplicate rows indices:", cleaned_df[cleaned_df.duplicated()].index.tolist())
        
        nan_counts_cleaned = cleaned_df.isna().sum()
        print("NaN counts per column in cleaned_df:")
        print(nan_counts_cleaned[nan_counts_cleaned > 0])
        #check duplicates and NaNs for normalized_df 
        print("\nNormalized DataFrame")
        num_duplicates_normalized = normalized_df.duplicated().sum()
        print(f"Number of duplicate rows in normalized_df: {num_duplicates_normalized}")
        if num_duplicates_normalized > 0:
            print("Duplicate rows indices:", normalized_df[normalized_df.duplicated()].index.tolist())
        
        nan_counts_normalized = normalized_df.isna().sum()
        print("NaN counts per column in normalized_df:")
        print(nan_counts_normalized[nan_counts_normalized > 0])
        
    except Exception as e:
        print(f"Error running PV validation pipeline: {e}")
        raise
    
    #merge pv data with dwd weather data 
    normalized_pv_path = os.path.join(run_folder, "pv_normalized.csv")
    scaling_json_path = os.path.join(run_folder, "dwd_irradiance_scaling_factors.json")
    merged_csv_path = merge_pv_dwd_data(
        pv_csv_path=normalized_pv_path,
        dwd_csv_path=weather_csv,
        scaling_json_path=scaling_json_path,
        run_folder=run_folder
    )
    
    #feature engineering
    training_data_dir = os.path.join(run_folder, "training_data")
    pipeline = FeatureEnggPipeline(
        out_dir = training_data_dir,
        window = 48,
        horizon = 12
    )
    print("\nFeature engineering and splitting")
    pipeline.process_pipeline(input_csv=merged_csv_path)

    print("\nStarting LSTM Training")
    try:
        # Run LSTM training
        training_results = run_lstm_training(
            training_data_dir=training_data_dir,
            output_dir=os.path.join(run_folder, "lstm_results"),  
            window=48,
            horizon=12,
            batch_size=36,
            epochs=250,
            lr=0.0001,
            patience=25,
            hidden_size=64,
            num_layers=4,
            dropout=0.1,
            target_col="P",
            use_bad_day=True,
            mask_bad_days=True,
        )
 
        print(f"All results saved in: {run_folder}")
        print(f"Training data: {training_data_dir}")
        print(f"LSTM results: {os.path.join(run_folder, 'lstm_results')}")
        
    except Exception as e:
        print(f"\nERROR in LSTM training: {e}")
        raise
    