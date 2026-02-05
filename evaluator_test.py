#Version: 1.0 
#Handles only 1 output
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from get_influxdb_pv_data import InfluxDBDataExporter
from get_dwd_weather_data import DWDDownloader
from data_validation import PVValidationPipeline
from merge_pv_and_dwd_data import PVDWDDataMerger
from create_features_and_scale import FeatureEnggPipeline
from train_lstm_model import run_lstm_training

def create_run_folder(base_name="evaluate"):
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
    Saves cleaned and averaged outputs in run_folder.
    
    Args:
        df_pv (pd.DataFrame): Raw PV data
        weather_csv (str): Path to cleaned DWD weather CSV
        run_folder (str): Folder to save output CSVs
    
    Returns:
        tuple: (cleaned_df, normalized_df)
    """
    validation_modules = [
        "Atersa_1_1", "Atersa_2_1", "Atersa_3_1", "Atersa_4_1", "Atersa_5_1", "Atersa_6_1",
        "Atersa_1-1", "Atersa_2-1", "Atersa_3-1", "Atersa_4-1", "Atersa_5-1", "Atersa_6-1", 
        "Sanyo_1_1", "Sanyo_2_1", "Sanyo_3_1", "Sanyo_4_1", "Sanyo_5_1", 
        "Sanyo_1-1", "Sanyo_2-1", "Sanyo_3-1", "Sanyo_4-1", "Sanyo_5-1", 
        "Solon_1_1","Solon_1_2", "Solon_2_1", "Solon_2_2", "Solon_3_1", "Solon_3_2", "Solon_4_2", 
        "Solon_1-1","Solon_1-2", "Solon_2-1", "Solon_2-2", "Solon_3-1", "Solon_3-2", "Solon_4-2", 
        "Sun_Power_1_1", "Sun_Power_2_1", "Sun_Power_3_1", "Sun_Power_4_1", "Sun_Power_5_1",
        "Sun_Power_1-1", "Sun_Power_2-1", "Sun_Power_3-1", "Sun_Power_4-1", "Sun_Power_5-1",

        #"Perovskite_1", "Perovskite_1_1", "Perovskite_1_2", "Perovskite_1_3", 
        #"Perovskite_2", "Perovskite_2_1", "Perovskite_2_2", "Perovskite_2_3", 
        #"Perovskite_3_1", "Perovskite_3_2", "Perovskite_3_3",
        #"Perovskite_4_1", "Perovskite_4_2", "Perovskite_4_3"
    ]
    flag_invalid = False
    
    print("\nRunning PV Validation Pipeline.")
    pipeline = PVValidationPipeline(df_pv, validation_modules=validation_modules, results_dir=run_folder)
    cleaned_df, averaged_df = pipeline.run(dwd_file=weather_csv, flag_invalid=flag_invalid)
    cleaned_out = os.path.join(run_folder, "pv_cleaned_masked.csv")
    normalized_out = os.path.join(run_folder, "pv_normalized.csv")
    
    print(f"\nValidation done. Files saved by pipeline:")
    print(f"Cleaned PV data: {cleaned_out}")
    print(f"Averaged PV data: {normalized_out}")
    print(f"Cleaned rows: {len(cleaned_df):,}")
    print(f"Averaged rows: {len(averaged_df):,}")
    
    return cleaned_df, averaged_df

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
    start_date = "2025-07-16"
    end_date = "2025-09-14"
    
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
        cleaned_df, averaged_df = run_pv_validation_pipeline(df_pv, weather_csv, run_folder)        
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
        num_duplicates_normalized = averaged_df.duplicated().sum()
        print(f"Number of duplicate rows in averaged_df: {num_duplicates_normalized}")
        if num_duplicates_normalized > 0:
            print("Duplicate rows indices:", averaged_df[averaged_df.duplicated()].index.tolist())
        
        nan_counts_normalized = averaged_df.isna().sum()
        print("NaN counts per column in averaged_df:")
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
        horizon = 36
    )
    print("\nFeature engineering and splitting")
    test_df = pipeline.load_input_csv(merged_csv_path)
    test_df = pipeline.add_time_features(test_df)
    test_df = pipeline.retain_only_features_and_target(test_df)

    SCALER_FILE = "lstm_run_2026_01_31_182240_si/training_data/scalers.pkl" #change this
    P_SCALER_FILE = "lstm_run_2026_01_31_182240_si/training_data/p_scaling.pkl" #change this

    with open(SCALER_FILE, "rb") as f:
        scaler_payload = pickle.load(f)
    feature_scalers = scaler_payload["feature_scalers"]
    print(f"Loaded feature sclaers for: {list(feature_scalers.keys())}")

    with open(P_SCALER_FILE, "rb") as f:
        p_info = pickle.load(f)
    p_scaler = p_info["scaler"]
    print(f"Loaded target scaler with range: [{p_info['data_min']:.6f}, {p_info['data_max']:.6f}]")

    # Transform test data using the loaded scalers
    def transform_df(df, feature_scalers, p_scaler, target_col):
        """Transform dataframe using pre-fitted scalers"""
        df_t = df.copy()
        for col, scaler in feature_scalers.items():
            if col in df_t.columns:
                col_vals = df_t[[col]].astype(float)
                mask = col_vals.notna().values.ravel()
                if mask.any():
                    transformed = scaler.transform(col_vals[mask])
                    df_t.loc[col_vals.index[mask], col] = transformed.flatten()
        
        # Transform target column
        if target_col in df_t.columns:
            p_vals = df_t[[target_col]].astype(float)
            mask = p_vals.notna().values.ravel()
            if mask.any():
                df_t.loc[p_vals.index[mask], target_col] = (
                    p_scaler.transform(p_vals[mask])[:, 0]
                )
        return df_t
    
    test_scaled = transform_df(test_df, feature_scalers, p_scaler, pipeline.target_col)

    # Save the scaled test data
    test_output_path = os.path.join(training_data_dir, "test_scaled.parquet")
    test_scaled.to_parquet(test_output_path)
    
    print(f"\nTest data preparation complete!")
    print(f"Test data shape: {test_scaled.shape}")
    print(f"Test data date range: {test_scaled.index.min()} to {test_scaled.index.max()}")
    print(f"Saved scaled test data to: {test_output_path}")
    
    # Print summary statistics
    print("\nTest data summary (after scaling):")
    all_feature_cols = list(dict.fromkeys(pipeline.sensor_cols + pipeline.env_cols))
    pipeline.print_stats(test_scaled, all_feature_cols + [pipeline.target_col], "TEST (scaled)")
    pipeline.print_stats(test_scaled, pipeline.engineered_cyclical, "TEST engineered_cyclical (unscaled)")

   #evaluation section
    MODEL_PATH = "lstm_run_2026_01_31_182240_si/lstm_results/31.01.2026.182441_model.pt" #change this
    MODEL_INFO_PATH = "lstm_run_2026_01_31_182240_si/lstm_results/model_info.pkl" #change this
    TEST_DATA_PATH = test_output_path
    OUTPUT_DIR = "lstm_run_2026_01_31_182240_si/lstm_results/test_evaluation_si_full" #change this
    BATCH_SIZE = 32
    RANDOM_SEED = 42

    print("Running evaluator code")
    print(f"Model: {MODEL_PATH}")
    print(f"Model info: {MODEL_INFO_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Scaler: {P_SCALER_FILE}") 
    print(f"Output: {OUTPUT_DIR}")

    try:
        from evaluator import evaluate_test_only     
        test_results = evaluate_test_only(
            model_path=MODEL_PATH,
            model_info_path=MODEL_INFO_PATH,
            test_data_path=TEST_DATA_PATH,
            scaler_path=P_SCALER_FILE,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            random_seed=RANDOM_SEED,
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()

    
    

    