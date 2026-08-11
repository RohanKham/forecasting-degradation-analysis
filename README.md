# Forecasting Degradation Analysis

ML based PV power forecasting framework using multi output LSTM model architecture. The framework downloads and merges real time sensor data from WSN (Wireless Sensor Network) and weather data from the DWD (German Weather Service), validates and cleans the data, adds time features, and trains a multi-output LSTM to forecast power output across multiple PV modules simultaneously. The framework also includes sensitivity analysis to identify which input features influences model predictions, Power Conversion Efficiency (PCE) trends, and degradation analysis to generate reference surface, daily and monthly degradation decomposition (reversible vs. irreversible losses) for both measured and predicted power.

# Training LSTM Model

## File Structure

```
forecasting-degradation-analysis/
├── get_influxdb_pv_data.py # PV data download from InfluxDB
├── get_dwd_weather_data.py # DWD weather data download and cleaning
├── data_validation.py # PV data validation and cleaning pipeline
├── merge_pv_and_dwd_data.py # Merges validated PV data with DWD weather data
├── create_features_and_scale.py # Feature engineering and MinMax scaling
├── train_lstm_model.py # Sequence creation and multi-output LSTM model training and evaluation
└── lstm_pipeline_entry.py # Orchestrates the full pipeline end-to-end
```

## File Descriptions

### `get_influxdb_pv_data.py`

Downloads 10-minute aggregated PV measurements (`P`, `I`, `U`, `Temp`, `Irr`) from InfluxDB for all modules. Results are saved to `data_export_<timestamp>.csv`.

### `get_dwd_weather_data.py`

Downloads and merges weather data from two DWD stations:

- Stuttgart-Echterdingen (04931): temperature, humidity, precipitation, cloud cover
- Stuttgart-Schnarrenberg (04928): global radiation, sunshine duration

Output columns: `temp_C`, `humidity`, `precip_mm`, `global_radiation`, `cloud_cover`, `precip_indicator`, `sunshine_duration`
Results are saved to: `weather_data_10min_cleaned.csv`

### `data_validation.py`

```
PVValidationPipeline
├── clean_module_names()            # Normalizes inconsistent module name strings (e.g. Sanyo_2-1 to Sanyo_2_1)
├── align_resample()                # Removes data before each module's installation date and drops duplicates
├── create_reference_copies()       # Saves copies of Sanyo_5_1 and Perovskite_1_1 Irradiance column for irradiance gap filling
├── drop_non_validation_modules()   # Drops modules not listed in validation_modules (configured in `lstm_pipeline_entry.py`)
├── fill_irradiance_gaps()          # Fills missing Irr using merged reference sensors and scaled DWD radiation. Also, saves scaling factors in a JSON file
├── night_detector()                # Handles nighttime rows
├── map_module_metadata()           # Adds module_id, module_type, and category (si/psc) columns
├── range_validation()              # Flags physically implausible values per module type (P, I, U, Irr, Temp bounds)
├── detect_statistical_outliers()   # Flags daytime outliers per module using z-score thresholding
├── detect_correlation_anomalies()  # Flags timesteps where P and Irr changes are decorrelated beyond a minimum ratio
├── generate_feature_nan_masks()    # Creates binary NaN mask columns per feature (e.g. P_nan, Irr_nan)
├── combine_masks()                 # Combines all invalid flags into a single invalid_any column
├── normalise_power_per_module()    # Scales power per module by its 95th percentile; saves factors to JSON
├── compute_module_average()        # Aggregates to category-level (si/psc) or per-module columns
└── run()                           # Saves either pv_averaged_by_category.csv or pv_per_module.csv with categorical or module granularity
```

### `merge_pv_and_dwd_data.py`

```
PVDWDDataMerger
├── detect_data_structure_from_filename()   # Infers category-averaged or per-module structure from filename
├── build_column_mapping()                  # Dynamically maps PV column names to parameters and module metadata
├── load_and_parse_pv()                     # Loads and parses the validated PV csv; removes duplicate timestamps
├── load_and_parse_dwd()                    # Loads DWD csv; resamples to 10 min; fills missing timestamps with NaN
├── align_pv_to_backbone()                  # Left-joins PV data onto the DWD time grid using merge_asof
├── fill_missing_irradiance()               # Applies saved scaling factors to fill Irr gaps
├── create_specific_bad_day_columns()       # Creates bad_day_<category> or bad_day_<module> columns initialized to 0
├── mark_before_installation_bad()          # Flags and nullifies PV data before each category's installation date
├── mark_maintenance_periods()              # Sets PV values to NaN and sets bad_day flags for maintenance windows
├── mark_low_quality_days_by_p_count()      # Marks entire day as bad if count of power readings per day fall below a threshold
├── enforce_bounds_and_irradiance()         # Zeros out-of-range values on good days only
├── drop_unwanted_columns()                 # Retains only PV, weather and bad_day columns for training
├── finalize_nan_cleanup()                  # Reports NaN statistics
└── process_data()                          # Exports merged data to merged_pv_dwd_step.csv
```

### `create_features_and_scale.py`

```
FeatureEnggPipeline
├── load_input_csv()                        # Loads merged_pv_dwd_step.csv
├── add_time_features()                     # Adds cyclical sin/cos encodings for hour, month, day-of-year features
├── add_age_features()                      # Adds fractional days_since_installation_ per technology/module
├── retain_only_features_and_target()       # Drops all columns not in sensor_cols, env_cols, cyclical, age, or targets
├── temporal_fixed_date_split()             # Splits by fixed calendar dates into train / val / test datasets
├── temporal_fixed_date_split_train_test()  # Train/test only split without a validation set
├── scale_features_and_save()               # Fits MinMaxScaler per feature column on train
└── process_pipeline()                      # Saves parquets and column metadata
```

### `train_lstm_model.py`

```
├──MultiOutputLSTMModel                     # LSTM model arch with stacked layers and ReLU activated output
├──MultiOutputDataset                       # Sequence created using overlapping sliding-window mechanism (stride=1)
├──NonOverlappingMultiOutputDataset         # Sequence creation using non-overlapping variant (stride=horizon) used for evaluation
├── detect_bad_day_columns()                # Maps each target to its bad_day_ column
├── masked_mae_with_multiple_bad_days()     # MAE loss on good-day timesteps only; bad days contribute zero gradient
├── split_train_val_from_sequences()        # Splits train sequences into train/val using modulo logic
├── train_loop()                            # Adam + ReduceLROnPlateau scheduler, gradient clipping, early stopping, best-model checkpointing
├── predict_with_loader()                   # Runs inference; returns stacked preds, trues, bad masks
├── build_continuous_series_multi_output()  # Reconstructs continuous time series per target from non-overlapping predictions
├── compute_metrics_from_continuous_series()# Computes error metrics per target
├── save_target_visualizations()            # Saves per-target scatter plots, error histograms, and time series plots
├── save_all_metrics_to_txt()               # Writes full metrics report for all targets and splits to a text file
└── run_lstm_training_multi_output()        # Top-level entry point
```

### `lstm_pipeline_entry.py`

Imports all the files and runs each class. Each run produces a timestamped output folder:

```
lstm_run_YYYY_MM_DD_HHMMSS/
├── data_export_<timestamp>.csv # Raw PV data from InfluxDB
├── pv_cleaned_masked.csv # Cleaned per-module PV data
├── pv_per_module.csv # Aggregated per-module PV data
├── dwd_irradiance_scaling_factors.json # Irradiance scaling factors
├── validation_plot/
│ ├── irradiance_gap_filling_overview.png
│ └── monthly_period_scaling_factors.png
├── merged_pv_dwd_step.csv # Merged PV + DWD dataset
├── training_data/
│ ├── train_scaled.parquet
│ ├── val_scaled.parquet
│ ├── test_scaled.parquet
│ ├── scalers.pkl # Feature scalers
│ ├── p_scalers.pkl # Per-target power scalers
│ └── column_info.pkl # Feature and target column metadata
└── lstm_results/
├── <timestamp>\_model_multi.pt # Best model
├── training_metrics_multi.txt # Metrics report per target per split
├── model_info_multi.json # Architecture, hyperparameters, etc
└── plots_multi/
├── training_history.png
├── <target>\_scatter.png
├── <target>\_histogram.png
└── <target>\_continuous.png
```

## Steps to run the project

```bash
# 1. Clone repository
git clone <repository-url>
cd forecasting-degradation-analysis

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# PyTorch according to operating system:
# macOS:
pip install -r requirements-torch_mac.txt

# 5. Configure .env file
# Copy the example file:
cp .env.example .env

# Open .env and replace the placeholder values with your credentials:
INFLUXDB_URL=https://your-influxdb-host:8086
INFLUXDB_TOKEN=your_api_token_here
INFLUXDB_ORG=your_org_name
INFLUXDB_BUCKET=your_bucket_name

# 6. Configure lstm_pipeline_entry.py
# Set the date range, weather CSV path, validation_modules list, and target_cols list.
# Ensure target_cols in lstm_pipeline_entry.py matches the sensor_cols and target_cols defined in create_features_and_scale.py

# 7. Run complete pipeline
python lstm_pipeline_entry.py
```

# Sensitivity and Degradation Analysis

## `lstm_sensitivity_analysis.py`

This file performs feature perturbation sensitivity analysis to quantify how changes in input features affect the LSTM model's power predictions. It systematically perturbs each feature (e.g., irradiance, temperature, humidity) and measures the resulting change in predicted power output. It creates following output files,

```
sensitivity_analysis/
├── plots/
│ ├── p_irr_<target>merged_raw.png      # Raw 10-min PCE trend with OLS
│ ├── p_irr<target>merged_mean.png      # Daily mean PCE ±1 SD with trend
│ ├── p_irr<target>merged_median.png    # Daily median PCE with trend
│ └── p_irr<target>_merged_p90.png      # Daily 90th percentile PCE trend
└── sensitivity_report.txt              # Complete analysis report
```

## `degradation_analysis.py`

This code file computes degradation metrics using a reference surface method. It decomposes daily energy losses into reversible and irreversible components for both measured and predicted power. It creates following output files,

```
pce_analysis_output/
├── aligned_df.parquet                      # Full aligned DataFrame with PCE
├── daily_degradation.csv                   # Per-day degradation metrics
├── reference_surface_heatmap.png           # PCE lookup table heatmap
├── coverage_heatmap_<target>.png           # Bin coverage analysis
├── pce_temperature_dependence.png          # PCE vs temperature scatter
├── pce_irradiance_dependence.png           # PCE vs irradiance scatter
├── panel_temp_vs_irradiance_<module>.png   # Panel temperature correlation
├── ambient_temp_vs_irradiance.png          # Ambient temperature correlation
├── daily_loss_bars_valid_only_<dates>.png  # Daily loss decomposition
└── monthly_loss_bars_with_diffs.png        # Monthly aggregated losses
```

# Results

## Forecasting Performance

The LSTM model achieved high forecasting accuracy for silicon modules ($R^2$ > 0.9) and moderate accuracy for perovskite modules ($R^2$ > 0.7). Introducing the **age feature** substantially improved the forecasting performance of the perovskite models (see the figure below) by enabling the LSTM to learn their long-term degradation behaviour directly from operational data. In contrast, the silicon modules showed only marginal improvement because their performance remained comparatively stable over the observation period, providing little additional information through the age feature.

<img width="1564" height="1490" alt="single_module_baselines_1" src="https://github.com/user-attachments/assets/2f143cdc-e930-4094-8138-e68c011a468e" />

## Feature Perturbation Analysis

Feature perturbation analysis was performed to evaluate whether the trained LSTM learned physically meaningful relationships between the input variables and module power output. The results show that **irradiance** is the dominant input feature for all modules and exhibits an approximately linear relationship with predicted power across the investigated perturbation magnitudes. **Temperature** and **relative humidity** produce smaller but physically consistent responses, with silicon modules exhibiting greater temperature sensitivity than perovskite modules, reflecting their larger temperature coefficients. **Cloud cover** displays a nonlinear and asymmetric response, where decreasing cloud cover produces a substantially larger increase in predicted power than the corresponding decrease caused by increasing cloud cover. The **age feature** exhibits the strongest nonlinear behaviour for the perovskite modules, demonstrating that the model has learned their long-term degradation dynamics.

<img width="1578" height="979" alt="normalised_perturbation_positive" src="https://github.com/user-attachments/assets/1e11676f-1331-4943-b69e-a276efb99c24" />
<img width="1578" height="979" alt="normalised_perturbation_negative" src="https://github.com/user-attachments/assets/1c0b0062-e9a5-4acb-bd57-fb50171a0062" />

## Degradation Analysis

Reversible–irreversible degradation decomposition framework originally developed for laboratory measurements is adapted for outdoor operational data. The predicted power successfully reproduces the temporal evolution of irreversible degradation, demonstrating that the model captures long-term degradation behaviour despite being trained solely on power output. The resulting irreversible degradation component is reproduced with a monthly **$R^2$ ∼ 0.96**, while the faster-varying reversible component is captured with lower precision, reflecting its greater sensitivity to the decomposition framework and to small deviations in the model’s predicted signal

<img width="1312" height="920" alt="monthly_loss_bars_psc_1" src="https://github.com/user-attachments/assets/c31569ac-4076-48c2-8a2f-710944bcaf0c" />
<img width="1312" height="920" alt="monthly_loss_bars_psc_2" src="https://github.com/user-attachments/assets/0083ccf0-3fdd-4e18-99ef-fbc16fa099c2" />

# Conclusions

This work demonstrates that LSTM-based forecasting models can provide information beyond conventional prediction metrics and serve as practical tools for monitoring degradation in perovskite photovoltaic modules. 

Certain limitations should be considered when interpreting the results:
- The degradation decomposition framework relies on a reference efficiency surface constructed using irradiance and temperature only. Environmental factors such as humidity, cloud cover, and spectral variations are not explicitly represented.
- As a consequence, reversible degradation estimates can be systematically overestimated under outdoor conditions.
