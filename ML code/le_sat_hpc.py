"""
===============================================================================
MULTIMODAL LIFE EXPECTANCY PREDICTION - UNIFIED HPC PIPELINE
===============================================================================
This script combines:
PHASE 1: Individual Dataset Analysis (USDA, JRC, etc.)
PHASE 2: Production ML Analysis (Feature Pruning, Final Model, SHAP, Tables)

Designed for Dedicated HPC Node Execution.
===============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import time
import gc
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import shap
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HPC CONFIGURATIONS & MASTER DATA LOAD
# ============================================================================
print("="*80)
print("MULTIMODAL LIFE EXPECTANCY PREDICTION - UNIFIED HPC PIPELINE")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

DATA_FILE = './full_clean_engineered_dataset_with_LE.csv'

OUTPUT_DIR_PHASE1 = Path('results_individual_modalities_final')
OUTPUT_DIR_PHASE2 = Path('results_production_final')
OUTPUT_DIR_PHASE1.mkdir(exist_ok=True)
OUTPUT_DIR_PHASE2.mkdir(exist_ok=True)

print(f"\n[MASTER INIT] Loading dataset into memory from: {DATA_FILE}")
try:
    df_master = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df_master):,} total rows from engineered CSV.")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the dataset at {DATA_FILE}. Please check the path on your cluster.")

# Clean Target globally to ensure consistency
if 'MeanLifeExpectency_x' in df_master.columns:
    df_master['MeanLifeExpectency'] = df_master['MeanLifeExpectency_x']
    df_master = df_master.drop(columns=['MeanLifeExpectency_x', 'MeanLifeExpectency_y'], errors='ignore')

initial_len = len(df_master)
df_master = df_master.dropna(subset=['MeanLifeExpectency']).reset_index(drop=True)
print(f"  Removed {initial_len - len(df_master)} rows missing Life Expectancy data.")

# Standardize FIPS codes globally
df_master['fips'] = df_master['fips'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)

# ----------------------------------------------------------------------------
# [TWEAK 1] STRICT CONUS FILTERING: Drop Alaska (02) and Hawaii (15)
# ----------------------------------------------------------------------------
len_before_conus = len(df_master)
df_master = df_master[~df_master['fips'].str.startswith(('02', '15'))].reset_index(drop=True)
print(f"  Removed {len_before_conus - len(df_master)} rows belonging to Alaska and Hawaii to restrict to CONUS.")


# ============================================================================
# ============================================================================
# PHASE 1: INDIVIDUAL MODALITY ANALYSIS
# ============================================================================
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: INDIVIDUAL MODALITY ANALYSIS")
print("="*80)

# Create a fresh copy for Phase 1
df = df_master.copy()

N_FOLDS = 5
RANDOM_STATE = 42

# Best parameters per modality
BEST_PARAMS_BY_MODALITY = {
    'Engineered': {'n_estimators': 2000, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_depth': 20, 'max_features': 0.5, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'USDA': {'n_estimators': 3000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'JRC': {'n_estimators': 1500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'DEM': {'n_estimators': 1500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'Landsat': {'n_estimators': 3000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'Soil': {'n_estimators': 1500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'S1': {'n_estimators': 2500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'S2': {'n_estimators': 2500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 8, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'Livestock': {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 30, 'max_features': 0.5, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'LST': {'n_estimators': 2500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'NDVI': {'n_estimators': 2500, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10, 'max_features': 0.8, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'Combined': {'n_estimators': 2000, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_depth': 40, 'max_features': 0.3, 'bootstrap': True, 'random_state': RANDOM_STATE, 'n_jobs': -1}
}

MODALITY_PREFIXES = {
    'USDA': ['USDA_'],
    'JRC': ['JRC_'],
    'DEM': ['DEM_'],
    'Landsat': ['Landsat_'],
    'Soil': ['Soil_'],
    'S1': ['S1_'],
    'S2': ['S2_'],
    'Livestock': ['mean_', 'head_count_', 'std_dev_', 'std_error_', 'count_', 'sum_', 'area_', 'usable_', 'growth_'], 
    'LST': ['LST_'],
    'NDVI': ['NDVI_EVI_', 'NDVI_mean', 'NDVI_p', 'NDVI_std'],
    'Engineered': ['ENG_']
}

y = df['MeanLifeExpectency'].to_numpy(dtype=float)
groups = df['fips'].to_numpy(dtype=str)

summary_results = []
all_modalities_results = {}

gkf = GroupKFold(n_splits=N_FOLDS)

for modality, prefixes in MODALITY_PREFIXES.items():
    print(f"\n{'-'*60}")
    print(f"ANALYZING MODALITY: {modality}")
    print(f"{'-'*60}")
    
    feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)]
    
    if not feature_cols:
        print(f"  ⚠️  No features found for {modality}. Skipping...")
        continue
    
    print(f"  Found {len(feature_cols)} features matching prefixes: {prefixes}")
    
    modality_output_dir = OUTPUT_DIR_PHASE1 / modality
    modality_output_dir.mkdir(exist_ok=True)
    
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    best_params = BEST_PARAMS_BY_MODALITY.get(modality, BEST_PARAMS_BY_MODALITY['Combined'])
    
    print(f"  Running {N_FOLDS}-fold grouped CV...")
    cv_results = []
    fold_predictions = []
    cv_start = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        fold_start = time.time()
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ----------------------------------------------------------------------------
        # [TWEAK 2] STRICT PHASE 1 LEAKAGE FIX
        # Imputing and clipping inside the fold for Phase 1 (matches Phase 2)
        # ----------------------------------------------------------------------------
        train_median = X_train.median()
        lower_b = X_train.quantile(0.01)
        upper_b = X_train.quantile(0.99)
        
        X_train = X_train.fillna(train_median).clip(lower=lower_b, upper=upper_b, axis=1)
        X_test = X_test.fillna(train_median).clip(lower=lower_b, upper=upper_b, axis=1)
        
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        cv_results.append({'Fold': fold, 'Test_R2': test_r2, 'MAE': mae, 'RMSE': rmse, 'Test_N': len(test_idx)})
        
        fold_pred_df = pd.DataFrame({
            'Fold': fold, 'FIPS': groups[test_idx], 'Year': df.iloc[test_idx]['year'].values,
            'Actual': y_test, 'Predicted': y_pred_test, 'Residual': y_test - y_pred_test,
            'Abs_Error': np.abs(y_test - y_pred_test)
        })
        fold_predictions.append(fold_pred_df)
        print(f"    Fold {fold}: R²={test_r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f} ({time.time() - fold_start:.1f}s)")
    
    cv_time = time.time() - cv_start
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    cv_df = pd.DataFrame(cv_results)
    
    mean_r2, std_r2 = cv_df['Test_R2'].mean(), cv_df['Test_R2'].std()
    mean_mae, std_mae = cv_df['MAE'].mean(), cv_df['MAE'].std()
    mean_rmse, std_rmse = cv_df['RMSE'].mean(), cv_df['RMSE'].std()
    
    print(f"\n  ✓ CV Results ({cv_time:.1f}s): R² = {mean_r2:.4f} ± {std_r2:.4f} | MAE = {mean_mae:.2f} years | RMSE = {mean_rmse:.2f} years")
    
    summary_results.append({'Modality': modality, 'N_Features': X.shape[1], 'R2_Mean': mean_r2, 'R2_Std': std_r2, 'MAE_Mean': mean_mae, 'MAE_Std': std_mae, 'RMSE_Mean': mean_rmse, 'RMSE_Std': std_rmse})
    
    all_modalities_results[modality] = {'cv_results': cv_df, 'predictions': predictions_df, 'best_params': best_params, 'feature_cols': feature_cols}
    cv_df.to_csv(modality_output_dir / 'cv_results.csv', index=False)
    predictions_df.to_csv(modality_output_dir / 'predictions.csv', index=False)

# Phase 1 Combined Model
print(f"\n{'-'*60}")
print("TRAINING COMBINED MODEL (Phase 1 Logic)")
print(f"{'-'*60}")

cols_to_ignore = ['year', 'fips', 'location_name', 'MeanLifeExpectency']
cols_to_ignore.extend([c for c in df.columns if c.startswith('count_') or c.startswith('sum_') or c.startswith('area_')])
combined_feature_cols = [c for c in df.columns if c not in cols_to_ignore]

X_combined = df[combined_feature_cols].copy()
X_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

best_params_combined = BEST_PARAMS_BY_MODALITY['Combined']
cv_results_combined = []
fold_predictions_combined = []
cv_start_combined = time.time()

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_combined, y, groups), 1):
    fold_start = time.time()
    
    X_train, X_test = X_combined.iloc[train_idx].copy(), X_combined.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    
    # ----------------------------------------------------------------------------
    # STRICT PHASE 1 LEAKAGE FIX (Combined Model)
    # ----------------------------------------------------------------------------
    train_median = X_train.median()
    lower_b = X_train.quantile(0.01)
    upper_b = X_train.quantile(0.99)
    
    X_train = X_train.fillna(train_median).clip(lower=lower_b, upper=upper_b, axis=1)
    X_test = X_test.fillna(train_median).clip(lower=lower_b, upper=upper_b, axis=1)
    
    model = RandomForestRegressor(**best_params_combined)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    test_r2, mae, rmse = r2_score(y_test, y_pred_test), mean_absolute_error(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    cv_results_combined.append({'Fold': fold, 'Test_R2': test_r2, 'MAE': mae, 'RMSE': rmse})
    fold_pred_df = pd.DataFrame({'Fold': fold, 'FIPS': groups[test_idx], 'Year': df.iloc[test_idx]['year'].values, 'Actual': y_test, 'Predicted': y_pred_test, 'Residual': y_test - y_pred_test, 'Abs_Error': np.abs(y_test - y_pred_test)})
    fold_predictions_combined.append(fold_pred_df)
    
    print(f"  Fold {fold}: R²={test_r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f} ({time.time() - fold_start:.1f}s)")

predictions_combined_df = pd.concat(fold_predictions_combined, ignore_index=True)
cv_df_combined = pd.DataFrame(cv_results_combined)
mean_r2_combined, std_r2_combined = cv_df_combined['Test_R2'].mean(), cv_df_combined['Test_R2'].std()
mean_mae_combined, mean_rmse_combined = cv_df_combined['MAE'].mean(), cv_df_combined['RMSE'].mean()

summary_results.append({'Modality': 'Combined', 'N_Features': X_combined.shape[1], 'R2_Mean': mean_r2_combined, 'R2_Std': std_r2_combined, 'MAE_Mean': mean_mae_combined, 'MAE_Std': cv_df_combined['MAE'].std(), 'RMSE_Mean': mean_rmse_combined, 'RMSE_Std': cv_df_combined['RMSE'].std()})
all_modalities_results['Combined'] = {'cv_results': cv_df_combined, 'predictions': predictions_combined_df, 'best_params': best_params_combined, 'feature_cols': combined_feature_cols}

# Phase 1 Wrap up
summary_df = pd.DataFrame(summary_results).sort_values('R2_Mean', ascending=False).reset_index(drop=True)
display_df = summary_df.copy()
display_df['R2'] = (display_df['R2_Mean'].round(4).astype(str) + ' ± ' + display_df['R2_Std'].round(4).astype(str))
display_df['MAE (years)'] = (display_df['MAE_Mean'].round(2).astype(str) + ' ± ' + display_df['MAE_Std'].round(2).astype(str))
display_df['RMSE (years)'] = display_df['RMSE_Mean'].round(2).astype(str)

print("\nMASTER SUMMARY TABLE")
print(display_df[['Modality', 'N_Features', 'R2', 'MAE (years)', 'RMSE (years)']].to_string(index=False))

summary_df.to_csv(OUTPUT_DIR_PHASE1 / 'master_summary.csv', index=False)
display_df[['Modality', 'N_Features', 'R2', 'MAE (years)', 'RMSE (years)']].to_csv(OUTPUT_DIR_PHASE1 / 'master_summary_formatted.csv', index=False)

combined_dir = OUTPUT_DIR_PHASE1 / 'Combined'
combined_dir.mkdir(exist_ok=True)
cv_df_combined.to_csv(combined_dir / 'cv_results.csv', index=False)
predictions_combined_df.to_csv(combined_dir / 'predictions.csv', index=False)

with open(OUTPUT_DIR_PHASE1 / 'all_results.pkl', 'wb') as f:
    pickle.dump(all_modalities_results, f)


# ============================================================================
# MEMORY CLEAR BEFORE PHASE 2
# ============================================================================
del df
del X_combined
del all_modalities_results
gc.collect()


# ============================================================================
# PHASE 2: PRODUCTION ML ANALYSIS (SHAP & TABLES)
# ============================================================================
print("\n\n" + "="*80)
print("PHASE 2: PRODUCTION ML ANALYSIS (SHAP & TABLES)")
print("="*80)

# Refresh from master data
df = df_master.copy()

print("[STEP 1] Applying strict intra-county temporal imputation (bfill/ffill)...")
fill_start = time.time()
df = df.sort_values(by=['fips', 'year']).reset_index(drop=True)

# ----------------------------------------------------------------------------
# [TWEAK 3] FIXED TARGET EXCLUSION NAME
# Ensures 'MeanLifeExpectency' is ignored during bfill/ffill before renaming
# ----------------------------------------------------------------------------
cols_to_fill = [c for c in df.columns if c not in ['fips', 'year', 'location_name', 'MeanLifeExpectency']]
df[cols_to_fill] = df.groupby('fips')[cols_to_fill].bfill().ffill()
print(f"  ✓ Temporal imputation complete ({time.time()-fill_start:.1f}s)")

BEST_PARAMS = {
    'n_estimators': 2000, 'min_samples_split': 3, 'min_samples_leaf': 2,
    'max_samples': 0.9, 'max_features': 0.3, 'max_depth': 40,
    'bootstrap': True, 'random_state': 42, 'n_jobs': -1
}

PRUNING_THRESHOLD = 0.001 
SHAP_SAMPLE_SIZE = 5000  
SHAP_BACKGROUND_SIZE = 500

FEATURE_RENAME_MAP = {
    'ENG_TVI': 'Thermal Vegetation Index (TVI)',
    'ENG_Diurnal_Temp_Range': 'Diurnal Temp Range',
    'ENG_Night_Cooling_Eff': 'Nighttime Cooling Efficiency',
    'ENG_NDVI_LST_Divergence': 'NDVI-LST Divergence',
    'ENG_Ag_Greenness_Ratio': 'Ag Greenness Ratio',
    'ENG_Livestock_Heat_Exposure': 'Livestock Heat Exposure',
    'ENG_Impervious_Heat_Index': 'Impervious Surface Heat Index',
    'ENG_Forest_Heat_Buffer': 'Forest Heat Buffer Score',
    'ENG_Soil_Moisture_Deficit': 'Soil Moisture Deficit',
    'ENG_Soil_Moisture_Excess': 'Soil Moisture Excess',
    'ENG_Wetland_Flood_Risk': 'Wetland Flood Risk Index',
    'ENG_SAR_NDVI_Struct': 'SAR-NDVI Structural Index',
    'ENG_Water_Permanence': 'Water Permanence Index',
    'ENG_Elevation_Thermal_Mod': 'Elevation Thermal Mod',
    'ENG_Topo_Roughness_LST': 'Topographic Roughness x LST',
    'ENG_Livestock_Diversity': 'Livestock Species Diversity',
    'ENG_Crop_Diversity': 'Crop Diversity Index',
    'ENG_Seasonal_NDVI_Amp': 'Seasonal NDVI Amplitude',
    'ENG_Cross_Sensor_NDVI': 'Cross-Sensor NDVI Consistency',
    'ENG_Wet_Bulb_Proxy': 'Wet Bulb Temperature Proxy',
    'ENG_Blue_Green_Proximity': 'Blue-Green Proximity Index',
    'ENG_Urban_Stressor': 'High-Intensity Urban Stressor',
    'ENG_Industrial_Ag_Proxy': 'Industrial Ag Proxy',
    'ENG_Livestock_Pollution_Load': 'Livestock Pollution Load',
    'ENG_Landscape_Complexity': 'Landscape Complexity',
    'USDA_mean': 'Avg Crop Productivity Index',
    'USDA_stdDev': 'Crop Productivity Variance',
    'USDA_Cropland_USDA_Corn_pct': 'Corn %',
    'USDA_Cropland_USDA_Fallow_pct': 'Fallow Land %',
    'USDA_Cropland_USDA_Soybeans_pct': 'Soybeans %',
    'USDA_Cropland_USDA_Grassland_pct': 'Grassland/Pasture %',
    'USDA_Cropland_USDA_Sunflower_pct': 'Sunflower %',
    'USDA_Cropland_USDA_Alfalfa_pct': 'Alfalfa %',
    'USDA_Cropland_USDA_Spring Wheat_pct': 'Spring Wheat %',
    'USDA_Cropland_USDA_Barley_pct': 'Barley %',
    'USDA_Cropland_USDA_Oats_pct': 'Oats %',
    'USDA_Cropland_USDA_Winter Wheat_pct': 'Winter Wheat %',
    'USDA_Cropland_USDA_Cotton_pct': 'Cotton %',
    'USDA_Cropland_USDA_Rice_pct': 'Rice %',
    'USDA_Cropland_USDA_Other Hay_pct': 'Other Hay/Non-Alfalfa %',
    'USDA_Cropland_USDA_Sweet Corn_pct': 'Sweet Corn %',
    'USDA_Cropland_USDA_Tobacco_pct': 'Tobacco %',
    'USDA_Cropland_USDA_Sorghum_pct': 'Sorghum %',
    'USDA_Cropland_USDA_Peanuts_pct': 'Peanuts %',
    'USDA_Cropland_USDA_Shrubland_pct': 'Shrubland %',
    'USDA_Cropland_USDA_Open Water_pct': 'Open Water (USDA) %',
    'USDA_Cropland_USDA_Other Crops_pct': 'Other Crops %',
    'USDA_Cropland_USDA_Forest_Evergreen_pct': 'Evergreen Forest %',
    'USDA_Cropland_USDA_Forest_Deciduous_pct': 'Deciduous Forest %',
    'USDA_Cropland_USDA_Forest_Mixed_pct': 'Mixed Forest %',
    'USDA_Cropland_USDA_Wetlands_Herbaceous_pct': 'Herbaceous Wetlands %',
    'USDA_Cropland_USDA_Wetlands_Woody_pct': 'Woody Wetlands %',
    'USDA_Cropland_USDA_Dev_Open_pct': 'Developed (Open Space) %',
    'USDA_Cropland_USDA_Dev_Low_pct': 'Developed (Low Intensity) %',
    'USDA_Cropland_USDA_Dev_Med_pct': 'Developed (Med Intensity) %',
    'USDA_Cropland_USDA_Dev_High_pct': 'Developed (High Intensity) %',
    'JRC_mean': 'Avg Water Surface Area',
    'JRC_stdDev': 'Water Surface Variance',
    'JRC_Water_JRC_NotWater_pct': 'Non-Water Area %',
    'JRC_Water_JRC_Permanent_pct': 'Permanent Water %',
    'JRC_Water_JRC_Seasonal_pct': 'Seasonal Water %',
    'JRC_Water_JRC_Class_0_pct': 'No Water Data %',
    'DEM_mean': 'Elevation Mean (m)',
    'DEM_p10': 'Elevation (10th Percentile)',
    'DEM_p25': 'Elevation (25th Percentile)',
    'DEM_p50': 'Elevation (Median)',
    'DEM_p75': 'Elevation (75th Percentile)',
    'DEM_p90': 'Elevation (90th Percentile)',
    'DEM_stdDev': 'Topographic Ruggedness',
    'Soil_mean': 'Soil Moisture Mean',
    'Soil_stdDev': 'Soil Moisture Variance',
    'Soil_p10': 'Soil Moisture (10th Percentile)',
    'Soil_p25': 'Soil Moisture (25th Percentile)',
    'Soil_p50': 'Soil Moisture (Median)',
    'Soil_p75': 'Soil Moisture (75th Percentile)',
    'Soil_p90': 'Soil Moisture (90th Percentile)',
    'LST_Day_1km_mean': 'Daytime Surface Temp (Mean)',
    'LST_Day_1km_p10': 'Daytime Temp (10th Percentile)',
    'LST_Day_1km_p25': 'Daytime Temp (25th Percentile)',
    'LST_Day_1km_p50': 'Daytime Temp (Median)',
    'LST_Day_1km_p75': 'Daytime Temp (75th Percentile)',
    'LST_Day_1km_p90': 'Daytime Temp (90th Percentile)',
    'LST_Day_1km_stdDev': 'Daytime Temp Variance',
    'LST_Night_1km_mean': 'Nighttime Surface Temp (Mean)',
    'LST_Night_1km_p10': 'Nighttime Temp (10th Percentile)',
    'LST_Night_1km_p25': 'Nighttime Temp (25th Percentile)',
    'LST_Night_1km_p50': 'Nighttime Temp (Median)',
    'LST_Night_1km_p75': 'Nighttime Temp (75th Percentile)',
    'LST_Night_1km_p90': 'Nighttime Temp (90th Percentile)',
    'LST_Night_1km_stdDev': 'Nighttime Temp Variance',
    'NDVI_EVI_mean': 'Vegetation Health Index (Mean)',
    'NDVI_EVI_stdDev': 'Vegetation Health Variance',
    'NDVI_EVI_p10': 'Vegetation Health (10th Percentile)',
    'NDVI_EVI_p25': 'Vegetation Health (25th Percentile)',
    'NDVI_EVI_p50': 'Vegetation Health (Median)',
    'NDVI_EVI_p75': 'Vegetation Health (75th Percentile)',
    'NDVI_EVI_p90': 'Vegetation Health (90th Percentile)',
    'NDVI_mean': 'NDVI Mean',
    'NDVI_p10': 'NDVI (10th Percentile)',
    'NDVI_p25': 'NDVI (25th Percentile)',
    'NDVI_p50': 'NDVI (Median)',
    'NDVI_p75': 'NDVI (75th Percentile)',
    'NDVI_p90': 'NDVI (90th Percentile)',
    'NDVI_stdDev': 'NDVI Variance',
    'Landsat_BSI_mean': 'Bare Soil Index (Mean)',
    'Landsat_BSI_p10': 'BSI (10th Percentile)',
    'Landsat_BSI_p25': 'BSI (25th Percentile)',
    'Landsat_BSI_p50': 'BSI (Median)',
    'Landsat_BSI_p75': 'BSI (75th Percentile)',
    'Landsat_BSI_p90': 'BSI (90th Percentile)',
    'Landsat_BSI_stdDev': 'BSI Variance',
    'Landsat_EVI_mean': 'Landsat EVI (Mean)',
    'Landsat_EVI_p10': 'Landsat EVI (10th Percentile)',
    'Landsat_EVI_p25': 'Landsat EVI (25th Percentile)',
    'Landsat_EVI_p50': 'Landsat EVI (Median)',
    'Landsat_EVI_p75': 'Landsat EVI (75th Percentile)',
    'Landsat_EVI_p90': 'Landsat EVI (90th Percentile)',
    'Landsat_EVI_stdDev': 'Landsat EVI Variance',
    'Landsat_NDMI_mean': 'Landsat NDMI (Mean)',
    'Landsat_NDMI_p10': 'Landsat NDMI (10th Percentile)',
    'Landsat_NDMI_p25': 'Landsat NDMI (25th Percentile)',
    'Landsat_NDMI_p50': 'Landsat NDMI (Median)',
    'Landsat_NDMI_p75': 'Landsat NDMI (75th Percentile)',
    'Landsat_NDMI_p90': 'Landsat NDMI (90th Percentile)',
    'Landsat_NDMI_stdDev': 'Landsat NDMI Variance',
    'Landsat_NDVI_mean': 'Landsat NDVI (Mean)',
    'Landsat_NDVI_p10': 'Landsat NDVI (10th Percentile)',
    'Landsat_NDVI_p25': 'Landsat NDVI (25th Percentile)',
    'Landsat_NDVI_p50': 'Landsat NDVI (Median)',
    'Landsat_NDVI_p75': 'Landsat NDVI (75th Percentile)',
    'Landsat_NDVI_p90': 'Landsat NDVI (90th Percentile)',
    'Landsat_NDVI_stdDev': 'Landsat NDVI Variance',
    'Landsat_NDWI_mean': 'Landsat NDWI (Mean)',
    'Landsat_NDWI_p10': 'Landsat NDWI (10th Percentile)',
    'Landsat_NDWI_p25': 'Landsat NDWI (25th Percentile)',
    'Landsat_NDWI_p50': 'Landsat NDWI (Median)',
    'Landsat_NDWI_p75': 'Landsat NDWI (75th Percentile)',
    'Landsat_NDWI_p90': 'Landsat NDWI (90th Percentile)',
    'Landsat_NDWI_stdDev': 'Landsat NDWI Variance',
    'Landsat_SAVI_mean': 'Landsat SAVI (Mean)',
    'Landsat_SAVI_p10': 'Landsat SAVI (10th Percentile)',
    'Landsat_SAVI_p25': 'Landsat SAVI (25th Percentile)',
    'Landsat_SAVI_p50': 'Landsat SAVI (Median)',
    'Landsat_SAVI_p75': 'Landsat SAVI (75th Percentile)',
    'Landsat_SAVI_p90': 'Landsat SAVI (90th Percentile)',
    'Landsat_SAVI_stdDev': 'Landsat SAVI Variance',
    'Landsat_SR_B2_mean': 'Landsat Blue (Mean)',
    'Landsat_SR_B2_p10': 'Landsat Blue (10th Percentile)',
    'Landsat_SR_B2_p25': 'Landsat Blue (25th Percentile)',
    'Landsat_SR_B2_p50': 'Landsat Blue (Median)',
    'Landsat_SR_B2_p75': 'Landsat Blue (75th Percentile)',
    'Landsat_SR_B2_p90': 'Landsat Blue (90th Percentile)',
    'Landsat_SR_B2_stdDev': 'Landsat Blue Variance',
    'Landsat_SR_B3_mean': 'Landsat Green (Mean)',
    'Landsat_SR_B3_p10': 'Landsat Green (10th Percentile)',
    'Landsat_SR_B3_p25': 'Landsat Green (25th Percentile)',
    'Landsat_SR_B3_p50': 'Landsat Green (Median)',
    'Landsat_SR_B3_p75': 'Landsat Green (75th Percentile)',
    'Landsat_SR_B3_p90': 'Landsat Green (90th Percentile)',
    'Landsat_SR_B3_stdDev': 'Landsat Green Variance',
    'Landsat_SR_B4_mean': 'Landsat Red (Mean)',
    'Landsat_SR_B4_p10': 'Landsat Red (10th Percentile)',
    'Landsat_SR_B4_p25': 'Landsat Red (25th Percentile)',
    'Landsat_SR_B4_p50': 'Landsat Red (Median)',
    'Landsat_SR_B4_p75': 'Landsat Red (75th Percentile)',
    'Landsat_SR_B4_p90': 'Landsat Red (90th Percentile)',
    'Landsat_SR_B4_stdDev': 'Landsat Red Variance',
    'Landsat_SR_B5_mean': 'Landsat NIR (Mean)',
    'Landsat_SR_B5_p10': 'Landsat NIR (10th Percentile)',
    'Landsat_SR_B5_p25': 'Landsat NIR (25th Percentile)',
    'Landsat_SR_B5_p50': 'Landsat NIR (Median)',
    'Landsat_SR_B5_p75': 'Landsat NIR (75th Percentile)',
    'Landsat_SR_B5_p90': 'Landsat NIR (90th Percentile)',
    'Landsat_SR_B5_stdDev': 'Landsat NIR Variance',
    'Landsat_SR_B6_mean': 'Landsat SWIR1 (Mean)',
    'Landsat_SR_B6_p10': 'Landsat SWIR1 (10th Percentile)',
    'Landsat_SR_B6_p25': 'Landsat SWIR1 (25th Percentile)',
    'Landsat_SR_B6_p50': 'Landsat SWIR1 (Median)',
    'Landsat_SR_B6_p75': 'Landsat SWIR1 (75th Percentile)',
    'Landsat_SR_B6_p90': 'Landsat SWIR1 (90th Percentile)',
    'Landsat_SR_B6_stdDev': 'Landsat SWIR1 Variance',
    'Landsat_SR_B7_mean': 'Landsat SWIR2 (Mean)',
    'Landsat_SR_B7_p10': 'Landsat SWIR2 (10th Percentile)',
    'Landsat_SR_B7_p25': 'Landsat SWIR2 (25th Percentile)',
    'Landsat_SR_B7_p50': 'Landsat SWIR2 (Median)',
    'Landsat_SR_B7_p75': 'Landsat SWIR2 (75th Percentile)',
    'Landsat_SR_B7_p90': 'Landsat SWIR2 (90th Percentile)',
    'Landsat_SR_B7_stdDev': 'Landsat SWIR2 Variance',
    'S1_VH_mean': 'SAR VH (Mean)',
    'S1_VH_p10': 'SAR VH (10th Percentile)',
    'S1_VH_p25': 'SAR VH (25th Percentile)',
    'S1_VH_p50': 'SAR VH (Median)',
    'S1_VH_p75': 'SAR VH (75th Percentile)',
    'S1_VH_p90': 'SAR VH (90th Percentile)',
    'S1_VH_stdDev': 'SAR VH Variance',
    'S1_VV_mean': 'SAR VV (Mean)',
    'S1_VV_p10': 'SAR VV (10th Percentile)',
    'S1_VV_p25': 'SAR VV (25th Percentile)',
    'S1_VV_p50': 'SAR VV (Median)',
    'S1_VV_p75': 'SAR VV (75th Percentile)',
    'S1_VV_p90': 'SAR VV (90th Percentile)',
    'S1_VV_stdDev': 'SAR VV Variance',
    'S1_VV_asm_mean': 'SAR Texture ASM (Mean)',
    'S1_VV_asm_p10': 'SAR Texture ASM (10th Percentile)',
    'S1_VV_asm_p25': 'SAR Texture ASM (25th Percentile)',
    'S1_VV_asm_p50': 'SAR Texture ASM (Median)',
    'S1_VV_asm_p75': 'SAR Texture ASM (75th Percentile)',
    'S1_VV_asm_p90': 'SAR Texture ASM (90th Percentile)',
    'S1_VV_asm_stdDev': 'SAR Texture ASM Variance',
    'S1_VV_contrast_mean': 'SAR Texture Contrast (Mean)',
    'S1_VV_contrast_p10': 'SAR Texture Contrast (10th Percentile)',
    'S1_VV_contrast_p25': 'SAR Texture Contrast (25th Percentile)',
    'S1_VV_contrast_p50': 'SAR Texture Contrast (Median)',
    'S1_VV_contrast_p75': 'SAR Texture Contrast (75th Percentile)',
    'S1_VV_contrast_p90': 'SAR Texture Contrast (90th Percentile)',
    'S1_VV_contrast_stdDev': 'SAR Texture Contrast Variance',
    'S1_VV_corr_mean': 'SAR Texture Correlation (Mean)',
    'S1_VV_corr_p10': 'SAR Texture Corr (10th Percentile)',
    'S1_VV_corr_p25': 'SAR Texture Corr (25th Percentile)',
    'S1_VV_corr_p50': 'SAR Texture Corr (Median)',
    'S1_VV_corr_p75': 'SAR Texture Corr (75th Percentile)',
    'S1_VV_corr_p90': 'SAR Texture Corr (90th Percentile)',
    'S1_VV_corr_stdDev': 'SAR Texture Corr Variance',
    'S1_VV_ent_mean': 'SAR Texture Entropy (Mean)',
    'S1_VV_ent_p10': 'SAR Texture Entropy (10th Percentile)',
    'S1_VV_ent_p25': 'SAR Texture Entropy (25th Percentile)',
    'S1_VV_ent_p50': 'SAR Texture Entropy (Median)',
    'S1_VV_ent_p75': 'SAR Texture Entropy (75th Percentile)',
    'S1_VV_ent_p90': 'SAR Texture Entropy (90th Percentile)',
    'S1_VV_ent_stdDev': 'SAR Texture Entropy Variance',
    'S2_B2_mean': 'Sentinel-2 Blue (Mean)',
    'S2_B2_p10': 'S2 Blue (10th Percentile)',
    'S2_B2_p25': 'S2 Blue (25th Percentile)',
    'S2_B2_p50': 'S2 Blue (Median)',
    'S2_B2_p75': 'S2 Blue (75th Percentile)',
    'S2_B2_p90': 'S2 Blue (90th Percentile)',
    'S2_B2_stdDev': 'S2 Blue Variance',
    'S2_B3_mean': 'Sentinel-2 Green (Mean)',
    'S2_B3_p10': 'S2 Green (10th Percentile)',
    'S2_B3_p25': 'S2 Green (25th Percentile)',
    'S2_B3_p50': 'S2 Green (Median)',
    'S2_B3_p75': 'S2 Green (75th Percentile)',
    'S2_B3_p90': 'S2 Green (90th Percentile)',
    'S2_B3_stdDev': 'S2 Green Variance',
    'S2_B4_mean': 'Sentinel-2 Red (Mean)',
    'S2_B4_p10': 'S2 Red (10th Percentile)',
    'S2_B4_p25': 'S2 Red (25th Percentile)',
    'S2_B4_p50': 'S2 Red (Median)',
    'S2_B4_p75': 'S2 Red (75th Percentile)',
    'S2_B4_p90': 'S2 Red (90th Percentile)',
    'S2_B4_stdDev': 'S2 Red Variance',
    'S2_B8_mean': 'Sentinel-2 NIR (Mean)',
    'S2_B8_p10': 'S2 NIR (10th Percentile)',
    'S2_B8_p25': 'S2 NIR (25th Percentile)',
    'S2_B8_p50': 'S2 NIR (Median)',
    'S2_B8_p75': 'S2 NIR (75th Percentile)',
    'S2_B8_p90': 'S2 NIR (90th Percentile)',
    'S2_B8_stdDev': 'S2 NIR Variance',
    'S2_B11_mean': 'Sentinel-2 SWIR1 (Mean)',
    'S2_B11_p10': 'S2 SWIR1 (10th Percentile)',
    'S2_B11_p25': 'S2 SWIR1 (25th Percentile)',
    'S2_B11_p50': 'S2 SWIR1 (Median)',
    'S2_B11_p75': 'S2 SWIR1 (75th Percentile)',
    'S2_B11_p90': 'S2 SWIR1 (90th Percentile)',
    'S2_B11_stdDev': 'S2 SWIR1 Variance',
    'S2_B12_mean': 'Sentinel-2 SWIR2 (Mean)',
    'S2_B12_p10': 'S2 SWIR2 (10th Percentile)',
    'S2_B12_p25': 'S2 SWIR2 (25th Percentile)',
    'S2_B12_p50': 'S2 SWIR2 (Median)',
    'S2_B12_p75': 'S2 SWIR2 (75th Percentile)',
    'S2_B12_p90': 'S2 SWIR2 (90th Percentile)',
    'S2_B12_stdDev': 'S2 SWIR2 Variance',
    'S2_BSI_mean': 'S2 Bare Soil Index (Mean)',
    'S2_BSI_p10': 'S2 BSI (10th Percentile)',
    'S2_BSI_p25': 'S2 BSI (25th Percentile)',
    'S2_BSI_p50': 'S2 BSI (Median)',
    'S2_BSI_p75': 'S2 BSI (75th Percentile)',
    'S2_BSI_p90': 'S2 BSI (90th Percentile)',
    'S2_BSI_stdDev': 'S2 BSI Variance',
    'S2_EVI_mean': 'S2 EVI (Mean)',
    'S2_EVI_p10': 'S2 EVI (10th Percentile)',
    'S2_EVI_p25': 'S2 EVI (25th Percentile)',
    'S2_EVI_p50': 'S2 EVI (Median)',
    'S2_EVI_p75': 'S2 EVI (75th Percentile)',
    'S2_EVI_p90': 'S2 EVI (90th Percentile)',
    'S2_EVI_stdDev': 'S2 EVI Variance',
    'S2_NDMI_mean': 'S2 NDMI (Mean)',
    'S2_NDMI_p10': 'S2 NDMI (10th Percentile)',
    'S2_NDMI_p25': 'S2 NDMI (25th Percentile)',
    'S2_NDMI_p50': 'S2 NDMI (Median)',
    'S2_NDMI_p75': 'S2 NDMI (75th Percentile)',
    'S2_NDMI_p90': 'S2 NDMI (90th Percentile)',
    'S2_NDMI_stdDev': 'S2 NDMI Variance',
    'S2_NDVI_mean': 'S2 NDVI (Mean)',
    'S2_NDVI_p10': 'S2 NDVI (10th Percentile)',
    'S2_NDVI_p25': 'S2 NDVI (25th Percentile)',
    'S2_NDVI_p50': 'S2 NDVI (Median)',
    'S2_NDVI_p75': 'S2 NDVI (75th Percentile)',
    'S2_NDVI_p90': 'S2 NDVI (90th Percentile)',
    'S2_NDVI_stdDev': 'S2 NDVI Variance',
    'S2_NDWI_mean': 'S2 NDWI (Mean)',
    'S2_NDWI_p10': 'S2 NDWI (10th Percentile)',
    'S2_NDWI_p25': 'S2 NDWI (25th Percentile)',
    'S2_NDWI_p50': 'S2 NDWI (Median)',
    'S2_NDWI_p75': 'S2 NDWI (75th Percentile)',
    'S2_NDWI_p90': 'S2 NDWI (90th Percentile)',
    'S2_NDWI_stdDev': 'S2 NDWI Variance',
    'S2_SAVI_mean': 'S2 SAVI (Mean)',
    'S2_SAVI_p10': 'S2 SAVI (10th Percentile)',
    'S2_SAVI_p25': 'S2 SAVI (25th Percentile)',
    'S2_SAVI_p50': 'S2 SAVI (Median)',
    'S2_SAVI_p75': 'S2 SAVI (75th Percentile)',
    'S2_SAVI_p90': 'S2 SAVI (90th Percentile)',
    'S2_SAVI_stdDev': 'S2 SAVI Variance',
    'mean_buffalo': 'Buffalo Density',
    'mean_cattle': 'Cattle Density',
    'mean_chicken': 'Chicken Density',
    'mean_duck': 'Duck Density',
    'mean_goat': 'Goat Density',
    'mean_horse': 'Horse Density',
    'mean_pig': 'Pig Density',
    'mean_sheep': 'Sheep Density',
    'head_count_cattle': 'Cattle Head Count',
    'head_count_chicken': 'Chicken Head Count',
    'head_count_goat': 'Goat Head Count',
    'head_count_horse': 'Horse Head Count',
    'head_count_sheep': 'Sheep Head Count',
    'std_dev_cattle': 'Cattle Density StdDev',
    'std_dev_chicken': 'Chicken Density StdDev',
    'std_dev_duck': 'Duck Density StdDev',
    'std_dev_goat': 'Goat Density StdDev',
    'std_dev_horse': 'Horse Density StdDev',
    'std_dev_pig': 'Pig Density StdDev',
    'std_dev_sheep': 'Sheep Density StdDev',
    'std_error_cattle': 'Cattle StdError',
    'std_error_goat': 'Goat StdError',
    'std_error_sheep': 'Sheep StdError',
    'MeanLifeExpectency': 'Life Expectancy (Years)'
}

for col in df.columns:
    if "USDA_Cropland_USDA_Class_" in col and col not in FEATURE_RENAME_MAP:
        class_num = col.split('_')[-2]
        FEATURE_RENAME_MAP[col] = f"Cropland Class {class_num} %"

df = df.rename(columns=FEATURE_RENAME_MAP)
print(f"  ✓ Renamed features for interpretability")

cols_to_drop = ['Life Expectancy (Years)', 'year', 'fips', 'location_name']
cols_to_drop.extend([c for c in df.columns if c.startswith('count_') or c.startswith('sum_') or c.startswith('area_')])

X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
y = df['Life Expectancy (Years)']
groups = df['fips']

print(f"  Pre-pruning dataset: {len(X_raw):,} observations × {X_raw.shape[1]} features")

print("\n[STEP 2] Pruning zero-impact features...")
pruning_start = time.time()

# For pruning only, we do a quick global fill for any completely barren counties
# to allow the RF to run. The strict rigorous filling happens in the CV loop.
X_raw_prune = X_raw.fillna(X_raw.median())

pruning_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
pruning_model.fit(X_raw_prune, y)

importances = pruning_model.feature_importances_
features_to_keep = X_raw.columns[importances >= PRUNING_THRESHOLD]
features_dropped = X_raw.columns[importances < PRUNING_THRESHOLD]

X = X_raw[features_to_keep].copy()

print(f"  ✓ Dropped {len(features_dropped)} features with near-zero impact (< {PRUNING_THRESHOLD})")
print(f"  ✓ Kept {len(X.columns)} strong features for Final Model and SHAP")
print(f"  Pruning time: {time.time()-pruning_start:.1f}s")

print("\n[STEP 3] Running 5-fold grouped cross-validation...")
cv_start = time.time()
gkf = GroupKFold(n_splits=5)
cv_results = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    fold_start = time.time()
    
    # ----------------------------------------------------------------------------
    # STRICT ISOLATION OF IMPUTATION & CLIPPING
    # ----------------------------------------------------------------------------
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Calculate stats exclusively from training data
    train_median = X_train.median()
    lower_bound = X_train.quantile(0.001)
    upper_bound = X_train.quantile(0.999)
    
    # Apply to Train and Test independently
    X_train = X_train.fillna(train_median).clip(lower=lower_bound, upper=upper_bound, axis=1)
    X_test = X_test.fillna(train_median).clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    # Model Training
    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)
    
    test_preds = model.predict(X_test)
    test_r2, test_mae, test_rmse = r2_score(y_test, test_preds), mean_absolute_error(y_test, test_preds), np.sqrt(mean_squared_error(y_test, test_preds))
    test_mse = mean_squared_error(y_test, test_preds)
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
    
    cv_results.append({'Fold': fold, 'Test_R2': test_r2, 'MAE': test_mae, 'RMSE': test_rmse, 'MSE': test_mse, 'MAPE': test_mape})
    fold_preds = pd.DataFrame({'fold': fold, 'fips': groups.iloc[test_idx].values, 'year': df.iloc[test_idx]['year'].values, 'actual': y_test.values, 'predicted': test_preds, 'residual': y_test.values - test_preds, 'abs_error': np.abs(y_test.values - test_preds)})
    all_predictions.append(fold_preds)
    print(f"  Fold {fold}/5: R²={test_r2:.3f}, MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, MSE={test_mse:.2f}, MAPE={test_mape:.2f}% ({time.time()-fold_start:.1f}s)")

predictions_df = pd.concat(all_predictions, ignore_index=True)
print(f"  Cross-validation complete: {time.time()-cv_start:.1f}s")

print("\n[STEP 4] Training Final Model & Computing SHAP...")
final_start = time.time()

# ----------------------------------------------------------------------------
# FINAL MODEL PREP
# ----------------------------------------------------------------------------
final_median = X.median()
final_lower = X.quantile(0.001)
final_upper = X.quantile(0.999)
X_final = X.fillna(final_median).clip(lower=final_lower, upper=final_upper, axis=1)

final_model = RandomForestRegressor(**BEST_PARAMS)
final_model.fit(X_final, y)

X_shap_sample = X_final.sample(n=min(SHAP_SAMPLE_SIZE, len(X_final)), random_state=42)
explainer = shap.TreeExplainer(final_model, data=X_final.sample(n=min(SHAP_BACKGROUND_SIZE, len(X_final)), random_state=42), feature_perturbation='interventional')

def calculate_shap_batch(batch_X, explainer_obj):
    return explainer_obj.shap_values(batch_X, check_additivity=False)

n_chunks = 5
X_batches = np.array_split(X_shap_sample, n_chunks)
print(f"  Running parallel SHAP on pruned feature space ({len(X_final.columns)} features)...")
results_shap = Parallel(n_jobs=-1)(delayed(calculate_shap_batch)(batch, explainer) for batch in X_batches)
shap_values = np.vstack(results_shap)

shap_importance = pd.DataFrame({'Feature': X_final.columns, 'Mean_|SHAP|': np.abs(shap_values).mean(0), 'RF_Importance': final_model.feature_importances_}).sort_values('Mean_|SHAP|', ascending=False)
print(f"  SHAP analysis complete: {time.time()-final_start:.1f}s")
print("\n[STEP 5] Generating publication tables...")

table1 = pd.DataFrame(cv_results)
table1_summary = table1[['Test_R2', 'MAE', 'RMSE', 'MSE', 'MAPE']].agg(['mean', 'std'])
table1 = pd.concat([table1, pd.DataFrame([{
    'Fold': 'Mean ± SD', 'Test_R2': f"{table1_summary.loc['mean', 'Test_R2']:.3f} ± {table1_summary.loc['std', 'Test_R2']:.3f}", 'MAE': f"{table1_summary.loc['mean', 'MAE']:.2f} ± {table1_summary.loc['std', 'MAE']:.2f}", 'RMSE': f"{table1_summary.loc['mean', 'RMSE']:.2f} ± {table1_summary.loc['std', 'RMSE']:.2f}", 'MSE': f"{table1_summary.loc['mean', 'MSE']:.2f} ± {table1_summary.loc['std', 'MSE']:.2f}", 'MAPE': f"{table1_summary.loc['mean', 'MAPE']:.2f} ± {table1_summary.loc['std', 'MAPE']:.2f}"
}])])

# ----------------------------------------------------------------------------
# [TWEAK 4] EXPANDED LE BRACKET BINS
# Extending the maximum boundary to catch 92.5+ values
# ----------------------------------------------------------------------------
county_metrics = predictions_df.groupby('fips').agg({'actual': 'mean', 'abs_error': 'mean', 'residual': lambda x: np.sqrt(np.mean(x**2))}).reset_index()
county_metrics['LE_Bracket'] = pd.cut(county_metrics['actual'], bins=[60, 70, 74, 76.5, 79, 93], labels=['60-70', '70-74', '74-76.5', '76.5-79', '79-93'])
table2 = county_metrics.groupby('LE_Bracket', observed=True).agg({'fips': 'count', 'actual': ['mean', 'std'], 'abs_error': 'mean', 'residual': 'mean'}).round(2)
table2.columns = ['Unique_Counties', 'Mean_LE', 'Std_LE', 'MAE', 'RMSE']

table3 = predictions_df.groupby('year').agg({'fips': 'nunique', 'abs_error': ['mean', 'std'], 'actual': lambda x: r2_score(x, predictions_df.loc[x.index, 'predicted']), 'residual': lambda x: np.sqrt(mean_squared_error(predictions_df.loc[x.index, 'actual'], predictions_df.loc[x.index, 'predicted']))}).round(3)
table3.columns = ['Counties', 'MAE', 'MAE_Std', 'R2', 'RMSE']

table4 = shap_importance.head(20).copy()
table4['Rank'] = range(1, 21)
table4 = table4[['Rank', 'Feature', 'Mean_|SHAP|', 'RF_Importance']]

predictions_df['state_fips'] = predictions_df['fips'].str[:2]
table5 = predictions_df.groupby('state_fips').agg({'fips': 'nunique', 'abs_error': ['mean', 'std', 'min', 'max'], 'residual': lambda x: np.sqrt(mean_squared_error(predictions_df.loc[x.index, 'actual'], predictions_df.loc[x.index, 'predicted']))}).round(2)
table5.columns = ['Counties', 'MAE', 'MAE_Std', 'Min_Error', 'Max_Error', 'RMSE']
table5 = table5.sort_values('MAE')

residuals = predictions_df['residual'].values
table6 = pd.DataFrame({'Statistic': ['Mean Residual', 'Std Residual', 'Skewness', 'Kurtosis', 'Shapiro-Wilk W', 'Shapiro-Wilk p-value', 'Min Residual', 'Max Residual', '95% CI Lower', '95% CI Upper'], 'Value': [residuals.mean(), residuals.std(), stats.skew(residuals), stats.kurtosis(residuals), stats.shapiro(residuals[:5000])[0], stats.shapiro(residuals[:5000])[1], residuals.min(), residuals.max(), np.percentile(residuals, 2.5), np.percentile(residuals, 97.5)]}).round(4)

county_performance = predictions_df.groupby('fips')['abs_error'].mean().sort_values()
table7_best = county_performance.head(10).reset_index()
table7_best.columns = ['FIPS', 'Mean_Absolute_Error']
table7_worst = county_performance.tail(10).reset_index()
table7_worst.columns = ['FIPS', 'Mean_Absolute_Error']

tables = {'table1_cv_performance': table1, 'table2_le_bracket': table2, 'table3_temporal': table3, 'table4_feature_importance': table4, 'table5_regional': table5, 'table6_residual_stats': table6, 'table7a_best_counties': table7_best, 'table7b_worst_counties': table7_worst}

print("\n[STEP 6] Saving results...")
results = {'metadata': {'date': datetime.now().isoformat(), 'n_observations': len(df), 'n_counties': groups.nunique(), 'n_features': X.shape[1], 'year_range': f"{df['year'].min()}-{df['year'].max()}", 'best_params': BEST_PARAMS}, 'cv_results': cv_results, 'predictions': predictions_df, 'final_model': final_model, 'feature_names': X.columns.tolist(), 'shap_values': shap_values, 'shap_sample': X_shap_sample, 'shap_importance': shap_importance, 'tables': tables}

with open(OUTPUT_DIR_PHASE2 / 'ml_results.pkl', 'wb') as f:
    pickle.dump(results, f)

for name, table in tables.items():
    table.to_csv(OUTPUT_DIR_PHASE2 / f'{name}.csv')

predictions_df.to_csv(OUTPUT_DIR_PHASE2 / 'predictions_all_folds.csv', index=False)
shap_importance.to_csv(OUTPUT_DIR_PHASE2 / 'feature_importances.csv', index=False)

print("\n" + "="*80)
print("ALL PIPELINES COMPLETE")
print("="*80)
print(f"Total Combined Output saved to:")
print(f"  - Phase 1: {OUTPUT_DIR_PHASE1.absolute()}")
print(f"  - Phase 2: {OUTPUT_DIR_PHASE2.absolute()}")
