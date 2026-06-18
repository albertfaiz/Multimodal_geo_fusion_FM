import os
import sys
from pathlib import Path

print("="*60)
print("🚀 HPC PRE-FLIGHT AUDIT")
print("="*60)

# ---------------------------------------------------------
# 1. DEPENDENCY AUDIT (Would have caught the KNN error)
# ---------------------------------------------------------
print("\n[1] Checking ML & Spatial Dependencies...")
required_modules = ['pandas', 'numpy', 'sklearn', 'scipy', 'geopandas', 'xgboost', 'lightgbm', 'statsmodels']
missing_modules = []

for mod in required_modules:
    try:
        __import__(mod)
        print(f"  ✅ {mod:12} installed")
    except ImportError:
        missing_modules.append(mod)
        print(f"  ❌ {mod:12} MISSING")

if missing_modules:
    print(f"\n🛑 FATAL: Missing modules. Run: conda install {' '.join(missing_modules)}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. FILE & PATH AUDIT (Would have caught the missing .pkl)
# ---------------------------------------------------------
print("\n[2] Checking Required Files...")
required_files = [
    "full_clean_engineered_dataset_with_LE.csv",
    "ntl_county_year_raw.csv",
    "pm25_county_year.csv",
    "ml_results.pkl",
    "cb_2018_us_county_500k/cb_2018_us_county_500k.shp" # Checks inside the folder
]

missing_files = False
for file_path in required_files:
    if Path(file_path).exists():
        print(f"  ✅ Found: {file_path}")
    else:
        print(f"  ❌ MISSING: {file_path}")
        missing_files = True

if missing_files:
    print("\n🛑 FATAL: Required files are missing. Pipeline will crash.")
    sys.exit(1)

# ---------------------------------------------------------
# 3. DATA TYPE AUDIT (Would have caught the merge ValueError)
# ---------------------------------------------------------
print("\n[3] Checking FIPS Column Compatibility (The Merge Test)...")
import pandas as pd

try:
    df_master = pd.read_csv("full_clean_engineered_dataset_with_LE.csv", nrows=5)
    df_ntl = pd.read_csv("ntl_county_year_raw.csv", nrows=5)
    
    # Check if FIPS exists in both
    master_fips_col = next((c for c in df_master.columns if c.lower() in ['fips', 'geoid']), None)
    ntl_fips_col = next((c for c in df_ntl.columns if c.lower() in ['fips', 'geoid']), None)
    
    type_mismatch = False
    
    if master_fips_col and ntl_fips_col:
        master_type = df_master[master_fips_col].dtype
        ntl_type = df_ntl[ntl_fips_col].dtype
        print(f"  ℹ️  Master '{master_fips_col}' type: {master_type}")
        print(f"  ℹ️  NTL '{ntl_fips_col}' type: {ntl_type}")
        
        if master_type != ntl_type:
            print("  ⚠️ WARNING: FIPS type mismatch detected! Your merge logic must coerce these to strings first (which we fixed!).")
    else:
        print("  ❌ Could not find FIPS columns to check.")
except Exception as e:
    print(f"  ❌ Failed to read CSVs for test: {e}")

print("\n" + "="*60)
print("✅ PRE-FLIGHT COMPLETE. IF NO FATAL ERRORS, YOU ARE GREEN TO LAUNCH.")
print("="*60)
