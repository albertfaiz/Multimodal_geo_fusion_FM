import importlib.util
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# HPC CONFIGURATION (NO GEE IMPORTS)
# ============================================================================
HERE = Path("./")
ML_DIAG_FILE = HERE / "01_reviewer_diagnostics_ml.py"
DATA_FILE = "./full_clean_engineered_dataset_with_LE.csv"
NTL_CSV  = "./ntl_county_year_raw.csv"
PM25_CSV = "./pm25_county_year.csv"

OUTPUT_DIR = Path("./nightlights_pm25_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_MODE = False   # FULL PRODUCTION RUN

def _load_ml_diag():
    if not ML_DIAG_FILE.exists():
        raise FileNotFoundError(f"Could not find {ML_DIAG_FILE}.")
    spec = importlib.util.spec_from_file_location("ml_diag", ML_DIAG_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ml_diag"] = mod
    spec.loader.exec_module(mod)
    return mod

def _standardise_join_keys(df):
    """
    Standardises keys by destroying duplicates BEFORE renaming, 
    ensuring 'fips' and 'year' are singular, clean Series.
    """
    df = df.copy()

    # 1. If both FIPS and fips exist, destroy the uppercase one to avoid collision.
    if 'FIPS' in df.columns and 'fips' in df.columns:
        df = df.drop(columns=['FIPS'])
        
    # 2. If ONLY uppercase FIPS exists, rename it.
    elif 'FIPS' in df.columns and 'fips' not in df.columns:
        df = df.rename(columns={'FIPS': 'fips'})
        
    # 3. Destroy any remaining duplicated column names
    df = df.loc[:, ~df.columns.duplicated()]

    # 4. Now we are mathematically guaranteed to have exactly ONE 'fips' column.
    if 'fips' not in df.columns:
        raise KeyError(f"Fatal: No 'fips' column survived cleanup. Available: {list(df.columns)}")
        
    # Standardise FIPS: string -> split decimals -> pad to 5 digits
    df['fips'] = df['fips'].astype(str).str.split('.').str[0].str.zfill(5)
    
    # 5. Handle Year
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    else:
        raise KeyError(f"Fatal: No 'year' column found in dataset. Available: {list(df.columns)}")
        
    return df

def load_ntl(ntl_csv=NTL_CSV):
    df = _standardise_join_keys(pd.read_csv(ntl_csv))
    stat_cols = ["mean", "stdDev", "p10", "p25", "p50", "p75", "p90"]
    stat_cols = [c for c in stat_cols if c in df.columns]

    out = df[["fips", "year"]].copy()
    if "sensor" in df.columns:
        out["sensor"] = df["sensor"].values
        for c in stat_cols:
            col = f"NTL_harmonized_{c}"
            out[col] = np.nan
            for sensor in out["sensor"].dropna().unique():
                m = out["sensor"] == sensor
                v = pd.to_numeric(df.loc[m, c], errors="coerce")
                lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
                out.loc[m, col] = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        out = out.drop(columns=["sensor"])
    else:
        for c in stat_cols: out[f"NTL_harmonized_{c}"] = pd.to_numeric(df[c], errors="coerce")
    return out

def load_pm25(pm25_csv=PM25_CSV):
    df = _standardise_join_keys(pd.read_csv(pm25_csv))
    ren = {}
    for c in df.columns:
        if c.lower().startswith("pm25"):
            ren[c] = c if c.startswith("PM25_") else "PM25_" + c.split("pm25_")[-1]
    for c in ["mean", "stdDev", "p10", "p25", "p50", "p75", "p90"]:
        if c in df.columns and f"PM25_{c}" not in ren.values(): ren[c] = f"PM25_{c}"
    df = df.rename(columns=ren)
    keep = ["fips", "year"] + [c for c in df.columns if c.startswith("PM25_")]
    out = df[keep].copy()
    
    # FIXED: Skip 'fips' and 'year' so they stay strings/ints for the merge!
    for c in out.columns: 
        if c not in ["fips", "year"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _block_columns(df, prefix_tuple):
    return [c for c in df.columns if c.startswith(prefix_tuple)]

def main():
    ml = _load_ml_diag()
    ml.FAST_MODE = FAST_MODE         
    ml.OUTPUT_DIR = OUTPUT_DIR        
    ml.DATA_FILE = DATA_FILE

    base = ml.load_and_preprocess(DATA_FILE)
    ntl = load_ntl(NTL_CSV)
    pm25 = load_pm25(PM25_CSV)

    # Force the base dataframe's fips to be a padded string matching ntl/pm25
    base['fips'] = base['fips'].astype(str).str.split('.').str[0].str.zfill(5)
    
    # The Merge
    df = base.merge(ntl, on=["fips", "year"], how="left").merge(pm25, on=["fips", "year"], how="left")
    
    new_cols = _block_columns(df, ("NTL_", "PM25_"))
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)
    df[new_cols] = df.groupby("fips")[new_cols].bfill().ffill()

    y = df["MeanLifeExpectency"].astype(float)
    groups = df["fips"].values
    years = df["year"].values

    base_feats = [c for c in ml.feature_columns(df, drop_backfilled=False) if not c.startswith(("NTL_", "PM25_"))]
    prod_feats = ml.prune_features(df[base_feats], y)
    
    ntl_feats = _block_columns(df, ("NTL_harmonized_",)) or _block_columns(df, ("NTL_",))
    pm_feats = _block_columns(df, ("PM25_",))

    def run(tag, feats, note):
        feats = [c for c in feats if c in df.columns]
        if not feats: return
        agg, _ = ml.grouped_cv(df[feats], y, groups, years, ml.make_rf, save_predictions_to=f"pred_{tag}.csv", tag=tag)
        ml.record("NTL_PM25", tag, agg, len(feats), len(df), note)

    print("\n" + "=" * 78)
    print("NIGHTTIME-LIGHTS / PM2.5 ABLATIONS (HPC RUN)")
    print("=" * 78)
    run("full_reference", prod_feats, "Production satellite-only model (no NTL/PM2.5)")
    run("NTL_only", ntl_feats, "R1.4: nighttime lights as the SOLE feature set.")
    run("PM25_only", pm_feats, "R3.1: PM2.5 as the sole feature set.")
    run("NTL_plus_PM25_only", ntl_feats + pm_feats, "Pure pollution/urbanisation proxy set.")
    run("full_plus_NTL", prod_feats + ntl_feats, "Marginal value of adding NTL.")
    run("full_plus_PM25", prod_feats + pm_feats, "Marginal value of adding PM2.5.")
    run("full_plus_NTL_PM25", prod_feats + ntl_feats + pm_feats, "Fused model with both.")

    res = pd.DataFrame(ml.RESULTS)
    out_csv = OUTPUT_DIR / "nightlights_pm25_ablation_results.csv"
    res.to_csv(out_csv, index=False)
    _make_ablation_figure(res, OUTPUT_DIR)

def _make_ablation_figure(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = res[res["Analysis"] == "NTL_PM25"].copy()
    if sub.empty: return
    order = sub.sort_values("R2_mean")
    labels = {
        "full_reference": "Production (satellite-only)",
        "full_plus_NTL_PM25": "Production + NTL + PM2.5",
        "full_plus_NTL": "Production + NTL",
        "full_plus_PM25": "Production + PM2.5",
        "NTL_plus_PM25_only": "NTL + PM2.5 only",
        "NTL_only": "NTL only  (R1.4)",
        "PM25_only": "PM2.5 only  (R3.1)",
    }
    names = [labels.get(v, v) for v in order["Variant"]]
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(order) + 2))
    colors = ["#0072B2" if "Production" in n and "only" not in n else "#E69F00" for n in names]
    ax.barh(range(len(order)), order["R2_mean"], xerr=order["R2_sd"], color=colors, alpha=0.9, edgecolor="white", capsize=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Cross-validated $R^2$ (county-grouped 5-fold)")
    ax.set_title("Nighttime lights & PM2.5: feature-block ablation", fontsize=11)
    for i, (r2, sd) in enumerate(zip(order["R2_mean"], order["R2_sd"])):
        ax.text(r2 + sd + 0.005, i, f"{r2:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figR_nightlights_pm25_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
