import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, StackingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================================
# HPC CONFIGURATION
# ============================================================================
DATA_FILE  = "./full_clean_engineered_dataset_with_LE.csv"
COUNTY_SHP = "./cb_2018_us_county_500k/cb_2018_us_county_500k.shp"
OUTPUT_DIR = Path("./reviewer_diagnostics_outputs")

FAST_MODE   = False     # FULL PRODUCTION RUN
RUN_MULTILEARNER_HERE = False 
N_EST_FAST  = 400       
RANDOM_STATE = 42
N_FOLDS      = 5
N_JOBS       = -1       # Uses all available HPC cores

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKFILLED_PREFIXES = ("Landsat_", "S1_", "S2_")
BACKFILLED_ENG = {"ENG_SAR_NDVI_Struct", "ENG_Cross_Sensor_NDVI", "ENG_Wet_Bulb_Proxy"}

PRODUCTION_RF_PARAMS = dict(
    n_estimators=2000, min_samples_split=3, min_samples_leaf=2,
    max_samples=0.9, max_features=0.3, max_depth=40,
    bootstrap=True, random_state=RANDOM_STATE, n_jobs=N_JOBS,
)
PRUNE_THRESHOLD = 0.001

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================
def load_and_preprocess(data_file=DATA_FILE):
    print(f"[load] reading {data_file}")
    df = pd.read_csv(data_file)
    if "MeanLifeExpectency_x" in df.columns:
        df["MeanLifeExpectency"] = df["MeanLifeExpectency_x"]
        df = df.drop(columns=["MeanLifeExpectency_x", "MeanLifeExpectency_y"], errors="ignore")
    df = df.dropna(subset=["MeanLifeExpectency"]).reset_index(drop=True)
    df["fips"] = df["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    df = df[~df["fips"].str.startswith(("02", "15"))].reset_index(drop=True)
    
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)
    cols_to_fill = [c for c in df.columns if c not in ["fips", "year", "location_name", "MeanLifeExpectency"]]
    df[cols_to_fill] = df.groupby("fips")[cols_to_fill].bfill().ffill()
    return df

def feature_columns(df, drop_backfilled=False):
    drop = ["MeanLifeExpectency", "year", "fips", "location_name"]
    drop += [c for c in df.columns if c.startswith(("count_", "sum_", "area_"))]
    feats = [c for c in df.columns if c not in drop]
    if drop_backfilled:
        feats = [c for c in feats if not c.startswith(BACKFILLED_PREFIXES) and c not in BACKFILLED_ENG]
    return feats

def prune_features(X_raw, y, threshold=PRUNE_THRESHOLD, seed=RANDOM_STATE):
    Xf = X_raw.replace([np.inf, -np.inf], np.nan)
    Xf = Xf.fillna(Xf.median())
    m = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, n_jobs=N_JOBS)
    m.fit(Xf, y)
    imp = m.feature_importances_
    return list(X_raw.columns[imp >= threshold])

def _fold_preprocess(Xtr, Xte):
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)
    med = Xtr.median()
    lo, hi = Xtr.quantile(0.001), Xtr.quantile(0.999)
    Xtr = Xtr.fillna(med).clip(lower=lo, upper=hi, axis=1)
    Xte = Xte.fillna(med).clip(lower=lo, upper=hi, axis=1)
    return Xtr, Xte

def make_rf(fast=None):
    p = dict(PRODUCTION_RF_PARAMS)
    if (FAST_MODE if fast is None else fast):
        p["n_estimators"] = N_EST_FAST
    return RandomForestRegressor(**p)

def _metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAPE=float(np.mean(np.abs((y - p) / y)) * 100))

def _agg(fold_metrics):
    df = pd.DataFrame(fold_metrics)
    out = {}
    for k in ["R2", "MAE", "RMSE", "MAPE"]:
        out[f"{k}_mean"] = df[k].mean()
        out[f"{k}_sd"] = df[k].std()
    return out

def grouped_cv(X, y, groups, years, model_factory, n_splits=N_FOLDS, save_predictions_to=None, tag=""):
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics, preds = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        t0 = time.time()
        Xtr, Xte = _fold_preprocess(X.iloc[tr].copy(), X.iloc[te].copy())
        model = model_factory()
        model.fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        m = _metrics(y.iloc[te].values, pred); m["Fold"] = fold
        fold_metrics.append(m)
        preds.append(pd.DataFrame({
            "fold": fold, "fips": np.asarray(groups)[te],
            "year": np.asarray(years)[te], "actual": y.iloc[te].values,
            "predicted": pred, "residual": y.iloc[te].values - pred,
            "abs_error": np.abs(y.iloc[te].values - pred)}))
        print(f"    [{tag}] fold {fold}: R2={m['R2']:.3f} MAE={m['MAE']:.3f} ({time.time()-t0:.0f}s)")
    pred_df = pd.concat(preds, ignore_index=True)
    if save_predictions_to:
        pred_df.to_csv(OUTPUT_DIR / save_predictions_to, index=False)
    return _agg(fold_metrics), pred_df

RESULTS = []
def record(analysis, variant, agg, n_feat, n_obs, note=""):
    row = {"Analysis": analysis, "Variant": variant, "N_features": n_feat, "N_obs": n_obs}
    row.update({k: round(v, 4) for k, v in agg.items()})
    row["Note"] = note
    RESULTS.append(row)
    print(f"  >> {analysis} | {variant}: R2={agg['R2_mean']:.3f}+/-{agg['R2_sd']:.3f}, MAE={agg['MAE_mean']:.3f}+/-{agg['MAE_sd']:.3f}")

# ============================================================================
# EXPERIMENTS
# ============================================================================
def run_naive(y, groups, n_splits=N_FOLDS):
    print("\n[A_naive] Naive mean baseline")
    gkf = GroupKFold(n_splits=n_splits)
    fm = []
    Xdummy = pd.DataFrame({"_": np.zeros(len(y))})
    for fold, (tr, te) in enumerate(gkf.split(Xdummy, y, groups), 1):
        pred = np.full(len(te), y.iloc[tr].mean())
        fm.append(_metrics(y.iloc[te].values, pred))
    record("A_naive", "predict_training_mean", _agg(fm), 0, len(y), "R1.1 floor baseline")

def run_spatial_knn(df, y, groups, county_shp=COUNTY_SHP, n_splits=N_FOLDS):
    print("\n[A_spatial_knn] Spatial proximity baseline (R1.3)")
    try:
        import geopandas as gpd
        from scipy.spatial import cKDTree
    except Exception as e:
        print(f"  !! geopandas/scipy unavailable ({e}); skipping spatial k-NN.")
        return
    if not os.path.exists(county_shp):
        print(f"  !! shapefile not found at {county_shp}; skipping spatial k-NN.")
        return

    gdf = gpd.read_file(county_shp).to_crs("EPSG:5070")
    gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    gdf = gdf[gdf["fips"].isin(df["fips"].unique())].copy()
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    cent = gdf.set_index("fips")[["cx", "cy"]].to_dict("index")
    le_lut = {(r.fips, int(r.year)): r.MeanLifeExpectency for r in df[["fips", "year", "MeanLifeExpectency"]].itertuples(index=False)}

    Xdummy = pd.DataFrame({"_": np.zeros(len(df))})
    fips_arr = df["fips"].values
    year_arr = df["year"].values
    y_arr = y.values

    for k, weighting in [(4, "idw"), (4, "uniform"), (8, "idw")]:
        gkf = GroupKFold(n_splits=n_splits)
        fm = []
        for fold, (tr, te) in enumerate(gkf.split(Xdummy, y, groups), 1):
            train_fips = sorted(set(fips_arr[tr]) & set(cent.keys()))
            tf_xy = np.array([[cent[f]["cx"], cent[f]["cy"]] for f in train_fips])
            tree = cKDTree(tf_xy)
            test_fips_unique = sorted(set(fips_arr[te]) & set(cent.keys()))
            nbr = {}
            for f in test_fips_unique:
                d, idx = tree.query([cent[f]["cx"], cent[f]["cy"]], k=k)
                d = np.atleast_1d(d); idx = np.atleast_1d(idx)
                nbr[f] = [(train_fips[j], dd) for j, dd in zip(idx, d)]
            preds, actuals = [], []
            for i in te:
                f, yr = fips_arr[i], int(year_arr[i])
                if f not in nbr: continue
                vals, wts = [], []
                for nf, dd in nbr[f]:
                    le = le_lut.get((nf, yr))
                    if le is None or not np.isfinite(le): continue
                    vals.append(le)
                    wts.append(1.0 / max(dd, 1.0) if weighting == "idw" else 1.0)
                if not vals: continue
                preds.append(np.average(vals, weights=wts))
                actuals.append(y_arr[i])
            fm.append(_metrics(actuals, preds))
        record("A_spatial_knn", f"k={k}_{weighting}", _agg(fm), 0, len(df), "R1.3: Needs neighbors OBSERVED LE")

def run_temporal_holdout(df, X, y, groups, years, prod_feats):
    print("\n[C_temporal_holdout] Temporal generalisation (R3.3, R4.3)")
    for tr_end, label in [(2014, "train<=2014_test>=2015"), (2009, "train<=2009_test>=2010")]:
        tr = np.where(years <= tr_end)[0]
        te = np.where(years > tr_end)[0]
        Xtr, Xte = _fold_preprocess(X.iloc[tr].copy(), X.iloc[te].copy())
        model = make_rf()
        model.fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        m = _metrics(y.iloc[te].values, pred)
        agg = {f"{k}_mean": m[k] for k in ["R2", "MAE", "RMSE", "MAPE"]}
        agg.update({f"{k}_sd": 0.0 for k in ["R2", "MAE", "RMSE", "MAPE"]})
        record("C_temporal_holdout", label, agg, len(prod_feats), len(df), "R3.3")
        pd.DataFrame({"fips": np.asarray(groups)[te], "year": years[te], "actual": y.iloc[te].values, "predicted": pred}).to_csv(OUTPUT_DIR / f"predictions_temporal_{label}.csv", index=False)

def run_state_generalisation(df, X, y, prod_feats):
    print("\n[C_state] Region-independent generalisation (R1.5)")
    states = df["fips"].str[:2].values
    years = df["year"].values
    agg, _ = grouped_cv(X, y, states, years, make_rf, save_predictions_to="predictions_state_groupcv.csv", tag="state-GroupKFold")
    record("C_state_groupcv", "leave-states-out 5-fold", agg, len(prod_feats), len(df), "R1.5 proper spatial-block CV")

    uniq_states = sorted(set(states))
    rng = np.random.default_rng(RANDOM_STATE)
    fm = []
    for rep in range(8):
        train_states = set(rng.choice(uniq_states, size=10, replace=False))
        tr = np.where(np.isin(states, list(train_states)))[0]
        te = np.where(~np.isin(states, list(train_states)))[0]
        Xtr, Xte = _fold_preprocess(X.iloc[tr].copy(), X.iloc[te].copy())
        model = make_rf()
        model.fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        fm.append(_metrics(y.iloc[te].values, pred))
    record("C_state_random_10_40", "train 10 states / test 40 (8 random draws)", _agg(fm), len(prod_feats), len(df), "R1.5 exact request")

def run_temporally_complete(df, y, groups, years):
    print("\n[D_temporally_complete] Only-temporally-complete-sources control (R3.5, R2.1, R4.3)")
    feats_raw = feature_columns(df, drop_backfilled=True)
    kept = prune_features(df[feats_raw], y)
    agg, _ = grouped_cv(df[kept].copy(), y, groups, years, make_rf, tag="temporally-complete")
    record("D_temporally_complete", "drop Landsat8-9 + Sentinel-1/2", agg, len(kept), len(df), "R3.5")

def run_modern_only(df, y, groups, years, prod_feats):
    print("\n[D_modern_only] Modern-period-only model 2015-2019 (R2.1, R4.3)")
    mask = years >= 2015
    sub = df.loc[mask].reset_index(drop=True)
    agg, _ = grouped_cv(sub[prod_feats].copy(), sub["MeanLifeExpectency"], sub["fips"].values, sub["year"].values, make_rf, tag="modern-only")
    record("D_modern_only", "2015-2019 county-years", agg, len(prod_feats), len(sub), "R2.1")

def run_drift_detrended(df, y, groups, years):
    print("\n[E_drift_detrended] Orbital-drift robustness: year-detrended nighttime LST (R3.4)")
    d2 = df.copy()
    lst_cols = [c for c in d2.columns if c.startswith("LST_")]
    for c in lst_cols:
        yr_mean = d2.groupby("year")[c].transform("mean")
        d2[c] = d2[c] - yr_mean + d2[c].mean()
    feats_raw = feature_columns(d2)
    kept = prune_features(d2[feats_raw], y)
    agg, _ = grouped_cv(d2[kept].copy(), y, groups, years, make_rf, tag="drift-detrended")
    record("E_drift_detrended", "year-detrended LST", agg, len(kept), len(df), "R3.4")

def run_perfold_pruning(df, y, groups, years, prod_feats):
    print("\n[G_perfold_pruning] Per-fold supervised pruning leakage check (R4.7)")
    feats_raw = feature_columns(df)
    Xraw = df[feats_raw]
    gkf = GroupKFold(n_splits=N_FOLDS)
    fm, selected_sets = [], []
    for fold, (tr, te) in enumerate(gkf.split(Xraw, y, groups), 1):
        Xtr_full = Xraw.iloc[tr].replace([np.inf, -np.inf], np.nan).fillna(Xraw.iloc[tr].median())
        pruner = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        pruner.fit(Xtr_full, y.iloc[tr])
        sel = list(Xraw.columns[pruner.feature_importances_ >= PRUNE_THRESHOLD])
        selected_sets.append(set(sel))
        Xtr, Xte = _fold_preprocess(Xraw.iloc[tr][sel].copy(), Xraw.iloc[te][sel].copy())
        model = make_rf()
        model.fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        m = _metrics(y.iloc[te].values, pred); m["Fold"] = fold; m["n_selected"] = len(sel)
        fm.append(m)
    record("G_perfold_pruning", "pruning fit on TRAIN fold only", _agg(fm), int(np.mean([m['n_selected'] for m in fm])), len(df), "R4.7")

def make_summary_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.DataFrame(RESULTS)
    wanted = [("A_naive", "Naive mean"), ("A_spatial_knn", "Spatial k-NN (k=4, IDW)"),
              ("C_state_groupcv", "Leave-states-out CV"), ("C_temporal_holdout", "Temporal holdout (2015-19)"),
              ("D_temporally_complete", "Always-available sources"), ("D_modern_only", "Modern only (2015-19)"),
              ("Production", "Production (county-grouped)")]
    rows = []
    for analysis, label in wanted:
        sub = res[res["Analysis"] == analysis]
        if analysis == "A_spatial_knn": sub = sub[sub["Variant"].str.startswith("k=4_idw")]
        if analysis == "C_temporal_holdout": sub = sub[sub["Variant"].str.contains("2015")]
        if len(sub):
            r = sub.iloc[0]
            rows.append((label, r["R2_mean"], r.get("R2_sd", 0), r["MAE_mean"], r.get("MAE_sd", 0)))
    if not rows: return
    labels = [r[0] for r in rows]; r2 = [r[1] for r in rows]; r2e = [r[2] for r in rows]
    mae = [r[3] for r in rows]; maee = [r[4] for r in rows]
    colors = ["#999999" if "Naive" in l or "k-NN" in l else "#0072B2" for l in labels]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    x = np.arange(len(labels))
    ax[0].bar(x, r2, yerr=r2e, capsize=4, color=colors, edgecolor="white")
    ax[0].axhline(0.82, ls="--", color="#D55E00", lw=1.6, label="IHME socio-demographic R$^2$=0.82")
    ax[0].set_ylabel("R$^2$ (test)", fontweight="bold")
    ax[0].set_title("(A) R$^2$ across baselines & validation regimes", fontweight="bold", loc="left")
    ax[0].legend(fontsize=9)
    ax[1].bar(x, mae, yerr=maee, capsize=4, color=colors, edgecolor="white")
    ax[1].set_ylabel("MAE (years)", fontweight="bold")
    ax[1].set_title("(B) Mean absolute error", fontweight="bold", loc="left")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        a.grid(axis="y", alpha=0.25)
    fig.suptitle("Generalisation stress-test: baselines vs the satellite model", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"figR_generalisation_stress_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    print("=" * 80)
    print(f"REVIEWER DIAGNOSTICS (HPC RUN - FAST_MODE={FAST_MODE})")
    print("=" * 80)
    df = load_and_preprocess()
    y = df["MeanLifeExpectency"]
    groups = df["fips"].values
    years = df["year"].values

    feats_raw = feature_columns(df)
    prod_feats = prune_features(df[feats_raw], y)
    Xprod = df[prod_feats].copy()

    print("\n[Production] reference reproduction (county-grouped CV)")
    agg, _ = grouped_cv(Xprod, y, groups, years, make_rf, save_predictions_to="predictions_production_reference.csv", tag="production-ref")
    record("Production", "county-grouped 5-fold (reference)", agg, len(prod_feats), len(df), "Baseline")

    run_naive(y, groups)
    run_spatial_knn(df, y, groups)
    run_temporal_holdout(df, Xprod, y, groups, years, prod_feats)
    run_state_generalisation(df, Xprod, y, prod_feats)
    run_temporally_complete(df, y, groups, years)
    run_modern_only(df, y, groups, years, prod_feats)
    run_drift_detrended(df, y, groups, years)
    run_perfold_pruning(df, y, groups, years, prod_feats)

    res = pd.DataFrame(RESULTS)
    out_csv = OUTPUT_DIR / "reviewer_diagnostics_master_results.csv"
    res.to_csv(out_csv, index=False)
    make_summary_figure()
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()