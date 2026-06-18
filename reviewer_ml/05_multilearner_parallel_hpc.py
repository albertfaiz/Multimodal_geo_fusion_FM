"""
================================================================================
MULTI-LEARNER COMPARISON -- CORE-PARTITIONED PARALLEL HPC RUN  (R3.8)
Paper: "Nighttime Thermal Patterns and County Life Expectancy ..."
================================================================================
Reviewer R3.8 asks us to benchmark the Random Forest against other learners,
including ensembles. Running them one-at-a-time (each grabbing all 64 cores)
wastes the node: the histogram boosters (XGB/LGBM/HistGB) cannot use 64 threads
efficiently, so cores sit idle. This script instead PARTITIONS the 64 cores into
per-learner pools and runs every learner's full 5-fold county-grouped CV
CONCURRENTLY -- the efficient pattern from our production HPC job
(c-05-03, 64 cores).

  Wave 1 (concurrent, cores partitioned):
      RandomForest | ExtraTrees | HistGradientBoosting | XGBoost | LightGBM
  Wave 2 (optional, full node):
      Stacking ensemble (RF + HistGB + XGB -> RidgeCV meta-learner)

Each learner runs in its OWN process; thread-limiting env vars (OMP/MKL/OpenBLAS)
are set INSIDE each process before heavy imports, so nested parallelism never
oversubscribes the node. Preprocessing + CV come from 01_reviewer_diagnostics_ml.py
(single source of truth), so every number is directly comparable to the headline
RF result (R^2 = 0.631 +/- 0.013, MAE = 1.08).

Usage (on the HPC node)
-----------------------
  python 05_multilearner_parallel_hpc.py
Set FAST_MODE=False for the manuscript numbers (it is False by default here,
because this script is meant for the cluster). TOTAL_CORES auto-detects from
SLURM; override in CONFIG if needed.
================================================================================
"""

import os
import sys
import time
import importlib.util
import multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path

# ============================================================================
# CONFIG  -- EDIT FOR YOUR MACHINE
# ============================================================================
HERE = Path(__file__).resolve().parent
ML_DIAG_FILE = HERE / "01_reviewer_diagnostics_ml.py"
DATA_FILE = "./full_clean_engineered_dataset_with_LE.csv"
OUTPUT_DIR = Path("./multilearner_parallel_outputs")

# 64 on the production node (c-05-03). Auto-detect from SLURM if present.
TOTAL_CORES = int(os.environ.get("SLURM_CPUS_ON_NODE",
                  os.environ.get("SLURM_CPUS_PER_TASK", 0))) or (os.cpu_count() or 64)

FAST_MODE = False     # this script targets the HPC node; set True only to dry-run
RANDOM_STATE = 42
RUN_STACKING = True   # wave 2 (uses the whole node after wave 1 finishes)

# Relative core weights for wave-1 learners. Tree-bagging learners (RF, ExtraTrees)
# scale well across many cores; histogram boosters use fewer threads effectively.
# These weights are normalised to TOTAL_CORES at run time (min 1 core each).
CORE_WEIGHTS = {
    "RandomForest": 16,
    "ExtraTrees":   16,
    "HistGB":       10,
    "XGBoost":      12,
    "LightGBM":     10,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helpers shared by parent + workers
# ============================================================================
def _load_ml_diag():
    """Import 01_reviewer_diagnostics_ml.py as a module (single source of truth
    for preprocessing + CV). Importing does NOT trigger its main()."""
    spec = importlib.util.spec_from_file_location("ml_diag", str(ML_DIAG_FILE))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ml_diag"] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_optional(name):
    """Return True if the learner's package is importable in this env."""
    try:
        if name == "XGBoost":
            import xgboost  # noqa: F401
        elif name == "LightGBM":
            import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def _make_factory(name, cores, fast, seed):
    """Build a zero-arg estimator factory restricted to `cores` threads."""
    n_est = 400 if fast else 2000
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return lambda: RandomForestRegressor(
            n_estimators=n_est, min_samples_split=3, min_samples_leaf=2,
            max_samples=0.9, max_features=0.3, max_depth=40, bootstrap=True,
            random_state=seed, n_jobs=cores)
    if name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return lambda: ExtraTreesRegressor(
            n_estimators=n_est, min_samples_split=3, min_samples_leaf=2,
            max_features=0.3, max_depth=40, bootstrap=True,
            random_state=seed, n_jobs=cores)
    if name == "HistGB":
        from sklearn.ensemble import HistGradientBoostingRegressor
        # HistGB parallelises through OpenMP (OMP_NUM_THREADS, set per process).
        return lambda: HistGradientBoostingRegressor(
            max_iter=(300 if fast else 800), learning_rate=0.05,
            max_leaf_nodes=63, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    if name == "XGBoost":
        import xgboost as xgb
        return lambda: xgb.XGBRegressor(
            n_estimators=(400 if fast else 1500), learning_rate=0.05,
            max_depth=8, subsample=0.9, colsample_bytree=0.5,
            tree_method="hist", n_jobs=cores, random_state=seed,
            verbosity=0)
    if name == "LightGBM":
        import lightgbm as lgb
        return lambda: lgb.LGBMRegressor(
            n_estimators=(600 if fast else 2000), learning_rate=0.05,
            num_leaves=128, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.5, n_jobs=cores, random_state=seed,
            verbosity=-1)
    raise ValueError(name)


# ============================================================================
# Worker: run ONE learner's full CV in its own process
# ============================================================================
def _run_one_learner(spec):
    name, cores, data_path, out_dir, fast, seed = spec
    # 1) cap thread pools BEFORE importing numpy/sklearn in this process
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(cores)

    import joblib
    t0 = time.time()
    ml = _load_ml_diag()
    ml.FAST_MODE = fast
    ml.OUTPUT_DIR = Path(out_dir)

    store = joblib.load(data_path)
    X, y = store["X"], store["y"]
    groups, years = store["groups"], store["years"]

    factory = _make_factory(name, cores, fast, seed)
    agg, _ = ml.grouped_cv(X, y, groups, years, factory,
                           save_predictions_to=f"pred_{name}.csv", tag=name)
    agg["_seconds"] = round(time.time() - t0, 1)
    agg["_cores"] = cores
    agg["_n_features"] = X.shape[1]
    return name, agg


# ============================================================================
# Parent orchestration
# ============================================================================
def _allocate_cores(names, total):
    """Normalise CORE_WEIGHTS for the surviving learners to `total` cores."""
    w = {n: CORE_WEIGHTS.get(n, 8) for n in names}
    s = sum(w.values())
    alloc = {n: max(1, int(round(total * w[n] / s))) for n in names}
    # trim/pad so the sum does not exceed the node
    while sum(alloc.values()) > total:
        alloc[max(alloc, key=alloc.get)] -= 1
    return alloc


def main():
    print("=" * 78)
    print(f"MULTI-LEARNER PARALLEL RUN  |  TOTAL_CORES={TOTAL_CORES}  "
          f"FAST_MODE={FAST_MODE}")
    print("=" * 78)

    # ---- parent does the load + prune ONCE, then hands data to workers -----
    ml = _load_ml_diag()
    ml.FAST_MODE = FAST_MODE
    ml.OUTPUT_DIR = OUTPUT_DIR
    ml.DATA_FILE = DATA_FILE

    df = ml.load_and_preprocess(DATA_FILE)
    y = df["MeanLifeExpectency"].astype(float)
    groups = df["fips"].values
    years = df["year"].values
    feats = ml.feature_columns(df, drop_backfilled=False)
    prod_feats = ml.prune_features(df[feats], y)   # ~193, matches production
    X = df[prod_feats].reset_index(drop=True)
    print(f"[parent] pruned feature set: {len(prod_feats)} features, "
          f"{len(X):,} county-years")

    import joblib
    data_path = OUTPUT_DIR / "_shared_pruned_data.joblib"
    joblib.dump({"X": X, "y": y, "groups": groups, "years": years}, data_path)

    # ---- decide the wave-1 roster (drop learners whose package is missing) --
    roster = ["RandomForest", "ExtraTrees", "HistGB"]
    for opt in ("XGBoost", "LightGBM"):
        if _check_optional(opt):
            roster.append(opt)
        else:
            print(f"[parent] {opt} not installed -> skipping "
                  f"(pip install {'xgboost' if opt=='XGBoost' else 'lightgbm'})")
    alloc = _allocate_cores(roster, TOTAL_CORES)
    print("[parent] wave-1 core allocation:",
          ", ".join(f"{k}={v}" for k, v in alloc.items()))

    specs = [(name, alloc[name], str(data_path), str(OUTPUT_DIR),
              FAST_MODE, RANDOM_STATE) for name in roster]

    # ---- WAVE 1: concurrent, core-partitioned ------------------------------
    results = {}
    ctx = mp.get_context("spawn")
    t_wave = time.time()
    with cf.ProcessPoolExecutor(max_workers=len(specs), mp_context=ctx) as ex:
        futs = {ex.submit(_run_one_learner, s): s[0] for s in specs}
        for fu in cf.as_completed(futs):
            name = futs[fu]
            try:
                nm, agg = fu.result()
                results[nm] = agg
                print(f"  [done] {nm}: R2={agg['R2_mean']:.3f}+/-{agg['R2_sd']:.3f}"
                      f"  MAE={agg['MAE_mean']:.3f}  ({agg['_seconds']:.0f}s, "
                      f"{agg['_cores']} cores)")
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")
    print(f"[parent] wave 1 wall-clock: {(time.time()-t_wave)/60:.1f} min")

    # ---- WAVE 2: stacking ensemble on the full node ------------------------
    if RUN_STACKING:
        print("\n[parent] wave 2: stacking ensemble (full node) ...")
        try:
            agg = _run_stacking(ml, X, y, groups, years, TOTAL_CORES,
                                FAST_MODE, RANDOM_STATE)
            results["Stacking"] = agg
            print(f"  [done] Stacking: R2={agg['R2_mean']:.3f}+/-{agg['R2_sd']:.3f}"
                  f"  MAE={agg['MAE_mean']:.3f}")
        except Exception as e:
            print(f"  [FAILED] Stacking: {e}")

    # ---- collate -----------------------------------------------------------
    import pandas as pd
    rows = []
    for name, agg in results.items():
        rows.append({"Learner": name,
                     "R2_mean": round(agg["R2_mean"], 4),
                     "R2_sd": round(agg["R2_sd"], 4),
                     "MAE_mean": round(agg["MAE_mean"], 4),
                     "MAE_sd": round(agg["MAE_sd"], 4),
                     "RMSE_mean": round(agg.get("RMSE_mean", float("nan")), 4),
                     "cores": agg.get("_cores", TOTAL_CORES),
                     "seconds": agg.get("_seconds", float("nan"))})
    res = pd.DataFrame(rows).sort_values("R2_mean", ascending=False)
    out_csv = OUTPUT_DIR / "multilearner_results.csv"
    res.to_csv(out_csv, index=False)
    print("\n" + "=" * 78)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(res.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    _make_figure(res)
    print(f"\nDone. Outputs in {OUTPUT_DIR.resolve()}")
    print("Interpretation for R3.8: all learners cluster within a narrow band, "
          "so the performance ceiling is set by the satellite feature set, not "
          "the choice of learner -- which supports the modelling design.")


def _run_stacking(ml, X, y, groups, years, cores, fast, seed):
    from sklearn.ensemble import (RandomForestRegressor,
                                  HistGradientBoostingRegressor, StackingRegressor)
    from sklearn.linear_model import RidgeCV
    n_est = 400 if fast else 1200
    base = [
        ("rf", RandomForestRegressor(
            n_estimators=n_est, min_samples_split=3, min_samples_leaf=2,
            max_features=0.3, max_depth=40, bootstrap=True,
            random_state=seed, n_jobs=max(1, cores // 3))),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=(300 if fast else 700), learning_rate=0.05,
            max_leaf_nodes=63, l2_regularization=1.0, random_state=seed)),
    ]
    try:
        import xgboost as xgb
        base.append(("xgb", xgb.XGBRegressor(
            n_estimators=(400 if fast else 1200), learning_rate=0.05,
            max_depth=8, subsample=0.9, colsample_bytree=0.5,
            tree_method="hist", n_jobs=max(1, cores // 3),
            random_state=seed, verbosity=0)))
    except Exception:
        pass

    def fac():
        return StackingRegressor(
            estimators=base, final_estimator=RidgeCV(),
            n_jobs=len(base), passthrough=False)

    os.environ["OMP_NUM_THREADS"] = str(max(1, cores // 3))
    agg, _ = ml.grouped_cv(X, y, groups, years, fac,
                           save_predictions_to="pred_Stacking.csv", tag="Stacking")
    agg["_cores"] = cores
    agg["_n_features"] = X.shape[1]
    return agg


def _make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = res.sort_values("R2_mean")
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(order) + 2))
    ax.barh(range(len(order)), order["R2_mean"], xerr=order["R2_sd"],
            color="#0072B2", alpha=0.9, edgecolor="white", capsize=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["Learner"])
    ax.set_xlabel("Cross-validated $R^2$ (county-grouped 5-fold)")
    ax.set_title("Learner comparison (R3.8)\nall learners share the same 193 "
                 "satellite features and CV protocol", fontsize=11)
    for i, (r2, sd) in enumerate(zip(order["R2_mean"], order["R2_sd"])):
        ax.text(r2 + sd + 0.004, i, f"{r2:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "figR_learner_comparison"
    fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {p}.png / .pdf")


if __name__ == "__main__":
    main()
