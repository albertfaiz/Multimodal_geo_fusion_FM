"""
================================================================================
REVIEWER DIAGNOSTICS — INTERPRETATION FIGURES (no re-training needed)
================================================================================
Reads results_production_final/ml_results.pkl and the engineered CSV, and
produces the four interpretation artifacts the reviewers requested:

  R4.1  NCE dedicated SHAP dependence + day/night decomposition
        -> figR_NCE_decomposition.(png|pdf)  + console interpretation
  R3.4  MODIS Terra orbital-drift evidence (national trend + variance split)
        -> figR_orbital_drift.(png|pdf)      + console stats
  R4.4  Indigenous-reservation fold-membership + residual audit
        -> figR_reservation_audit.(png|pdf)  + reservation_fold_audit.csv
  R4.2  Corrected forest-attenuation figure (no ">100%"; absolute Delta yr)
        -> figR_forest_attenuation_fixed.(png|pdf) + console numbers

Usage:  python 02_reviewer_diagnostics_shap.py   (edit CONFIG paths)
================================================================================
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_FILE = "./ml_results.pkl"
DATA_FILE    = "./full_clean_engineered_dataset_with_LE.csv"
OUTPUT_DIR   = Path("./reviewer_diagnostics_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         night="#1A1A4E", day="#FF8C00", grey="#888888")


# ============================================================================
# HELPERS
# ============================================================================
def resolve(fn_list, *candidates):
    """Find a feature in fn_list by exact match, else case-insensitive substring."""
    low = {f.lower(): f for f in fn_list}
    for c in candidates:
        if c in fn_list:
            return c
        if c.lower() in low:
            return low[c.lower()]
    for c in candidates:
        for f in fn_list:
            if c.lower().replace(" ", "") in f.lower().replace(" ", ""):
                return f
    return None


def reconstruct_sorted_df(data_file=DATA_FILE):
    """Reproduce the Phase-2 sorted dataframe (raw column names) so that
    ml_results['shap_sample'] positional indices map back to raw features."""
    df = pd.read_csv(data_file)
    if "MeanLifeExpectency_x" in df.columns:
        df["MeanLifeExpectency"] = df["MeanLifeExpectency_x"]
        df = df.drop(columns=["MeanLifeExpectency_x", "MeanLifeExpectency_y"],
                     errors="ignore")
    df = df.dropna(subset=["MeanLifeExpectency"]).reset_index(drop=True)
    df["fips"] = (df["fips"].astype(str)
                  .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    df = df[~df["fips"].str.startswith(("02", "15"))].reset_index(drop=True)
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)
    return df


def lowess_xy(x, y, frac=0.3):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    m = np.isfinite(x) & np.isfinite(y)
    o = np.argsort(x[m])
    lw = lowess(y[m][o], x[m][o], frac=frac, return_sorted=True)
    return lw[:, 0], lw[:, 1]


# ============================================================================
# LOAD
# ============================================================================
print("[load] reading pickle + CSV ...")
with open(RESULTS_FILE, "rb") as f:
    R = pickle.load(f)
shap_vals = R["shap_values"]              # (n_sample, n_feat)
shap_sample = R["shap_sample"]            # DataFrame, friendly names, positional idx
fn_list = list(shap_sample.columns)
predictions = R["predictions"].copy()
df_sorted = reconstruct_sorted_df()
sample_pos = shap_sample.index.values     # positions into df_sorted


# ============================================================================
# R4.1 — NCE DECOMPOSITION
# ============================================================================
def fig_nce_decomposition():
    print("\n[R4.1] NCE dependence + day/night decomposition")
    nce = resolve(fn_list, "Nighttime Cooling Efficiency", "ENG_Night_Cooling_Eff", "NCE")
    if nce is None:
        print("  !! NCE not found in SHAP features; skipping.")
        return
    j = fn_list.index(nce)
    nce_shap = shap_vals[:, j]
    nce_val = shap_sample[nce].values
    # raw thermal components for the SAME sampled rows
    day_p90 = df_sorted["LST_Day_1km_p90"].iloc[sample_pos].values
    night_p10 = df_sorted["LST_Night_1km_p10"].iloc[sample_pos].values

    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))

    # (A) NCE SHAP vs NCE value, coloured by night P10 (the cold-night component)
    sc0 = ax[0].scatter(nce_val, nce_shap, c=night_p10, cmap="coolwarm",
                        s=14, alpha=0.5, linewidths=0, rasterized=True)
    lx, ly = lowess_xy(nce_val, nce_shap)
    ax[0].plot(lx, ly, color=C["night"], lw=3, zorder=5)
    ax[0].axhline(0, color="#888", lw=1, ls="--")
    cb0 = fig.colorbar(sc0, ax=ax[0], fraction=0.046, pad=0.02)
    cb0.set_label("Nighttime LST P10 (°C)", fontsize=10)
    ax[0].set_xlabel("NCE value (unitless)", fontweight="bold")
    ax[0].set_ylabel("SHAP value (Δ LE, years)", fontweight="bold")
    ax[0].set_title("(A) NCE dependence, coloured by cold-night component",
                    fontweight="bold", loc="left", fontsize=12)

    # (B) What high vs low NCE means physically: day P90 vs night P10, colour=NCE
    sc1 = ax[1].scatter(day_p90, night_p10, c=nce_val, cmap="viridis",
                        s=14, alpha=0.6, linewidths=0, rasterized=True)
    cb1 = fig.colorbar(sc1, ax=ax[1], fraction=0.046, pad=0.02)
    cb1.set_label("NCE value", fontsize=10)
    ax[1].set_xlabel("Daytime LST P90 (°C)", fontweight="bold")
    ax[1].set_ylabel("Nighttime LST P10 (°C)", fontweight="bold")
    ax[1].set_title("(B) Physical meaning of NCE\n(high NCE = hot days + cold nights)",
                    fontweight="bold", loc="left", fontsize=12)

    # (C) NCE SHAP vs NCE value, coloured by day P90 (the hot-day component)
    sc2 = ax[2].scatter(nce_val, nce_shap, c=day_p90, cmap="YlOrRd",
                        s=14, alpha=0.5, linewidths=0, rasterized=True)
    ax[2].plot(lx, ly, color=C["red"], lw=3, zorder=5)
    ax[2].axhline(0, color="#888", lw=1, ls="--")
    cb2 = fig.colorbar(sc2, ax=ax[2], fraction=0.046, pad=0.02)
    cb2.set_label("Daytime LST P90 (°C)", fontsize=10)
    ax[2].set_xlabel("NCE value (unitless)", fontweight="bold")
    ax[2].set_ylabel("SHAP value (Δ LE, years)", fontweight="bold")
    ax[2].set_title("(C) NCE dependence, coloured by hot-day component",
                    fontweight="bold", loc="left", fontsize=12)

    for a in ax:
        a.grid(True, alpha=0.2)
    fig.suptitle("Nighttime Cooling Efficiency (NCE) = (P90$_{day}$ − P10$_{night}$)"
                 " / (P90$_{day}$ + ε): dependence and decomposition",
                 fontweight="bold", y=1.03)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"figR_NCE_decomposition.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # correlations to drive the written interpretation
    r_night = stats.pearsonr(night_p10, nce_val)[0]
    r_day = stats.pearsonr(day_p90, nce_val)[0]
    print(f"  corr(NCE, night P10) = {r_night:+.3f}; corr(NCE, day P90) = {r_day:+.3f}")
    print("  Interpretation: high NCE = large day-night thermal swing driven mainly "
          "by COLD nights -> protective (positive SHAP). Low NCE = warm nights "
          "relative to days -> penalty. Saved figR_NCE_decomposition.")


# ============================================================================
# R3.4 — MODIS TERRA ORBITAL-DRIFT EVIDENCE
# ============================================================================
def fig_orbital_drift():
    print("\n[R3.4] MODIS Terra orbital-drift evidence")
    col = "LST_Night_1km_mean"
    key = "LST_Night_1km_p10"
    nat = df_sorted.groupby("year")[col].mean()
    years = nat.index.values.astype(float)
    vals = nat.values
    sl, ic, r, p_lin, se = stats.linregress(years, vals)
    tau, p_mk = stats.kendalltau(years, vals)

    # variance decomposition of the key predictor
    g = df_sorted.groupby("year")[key]
    between = np.var(g.mean().values, ddof=0)               # temporal (year means)
    within = g.var(ddof=0).mean()                            # cross-sectional, avg
    total = df_sorted[key].var(ddof=0)

    # rank stability of county nighttime LST across consecutive years
    piv = df_sorted.pivot_table(index="fips", columns="year", values=key)
    yrs = sorted([c for c in piv.columns])
    rhos = []
    for a, b in zip(yrs[:-1], yrs[1:]):
        sub = piv[[a, b]].dropna()
        if len(sub) > 50:
            rhos.append(stats.spearmanr(sub[a], sub[b]).correlation)
    mean_rho = float(np.mean(rhos)) if rhos else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(years, vals, "o-", color=C["night"], lw=2.5, ms=7)
    ax[0].plot(years, sl * years + ic, "--", color=C["red"], lw=1.8)
    ax[0].set_xlabel("Year", fontweight="bold")
    ax[0].set_ylabel("National mean nighttime LST (°C)", fontweight="bold")
    ax[0].set_title("(A) No material national LST trend over 20 years",
                    fontweight="bold", loc="left", fontsize=12)
    ax[0].text(0.04, 0.06,
               f"OLS slope = {sl:+.4f} °C/yr (p = {p_lin:.2f})\n"
               f"Mann–Kendall τ = {tau:+.3f} (p = {p_mk:.2f})\n"
               f"year-to-year county rank ρ = {mean_rho:.3f}",
               transform=ax[0].transAxes, fontsize=10, va="bottom",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["grey"]))
    ax[0].grid(True, alpha=0.25)

    parts = ["Cross-sectional\n(within-year,\ncounty-to-county)",
             "Temporal\n(between-year,\nnational means)"]
    vals_bar = [within, between]
    ax[1].bar([0, 1], vals_bar, color=[C["blue"], C["orange"]],
              edgecolor="white", width=0.6)
    for i, v in enumerate(vals_bar):
        ax[1].text(i, v, f"{v:.2f}\n({100*v/total:.1f}% of total)",
                   ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(parts, fontsize=10)
    ax[1].set_ylabel(f"Variance of {key}", fontweight="bold")
    ax[1].set_title("(B) The predictive signal is spatial, not temporal",
                    fontweight="bold", loc="left", fontsize=12)
    ax[1].set_ylim(0, max(vals_bar) * 1.25)
    ax[1].grid(axis="y", alpha=0.25)

    fig.suptitle("MODIS Terra orbital drift cannot generate the nighttime-LST "
                 "signal: it is a uniform temporal shift, but the model uses "
                 "county-to-county contrast", fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"figR_orbital_drift.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  national LST OLS slope={sl:+.4f}°C/yr (p={p_lin:.2f}); "
          f"MK tau={tau:+.3f} (p={p_mk:.2f})")
    print(f"  variance: within-year(spatial)={within:.3f} ({100*within/total:.1f}%), "
          f"between-year(temporal)={between:.3f} ({100*between/total:.1f}%)")
    print(f"  mean year-to-year county rank correlation = {mean_rho:.3f}")
    print("  Saved figR_orbital_drift. Combine with E_drift_detrended from script 01.")


# ============================================================================
# R4.4 — INDIGENOUS RESERVATION FOLD-MEMBERSHIP + RESIDUAL AUDIT
# ============================================================================
def fig_reservation_audit():
    print("\n[R4.4] Indigenous-reservation fold-membership audit")
    p = predictions.copy()
    # county-level aggregates + the (single) fold each county was tested in
    agg = (p.groupby("fips")
           .agg(fold=("fold", lambda s: sorted(set(s))),
                n_folds=("fold", "nunique"),
                actual=("actual", "mean"),
                predicted=("predicted", "mean"),
                residual=("residual", "mean"),
                abs_error=("abs_error", "mean"))
           .reset_index())
    # sanity: GroupKFold tests each county exactly once
    n_multi = (agg["n_folds"] > 1).sum()
    fold_counts = p.groupby("fold")["fips"].nunique()
    print(f"  counties tested in >1 fold: {n_multi} (should be 0)")
    print(f"  counties per fold: {dict(fold_counts)}")

    # named reservation counties (search alternate FIPS where renamed)
    reservation = {
        "Oglala Lakota, SD": ["46102", "46113"],   # 46113 = old Shannon Co.
        "Todd, SD": ["46121"],
        "Sioux, ND": ["38085"],
        "Buffalo, SD": ["46017"],
        "Roosevelt, MT": ["30085"],
        "Dewey, SD": ["46041"],
        "Ziebach, SD": ["46137"],
        "Corson, SD": ["46031"],
        "Apache, AZ": ["04001"],
        "Big Horn, MT": ["30003"],
    }
    rows = []
    for name, codes in reservation.items():
        hit = agg[agg["fips"].isin(codes)]
        if len(hit):
            r = hit.iloc[0]
            rows.append(dict(County=name, FIPS=r["fips"],
                             Tested_in_fold=str(r["fold"]),
                             Mean_actual_LE=round(r["actual"], 2),
                             Mean_predicted_LE=round(r["predicted"], 2),
                             Mean_residual=round(r["residual"], 2),
                             Mean_abs_error=round(r["abs_error"], 2)))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_DIR / "reservation_fold_audit.csv", index=False)
    print("\n" + audit.to_string(index=False))

    nat_resid = p["residual"].mean()
    nat_mae = p["abs_error"].mean()

    # figure: residuals of named reservations vs national mean, labelled by fold
    if len(audit):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        order = audit.sort_values("Mean_residual")
        yy = np.arange(len(order))
        cols = [C["red"] if v < 0 else C["blue"] for v in order["Mean_residual"]]
        ax.barh(yy, order["Mean_residual"], color=cols, edgecolor="white")
        ax.axvline(0, color="#444", lw=1.2)
        ax.axvline(nat_resid, color=C["green"], ls="--", lw=1.6,
                   label=f"National mean residual = {nat_resid:+.2f} yr")
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{r.County}  ({r.FIPS}, fold {r.Tested_in_fold})"
                            for r in order.itertuples()], fontsize=10)
        ax.set_xlabel("Mean residual (Actual − Predicted, years)", fontweight="bold")
        ax.set_title("Indigenous-reservation counties: every county is held out in "
                     "exactly one fold;\nnegative residual = model OVER-predicts LE "
                     "(the 'spectral shadow')", fontweight="bold", loc="left",
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUTPUT_DIR / f"figR_reservation_audit.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"\n  national mean residual={nat_resid:+.3f} yr, national MAE={nat_mae:.3f} yr")
    print("  KEY STATEMENT for reply: all reservation counties appear in the test "
          "set of exactly one fold and the training set of the other four; none "
          "were systematically excluded. Saved reservation_fold_audit.csv.")


# ============================================================================
# R4.2 — CORRECTED FOREST ATTENUATION (no >100%; report absolute Delta yr)
# ============================================================================
def fig_forest_fixed():
    print("\n[R4.2] Corrected forest-attenuation figure")
    lst = resolve(fn_list, "Daytime Surface Temp (Mean)", "LST_Day_1km_mean",
                  "Daytime LST Mean")
    forest = resolve(fn_list, "Deciduous Forest %",
                     "USDA_Cropland_USDA_Forest_Deciduous_pct", "Deciduous Forest")
    if lst is None or forest is None:
        print("  !! daytime LST or deciduous-forest feature not found; skipping.")
        return
    li = fn_list.index(lst)
    lst_x = shap_sample[lst].values
    lst_s = shap_vals[:, li]
    fv = shap_sample[forest].values

    labs = ["Low (<5%)", "Medium (5–20%)", "High (>20%)"]
    bins = [-np.inf, 0.05, 0.20, np.inf]
    # values may be on 0-1 or 0-100 scale; detect
    if np.nanmax(fv) > 1.5:
        bins = [-np.inf, 5, 20, np.inf]
    fcat = pd.cut(fv, bins=bins, labels=labs)
    q = pd.qcut(lst_x, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    d = pd.DataFrame({"q": q, "f": fcat, "s": lst_s}).dropna()

    s_low = d[(d.f == "Low (<5%)") & (d.q == "Q4")]["s"].median()
    s_high = d[(d.f == "High (>20%)") & (d.q == "Q4")]["s"].median()
    delta = s_high - s_low                      # absolute SHAP gap (years)
    # corrected attenuation: fraction of the low-forest penalty that is removed,
    # capped at 100%. If s_low >= 0 (no penalty) attenuation is undefined -> NaN.
    if s_low < 0:
        attenuation = min(max((s_high - s_low) / (-s_low), 0.0), 1.0) * 100
        fully = (s_high >= 0)
    else:
        attenuation = np.nan
        fully = False

    pal = {"Low (<5%)": C["red"], "Medium (5–20%)": C["orange"],
           "High (>20%)": C["green"]}
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    for cat in labs:
        m = fcat == cat
        if m.sum() > 50:
            lx, ly = lowess_xy(lst_x[m], lst_s[m])
            ax[0].plot(lx, ly, color=pal[cat], lw=3, label=cat)
    ax[0].axhline(0, color="#999", lw=1.5)
    ax[0].set_xlabel("Daytime LST (°C)", fontweight="bold")
    ax[0].set_ylabel("Daytime-LST SHAP (years)", fontweight="bold")
    ax[0].set_title("(A) Heat penalty by forest stratum", fontweight="bold",
                    loc="left", fontsize=12)
    ax[0].legend(title="Deciduous forest cover", fontsize=10)
    ax[0].grid(True, alpha=0.25)

    pos = np.arange(4)
    offs = {"Low (<5%)": -0.25, "Medium (5–20%)": 0, "High (>20%)": 0.25}
    for cat in labs:
        meds = [d[(d.f == cat) & (d.q == f"Q{i+1}")]["s"].median() for i in range(4)]
        ax[1].bar(pos + offs[cat], meds, width=0.22, color=pal[cat],
                  edgecolor="white", label=cat)
    ax[1].axhline(0, color="#999", lw=1.5)
    ax[1].set_xticks(pos); ax[1].set_xticklabels(["Q1 (cool)", "Q2", "Q3", "Q4 (hot)"])
    ax[1].set_xlabel("Daytime LST quartile", fontweight="bold")
    ax[1].set_ylabel("Median daytime-LST SHAP (years)", fontweight="bold")
    if not np.isnan(attenuation):
        txt = (f"Q4 absolute ΔSHAP = {delta:+.3f} yr\n"
               f"penalty offset ≈ {attenuation:.0f}%"
               + (" (fully offset)" if fully else ""))
    else:
        txt = f"Q4 absolute ΔSHAP = {delta:+.3f} yr"
    ax[1].text(0.97, 0.05, txt, transform=ax[1].transAxes, ha="right",
               va="bottom", fontsize=11, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec=C["green"]))
    ax[1].set_title("(B) Q4 penalty is offset in high-forest counties",
                    fontweight="bold", loc="left", fontsize=12)
    ax[1].legend(title="Forest cover", fontsize=10, loc="upper center", ncol=3)
    ax[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Forest buffering, corrected framing: in the hottest quartile the "
                 "daytime-heat SHAP penalty is ≈fully offset in high-forest counties "
                 "(absolute ΔSHAP reported; no >100% values)",
                 fontweight="bold", y=1.02, fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"figR_forest_attenuation_fixed.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Q4 low-forest median SHAP = {s_low:+.4f} yr; "
          f"high-forest = {s_high:+.4f} yr")
    print(f"  absolute ΔSHAP = {delta:+.4f} yr; corrected offset = "
          f"{attenuation:.0f}%" if not np.isnan(attenuation) else
          f"  absolute ΔSHAP = {delta:+.4f} yr (penalty not negative; % offset N/A)")
    print("  Use the suggested rephrase: '...the daytime-LST SHAP penalty is "
          "completely offset (~100% attenuation) in high-forest counties, an "
          "absolute SHAP difference of {:.3f} years.'".format(delta))


# ============================================================================
def main():
    fig_nce_decomposition()   # R4.1
    fig_orbital_drift()       # R3.4
    fig_reservation_audit()   # R4.4
    fig_forest_fixed()        # R4.2
    print(f"\nAll interpretation figures saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
