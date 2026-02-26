# 🛰️ Satellite-Based Life Expectancy Prediction

**Multimodal Fusion of Remote Sensing and Agricultural Data for High-Resolution US County-Level Life Expectancy Surveillance**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Paper](https://img.shields.io/badge/Paper-Remote%20Sensing%20(MDPI)-green)](https://doi.org/PAPER_DOI)

> **Key finding:** Nighttime land surface temperature is a 4.2× stronger predictor of county-level longevity than daytime heat — identifying a 9.2°C overnight cooling threshold that separates counties gaining physiological recovery from those accumulating chronic thermal burden.

---

## 🌍 What This Does

This repository contains the complete pipeline to predict **life expectancy at birth** for every county in the continental United States using **only satellite-observable environmental features** — no census data, no sociodemographic surveys.

| Metric | Value |
|--------|-------|
| **Counties** | 3,108 CONUS |
| **Time span** | 2000–2019 (20 years) |
| **Features** | 435 across 11 data streams |
| **R²** | 0.604 ± 0.026 |
| **MAE** | 1.12 ± 0.02 years |
| **Fusion gain** | 31% over best single modality |
| **IHME benchmark** | 75% of census-model performance |

---

## 🗂️ Repository Structure

```
life-expectancy-remote-sensing/
│
├── 📁 gee/                          # Google Earth Engine extraction scripts
│   ├── 01_extract_modis_lst.js      # MODIS LST (daytime + nighttime)
│   ├── 02_extract_modis_ndvi.js     # MODIS NDVI/EVI
│   ├── 03_extract_landsat.js        # Landsat surface reflectance
│   ├── 04_extract_sentinel1.js      # Sentinel-1 SAR + GLCM textures
│   ├── 05_extract_sentinel2.js      # Sentinel-2 multispectral
│   ├── 06_extract_jrc_water.js      # JRC Global Surface Water
│   ├── 07_extract_dem.js            # Copernicus DEM
│   └── 08_extract_soil_moisture.js  # ESA CCI soil moisture
│
├── 📁 data_prep/                    # Feature engineering & assembly
│   ├── merge_modalities.py          # Join all data streams by FIPS + year
│   ├── feature_engineering.py       # 19 derived cross-modal features
│   ├── quality_control.py           # Winsorization, imputation, drift tests
│   └── livestock_interpolation.py   # FAO GLW3 temporal interpolation
│
├── 📁 models/                       # ML training pipelines
│   ├── train_single_modality.py     # Per-modality ablation models
│   ├── train_combined.py            # Full 435-feature fusion model
│   ├── hyperparams/                 # Per-modality optimized configs (.json)
│   └── cross_validation.py          # County-grouped 5-fold CV
│
├── 📁 shap/                         # SHAP interpretation pipeline
│   ├── run_shap.py                  # TreeSHAP computation (12hr parallel)
│   ├── shap_plots.py                # Beeswarm, dependence, waterfall
│   ├── threshold_detection.py       # LOWESS + derivative inflection points
│   └── interaction_analysis.py     # Forest×LST synergy quantification
│
├── 📁 figures/                      # All publication figures
│   ├── fig01_spatial_residuals/
│   ├── fig02_temporal_stability/
│   ├── fig03_shap_beeswarm/
│   ├── fig04_metabolic_breakpoints/
│   ├── fig04b_physics_coupling/
│   ├── fig05_forest_buffer/
│   ├── fig06_soil_gradient/
│   ├── fig07_ablation/
│   ├── fig08_bracket_waterfall/
│   ├── figA_nighttime_paradox/      # NEW: nighttime vs daytime comparison
│   ├── figB_urban_spectrum/         # NEW: development intensity gradient
│   └── figC_spatial_fidelity/       # NEW: actual vs predicted choropleth
│
├── 📁 paper/                        # LaTeX manuscript
│   ├── main.tex
│   ├── sections/
│   └── references_final_clean.bib
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🔬 The 19 Derived Cross-Modal Features

After combining the 416 base features from raw modalities, **19 engineered features** are computed during combined-model preprocessing to capture cross-modal interactions:

| # | Feature | Formula / Description |
|---|---------|----------------------|
| 1 | **Thermal Vegetation Index (TVI)** | Nighttime LST mean × (1 − NDVI mean) — heat burden in low-vegetation areas |
| 2 | **Diurnal Temperature Range** | Daytime LST mean − Nighttime LST mean — proxy for continentality & radiative cooling |
| 3 | **Nighttime Cooling Efficiency** | (Daytime LST 90th − Nighttime LST 10th) / Daytime LST 90th — fraction of peak heat shed overnight |
| 4 | **NDVI–LST Divergence** | NDVI std / (Daytime LST std + ε) — landscape thermal heterogeneity |
| 5 | **Agricultural Greenness Ratio** | NDVI mean / (Corn% + Soybean% + 0.01) — vegetation quality beyond monoculture |
| 6 | **Livestock Heat Exposure** | Cattle density × Nighttime LST mean — animal density under thermal stress |
| 7 | **Impervious Surface Heat Index** | (Med + High Intensity Dev%) × Nighttime LST mean — development thermal penalty |
| 8 | **Forest Heat Buffer Score** | Deciduous Forest% × max(0, Daytime LST mean − 20°C) — buffering activation above 20°C |
| 9 | **Soil Moisture Deficit** | max(0, 6.0 − Soil Moisture mean) — distance below optimal field capacity |
| 10 | **Soil Moisture Excess** | max(0, Soil Moisture mean − 8.5) — distance above waterlogging threshold |
| 11 | **Wetland Flood Risk Index** | (Woody Wetlands% + Herbaceous Wetlands%) × Soil Moisture mean |
| 12 | **SAR–NDVI Structural Consistency** | Pearson r(SAR VH, NDVI) across years — vegetation structure stability |
| 13 | **Water Permanence Index** | Permanent Water% / (Permanent + Seasonal Water% + ε) — hydrological reliability |
| 14 | **Elevation Thermal Modifier** | Nighttime LST mean − (DEM mean × 0.0065) — LST adjusted for adiabatic lapse rate |
| 15 | **Topographic Roughness × LST** | DEM std × Daytime LST std — thermal complexity in rugged terrain |
| 16 | **Livestock Species Diversity** | Shannon entropy across 8 FAO species densities — mono- vs poly-species farming |
| 17 | **Crop Diversity Index** | Shannon entropy across 30+ CDL crop types — monoculture vs rotation intensity |
| 18 | **Seasonal NDVI Amplitude** | NDVI 90th percentile − NDVI 10th percentile — growing season strength |
| 19 | **Cross-Sensor NDVI Consistency** | |MODIS NDVI mean − Landsat NDVI mean| — sensor cross-validation flag |

> These features are computed in `data_prep/feature_engineering.py`. They are **not used in single-modality ablation models** — only in the combined fusion model, which is why feature count per modality sums to 416, and 416 + 19 = 435.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate le-rs
```

### 2. Download Processed Features (Zenodo)
```bash
# Download pre-processed county-year feature matrix (~2.1 GB)
wget https://zenodo.org/record/XXXXXXX/files/county_features_2000_2019.csv.gz
wget https://zenodo.org/record/XXXXXXX/files/county_life_expectancy_ihme.csv
```

### 3. Train the Combined Model
```bash
python models/train_combined.py \
    --features data/county_features_2000_2019.csv.gz \
    --targets data/county_life_expectancy_ihme.csv \
    --output results/
```

### 4. Run SHAP Analysis
```bash
# Warning: ~12 hours on 8-core machine
python shap/run_shap.py \
    --model results/combined_rf_model.pkl \
    --features data/county_features_2000_2019.csv.gz \
    --n_sample 5000 \
    --output results/shap_values.pkl
```

### 5. Reproduce All Figures
```bash
python figures/generate_all.py --shap results/shap_values.pkl
```

---

## 📡 Data Sources

| Stream | Product | Resolution | Years | Features |
|--------|---------|------------|-------|----------|
| MODIS LST | MOD11A2 v6.1 | 1 km, 8-day | 2000–2019 | 14 |
| MODIS Vegetation | MOD13A1 v6.1 | 500 m, 16-day | 2000–2019 | 14 |
| USDA Agriculture | Cropland Data Layer | 30 m, annual | 2000–2019 | 135 |
| Landsat | Collection 2 Tier 1 | 30 m | 2000–2019 | 84 |
| Sentinel-1 SAR | GRD IW | 10 m | 2014–2019 | 42 |
| Sentinel-2 | Level-2A | 10–20 m | 2017–2019 | 84 |
| JRC Surface Water | v1.4 | 30 m, annual | 2000–2019 | 6 |
| Copernicus DEM | GLO-30 | 30 m, static | — | 7 |
| ESA CCI Soil Moisture | v6.1 | 0.25°, daily | 2000–2019 | 7 |
| FAO Livestock | GLW3 | 10 km | 2005/10/15 | 23 |
| **Derived** | Cross-modal engineering | — | — | **19** |
| **TOTAL** | | | | **435** |

All extraction scripts target **Google Earth Engine** (free academic access). Pre-extracted county-level matrices are available on Zenodo.

---

## 🗝️ Key Results

### The Nighttime Thermal Paradox
Nighttime LST percentiles (minimum overnight cooling opportunity) outperform all other predictors — including daytime heat, vegetation, and agricultural land use — by a factor of **4.2×** in cumulative SHAP importance.

```
Nighttime LST (all 7 features):  1.071 years cumulative |SHAP|
Daytime LST  (all 7 features):   0.255 years cumulative |SHAP|
                                  ─────────────────────────────
                                  4.2× nighttime dominance
```

**Policy threshold:** Counties where nighttime LST 10th percentile exceeds **≈9.2°C** are denied the overnight cooling window essential for cardiovascular and immune recovery.

### SHAP Feature Hierarchy (Top 10)
| Rank | Feature | |SHAP| (years) | Direction |
|------|---------|------------|-----------|
| 1 | Nighttime LST (10th pct) | 0.297 | Negative |
| 2 | Nighttime LST (25th pct) | 0.257 | Negative |
| 3 | Nighttime LST (Mean) | 0.172 | Negative |
| 4 | Nighttime LST (Median) | 0.143 | Negative |
| 5 | Developed (Med Intensity) % | 0.128 | **Positive** |
| 6 | Nighttime LST (75th pct) | 0.106 | Negative |
| 7 | Nighttime LST (90th pct) | 0.076 | Negative |
| 8 | NDVI (10th pct) | 0.072 | Mixed |
| 9 | Woody Wetlands % | 0.069 | Negative |
| 10 | Horse Density | 0.061 | Negative |

### Policy-Actionable Thresholds
- **9.2°C** nighttime LST → heat mitigation priority threshold
- **≈4%** corn county coverage → agricultural benefit saturation
- **473 head/km²** cattle density → CAFO oversight inflection
- **>20% forest cover** → 54% attenuation of heat-driven LE penalty

---

## 📖 Citation

```bibtex
@article{lary2024satellite,
  author  = {Lary, David J and [co-authors]},
  title   = {Multimodal Fusion of Remote Sensing and Agricultural Data
             for High-Resolution Life Expectancy Prediction},
  journal = {Remote Sensing},
  year    = {2024},
  volume  = {XX},
  number  = {XX},
  pages   = {XXXX},
  doi     = {10.3390/rsXXXXXXXX}
}
```

---

## 📦 Data Availability

Processed county-year feature matrices and model outputs are archived at:
**Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX**

Raw satellite data is freely accessible via:
- [Google Earth Engine](https://earthengine.google.com/) (account required)
- [NASA Earthdata](https://earthdata.nasa.gov/) (MODIS products)
- [USDA NASS](https://nassgeodata.gmu.edu/CropScape/) (Cropland Data Layer)
- [ESA Climate Office](https://climate.esa.int/en/projects/soil-moisture/) (CCI Soil Moisture)

---

## 🤝 Contributing

Issues, pull requests, and forks are welcome. If you use this pipeline for a new geography, please consider contributing your GEE extraction script to `gee/international/`.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

Life expectancy data from IHME is subject to their [terms of use](http://www.healthdata.org/data-tools-practices/data-practices/terms-and-conditions).
