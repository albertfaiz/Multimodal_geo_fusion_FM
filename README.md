# 🛰️ Satellite-Based Life Expectancy Prediction

**Multimodal Fusion of Remote Sensing and Agricultural Data for High-Resolution US County-Level Life Expectancy Surveillance**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Paper](https://img.shields.io/badge/Paper-Remote%20Sensing%20(MDPI)-green)](https://doi.org/PAPER_DOI)

> **Key finding:** The nighttime thermal environment is a 3.0× stronger predictor of county-level longevity than daytime heat. The model identifies a 9.2°C overnight cooling threshold that separates counties gaining physiological recovery from those accumulating chronic thermal burden.

---

## 🌍 What This Does

This repository contains the complete pipeline to predict **life expectancy at birth** for every county in the continental United States using **only satellite-observable environmental features** — no census data, no sociodemographic surveys.

| Metric | Value |
|--------|-------|
| **Counties** | 3,108 CONUS |
| **Time span** | 2000–2019 (20 years) |
| **Features** | 450 across 11 data streams |
| **R²** | 0.604 ± 0.043 |
| **MAE** | 1.12 ± 0.05 years |
| **Fusion gain** | 31% over best single modality |
| **IHME benchmark** | 75% of census-model performance |

---

## 🗂️ Repository Structure

```text
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
│   ├── feature_engineering.py       # 25 derived cross-modal features
│   ├── quality_control.py           # Winsorization, imputation, drift tests
│   └── livestock_interpolation.py   # FAO GLW3 temporal interpolation
│
├── 📁 models/                       # ML training pipelines
│   ├── train_single_modality.py     # Per-modality ablation models
│   ├── train_combined.py            # Full 450-feature fusion model
│   ├── hyperparams/                 # Per-modality optimized configs (.json)
│   └── cross_validation.py          # County-grouped 5-fold CV
│
├── 📁 shap/                         # SHAP interpretation pipeline
│   ├── run_shap.py                  # TreeSHAP computation (12hr parallel)
│   ├── shap_plots.py                # Beeswarm, dependence, waterfall
│   ├── threshold_detection.py       # LOWESS + derivative inflection points
│   └── interaction_analysis.py      # Forest×LST synergy quantification
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
│   ├── figA_nighttime_paradox/      
│   ├── figB_urban_spectrum/         
│   └── figC_spatial_fidelity/       
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

## 🔬 The 25 Derived Cross-Modal Features

To capture complex environmental interactions, **25 engineered features** are computed dynamically. These achieve an $R^2$ of 0.563 on their own, rivalling single-sensor optical arrays:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Thermal Vegetation Index (TVI)** | Nighttime LST mean × (1 − NDVI mean) — heat burden in low-vegetation areas |
| 2 | **Diurnal Temp Range** | Daytime LST mean − Nighttime LST mean — proxy for continentality & radiative cooling |
| 3 | **Nighttime Cooling Efficiency** | Fraction of peak heat shed overnight |
| 4 | **NDVI–LST Divergence** | Landscape thermal heterogeneity |
| 5 | **Ag Greenness Ratio** | Vegetation quality beyond monoculture coverage |
| 6 | **Livestock Heat Exposure** | Cattle density × Nighttime LST mean |
| 7 | **Impervious Surface Heat Index** | Development intensity thermal penalty |
| 8 | **Forest Heat Buffer Score** | Buffering activation specifically above 20°C |
| 9 | **Soil Moisture Deficit** | Distance below optimal field capacity |
| 10 | **Soil Moisture Excess** | Distance above waterlogging threshold |
| 11 | **Wetland Flood Risk Index** | Interaction of wetland area and soil moisture |
| 12 | **SAR–NDVI Structural Index** | Vegetation structure stability via radar-optical synthesis |
| 13 | **Water Permanence Index** | Hydrological reliability (Permanent vs. Seasonal) |
| 14 | **Elevation Thermal Mod** | LST adjusted for adiabatic lapse rate |
| 15 | **Topographic Roughness × LST** | Thermal complexity in rugged terrain |
| 16 | **Livestock Species Diversity** | Mono- vs poly-species farming |
| 17 | **Crop Diversity Index** | Monoculture vs rotation intensity |
| 18 | **Seasonal NDVI Amplitude** | Growing season strength |
| 19 | **Cross-Sensor NDVI Consistency** | Sensor cross-validation flag |
| 20 | **Wet Bulb Temperature Proxy** | Synthesis of humidity and thermal conditions |
| 21 | **Blue-Green Proximity Index** | Spatial relationship of water and vegetation |
| 22 | **High-Intensity Urban Stressor** | Extreme built-environment density |
| 23 | **Industrial Ag Proxy** | Scale of monoculture operations |
| 24 | **Livestock Pollution Load** | Aggregated metabolic waste proxy |
| 25 | **Landscape Complexity** | Spatial heterogeneity index |

> These features are computed in `data_prep/feature_engineering.py`. They are **not used in single-modality ablation models** — only in the combined fusion model.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate geo_ml
```

### 2. Download Processed Features (Zenodo)
```bash
# Download pre-processed county-year feature matrix
wget [https://zenodo.org/record/XXXXXXX/files/county_features_2000_2019.csv.gz](https://zenodo.org/record/XXXXXXX/files/county_features_2000_2019.csv.gz)
wget [https://zenodo.org/record/XXXXXXX/files/county_life_expectancy_ihme.csv](https://zenodo.org/record/XXXXXXX/files/county_life_expectancy_ihme.csv)
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
python shap/run_shap.py \
    --model results/combined_rf_model.pkl \
    --features data/county_features_2000_2019.csv.gz \
    --n_sample 5000 \
    --output results/shap_values.pkl
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
| FAO Livestock | GLW3 | 10 km | 2005/10/15 | 32 |
| **Derived** | Cross-modal engineering | — | — | **25** |
| **TOTAL** | | | | **450** |

---

## 🗝️ Key Results

### The Nighttime Thermal Paradox
Nighttime features heavily dominate the predictive hierarchy, outperforming daytime heat signals by a factor of **3.0×** in cumulative SHAP importance.

```text
Nighttime LST (all 7 channels):  0.660 years cumulative |SHAP|
Daytime LST (all 7 channels):    0.218 years cumulative |SHAP|
                                 ─────────────────────────────
                                 3.0× nighttime dominance
```

### SHAP Feature Hierarchy (Top 10)
| Rank | Feature | Mean \|SHAP\| (years) | Direction |
|------|---------|-----------------------|-----------|
| 1 | Nighttime Cooling Efficiency | 0.363 | Divergent |
| 2 | Nighttime LST (10th pct) | 0.202 | Negative |
| 3 | Nighttime LST (25th pct) | 0.163 | Negative |
| 4 | Nighttime LST (Mean) | 0.095 | Negative |
| 5 | Wetland Flood Risk Index | 0.083 | Negative |
| 6 | Developed (Med Intensity) % | 0.080 | Positive |
| 7 | Nighttime LST (Median) | 0.075 | Negative |
| 8 | Nighttime LST (75th pct) | 0.062 | Negative |
| 9 | Nighttime LST (90th pct) | 0.054 | Negative |
| 10 | Pig Density | ~0.050 | Positive (up to saturation) |

### Policy-Actionable Thresholds
- **9.2°C Nighttime LST:** Essential threshold for overnight physiological recovery.
- **501 head/km² Cattle Density:** Inflection point where generalized rural metrics transition to concentrated environmental burden.
- **8% Developed Open Space:** Point at which suburban park benefits saturate.
- **>20% Forest Cover:** Provides a 54% attenuation of heat-driven life expectancy penalties during severe thermal stress.
- **Soil Moisture 'Goldilocks Zone':** Peak agricultural productivity/health benefits occur at SM ≈ 5.7, turning sharply negative above the 8.5 waterlogging threshold.

---

## 📖 Citation

```bibtex
@article{lary2024satellite,
  author  = {Lary, David J and [co-authors]},
  title   = {Multimodal Fusion of Remote Sensing and Agricultural Data
             for High-Resolution Life Expectancy Prediction},
  journal = {Sensors},
  year    = {2026},
  volume  = {XX},
  number  = {XX},
  pages   = {XXXX},
  doi     = {10.3390/sXXXXXXXX}
}
```

---

## 🤝 Contributing

Issues, pull requests, and forks are welcome. If you use this pipeline for a new geography, please consider contributing your GEE extraction script to `gee/international/`.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

Life expectancy data from IHME is subject to their [terms of use](http://www.healthdata.org/data-tools-practices/data-practices/terms-and-conditions).
