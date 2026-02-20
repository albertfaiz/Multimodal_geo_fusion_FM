The layout is given below for ease of use.

life-expectancy-multimodal-fusion/
│
├── README.md                        # Main project README
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
│
├── scripts/                         # Analysis scripts
│   ├── 01_ml_analysis_production.py
│   ├── 02_visualization_production.py
│   └── README.md                    # Scripts documentation
│
├── data/                            # Data files
│   ├── merged_agriculture_life_expectancy.csv
│   ├── data_dictionary.csv          # Feature descriptions
│   └── README.md                    # Data documentation
│
├── results/                         # Generated outputs
│   ├── ml_results.pkl               # Main results bundle
│   ├── predictions_all_folds.csv
│   ├── feature_importances.csv
│   ├── table1_cv_performance.csv
│   ├── table2_le_bracket.csv
│   ├── table3_temporal.csv
│   ├── table4_feature_importance.csv
│   ├── table5_regional.csv
│   ├── table6_residual_stats.csv
│   ├── table7a_best_counties.csv
│   ├── table7b_worst_counties.csv
│   └── README.md                    # Results documentation
│
├── figures/                         # Generated figures
│   ├── main/
│   │   ├── fig01_temporal_stability.pdf/.png
│   │   ├── fig02_physical_plausibility.pdf/.png
│   │   ├── ... (fig01-fig11)
│   │   └── fig11_state_performance.pdf/.png
│   ├── supplementary/
│   │   ├── figS01_correlation_matrix.pdf/.png
│   │   ├── ... (figS01-figS05)
│   │   └── figS05_shap_rankings_complete.pdf/.png
│   └── README.md                    # Figures documentation
│
├── docs/                            # Documentation
│   ├── VISUALIZATION_README.md
│   ├── CODE_README.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   └── TROUBLESHOOTING.md
│
└── manuscript/                      # LaTeX files
    ├── main.tex
    ├── abstract.tex
    ├── introduction.tex
    ├── methods.tex
    ├── results.tex
    ├── discussion.tex
    ├── conclusions.tex
    ├── supplementary_materials.tex
    ├── references.bib
    └── README.md
