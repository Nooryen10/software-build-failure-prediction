# Intelligent Software Build Failure Prediction Framework

A machine learning project investigating data leakage in CI build failure prediction,
using the TravisTorrent dataset.

## Research Question
Prior work on build failure prediction reports very high accuracy (95%+), but often uses
features only available *after* a build completes (test results, build duration). This
project quantifies the performance drop when models are restricted to features genuinely
available *before* a build runs — a realistic, deployable prediction setting.

## Pipeline
1. `notebooks/01_data_loading_filtering.ipynb` — load and filter TravisTorrent to Ruby, resolved builds
2. `notebooks/02_feature_engineering.ipynb` — split into pre-execution vs. post-execution features
3. `notebooks/03_eda.ipynb` — class balance, distributions, correlations
4. `notebooks/04_modeling_leaky_baseline.ipynb` — train LR/RF/XGBoost on all features
5. `notebooks/05_modeling_clean_pipeline.ipynb` — train LR/RF/XGBoost on pre-execution features only
6. `notebooks/06_comparison_results.ipynb` — compare both pipelines, feature importance, final figures

## Key Result
| Model | F1 (Leaky) | F1 (Clean) | Drop |
|---|---|---|---|
| Logistic Regression | 0.517 | 0.331 | 35.9% |
| Random Forest | 0.799 | 0.181 | 77.3% |
| XGBoost | 0.751 | 0.356 | 52.5% |

All models show substantial performance degradation once post-execution (leaky) features
are removed, confirming that realistic pre-execution build failure prediction is a
meaningfully harder task than prior literature's reported accuracy suggests.

## Dataset
TravisTorrent (Beller et al., 2017) — see `data/README.md` for filtering details.

## Reproducing
Run notebooks 01–06 in order inside Google Colab, with this repository's folder
structure mounted via Google Drive. See `requirements.txt` for dependencies.
