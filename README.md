# Predicting the Remaining Useful Life of Turbofan Engines with Machine Learning

Predictive-maintenance project on NASA's **C-MAPSS** turbofan degradation dataset (used in the PHM08 challenge). The goal is to predict the **Remaining Useful Life (RUL)** — the number of operating cycles left before failure — of aircraft engines from multivariate sensor time series.

## Dataset

Each engine is a time series of operational cycles with 3 operational settings and 21 sensor readings, run from a healthy state to failure (training set) or truncated before failure (test set, with the true RUL provided separately). There are four sub-datasets with different operating conditions and fault modes:

| Dataset | Train trajectories | Test trajectories | Conditions | Fault modes |
|---|---:|---:|---|---|
| FD001 | 100 | 100 | ONE (sea level) | ONE (HPC degradation) |
| FD002 | 260 | 259 | SIX | ONE (HPC degradation) |
| FD003 | 100 | 100 | ONE (sea level) | TWO (HPC + Fan degradation) |
| FD004 | 248 | 249 | SIX | TWO (HPC + Fan degradation) |

Raw text files live in `data/raw/`; `data/*.csv` holds the parsed/labeled versions (`labeled_train_*`, `labeled_test_*`, `rul_*`) used by the notebooks. All modeling work in this repo uses **FD001**.

## Repository structure

```
python/
  eda.ipynb                 Exploratory data analysis: dataset description, sensor behavior, operating settings
  models_search_grid.ipynb  Feature engineering + model selection (the core of the project)
  test.ipynb                Final train/test evaluation of the chosen models on held-out engines
outputs/                    Saved figures (sensor smoothing, slopes, NASA score, test results)
data/                       Raw and processed C-MAPSS data 
requirements.txt            Python dependencies
```

## Approach

The target is `rul_limited`: RUL capped at 125 cycles, since degradation is not linearly observable far from failure — a standard trick in this benchmark.

### Feature engineering (`models_search_grid.ipynb`)

- Sensors with little/no signal (constant or redundant) were excluded, keeping 14 informative sensors out of 21.
- **Moving average smoothing** (per engine, rolling mean) was tested with several window sizes to denoise raw sensor readings; a window of **3** gave the best cross-validated score.
- **Slope / trend features** (rate of change of each smoothed sensor over a window) were added on top of the smoothed signal; a window of **20 cycles** gave the largest improvement (e.g. HistGradientBoosting R² went from ~0.84 to ~0.89).
- **PCA** was explored as a dimensionality-reduction step but did not outperform using the engineered features directly, so it was dropped from the final pipeline.
- Feature importance (impurity-based for Random Forest, permutation-based for HistGradientBoosting) was used to sanity-check which engineered sensors matter most.

### Evaluation methodology

- **GroupKFold (5 splits)** cross-validation, grouping by `engine_id`, so cycles from the same engine never leak between train and validation folds.
- Metrics: **R²**, **RMSE**, and the **NASA PHM08 asymmetric score** (penalizes late predictions — predicting a longer life than reality — more heavily than early ones, since underestimating RUL is the safer error for maintenance).
- A `DummyRegressor` (mean prediction) is used as the baseline.

### Models compared

Ridge regression (with polynomial features), Random Forest, HistGradientBoosting regression were tuned via `GridSearchCV`; a Lasso and a survival-analysis (AFT) model were prototyped but left commented out as they didn't improve on the tree-based models.

Cross-validated results on FD001 training data (best configuration per model):

| Model | R² | RMSE | NASA score |
|---|---:|---:|---:|
| HistGradientBoosting | 0.888 | 13.93 | 3.61 |
| Random Forest | 0.882 | 14.32 | 3.73 |
| Ridge (poly degree 2) | 0.856 | 15.78 | 4.16 |
| Dummy (baseline) | 0.000 | 41.67 | 308.0 |

### Final evaluation (`test.ipynb`)

The three best models (Ridge, Random Forest, HistGradientBoosting) are refit on the full FD001 training set with the chosen features (smoothed sensors + 20-cycle slope) and scored on the **official held-out test set**, using only the last observed cycle of each engine (one prediction per engine — the standard PHM08 benchmark protocol):

| Model | RMSE | R² | NASA score |
|---|---:|---:|---:|
| Gradient Boosting | **13.28** | **0.890** | **2.70** |
| Random Forest | 13.85 | 0.881 | 2.90 |
| Ridge | 15.14 | 0.857 | 3.34 |

HistGradientBoosting is the best-performing model on the final test set, closely followed by Random Forest; both clearly outperform the linear Ridge baseline. Predicted-vs-true scatter plots and per-engine RUL trajectories (see `outputs/Test_results.png`) confirm the models track the degradation curve well, with the expected larger errors far from failure (where the RUL cap and the asymmetric NASA score matter most).

## Setup

```bash
pip install -r requirements.txt
```

Then run the notebooks in `python/` with Jupyter (`eda.ipynb` → `models_search_grid.ipynb` → `test.ipynb`).
