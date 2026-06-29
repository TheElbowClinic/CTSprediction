# CTS GROC Prediction (Exploratory Analysis)

Code supporting a hypothesis-generating analysis of predictors of Global Rating of Change (GROC) at 24 weeks in carpal tunnel syndrome (CTS).

## Main script

Run `CTS - 260528 GROC.py`.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (tested with Anaconda).

## How to run

```bash
python "CTS - 260528 GROC.py"
```

When prompted, select an Excel (`.xlsx`) file containing the study dataset. The script expects the column names defined in `src/data_config.py`.

Model settings (e.g. cluster size, model type, cross-validation folds) can be changed in the `CONFIG` dictionary in `src/data_config.py`.

## Analysis pipeline

1. Missing-data summary and MICE imputation (`miceforest`)
2. Spearman correlation and signal-to-noise ranking
3. Boruta feature selection
4. Leave-one-out personalised modelling using Gower-distance neighbour clusters
5. Comparison of model RMSE against a naive (mean) baseline (paired *t*-test)

## Outputs

- `results/` — model and naive residuals (`.txt`)
- `plots/` — decision-tree figures (when `model_name` is `"dt"`)
- Interactive figures are displayed during the run

## Data

Patient-level data are **not** included in this repository. Access is restricted due to privacy and ethics requirements. Contact the authors to request data availability.
