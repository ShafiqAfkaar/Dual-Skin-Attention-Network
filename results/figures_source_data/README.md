# Figure Source Data

This folder contains CSV source data and report files used to audit DSAN / VetDerm-Mobile publication figures.

The CSV files support metric, calibration, decoding, subgroup, deployment, and figure-value checks. Raw images are not included in this package.

## Prediction files for the manuscript splits

| File | Rows | Content |
|---|---|---|
| `predictions_dsan_val.csv` | 3,254 | Validation split predictions |
| `predictions_dsan_internal_test.csv` | 3,259 | Internal test split predictions |
| `predictions_dsan_development_external.csv` | 1,351 | Development external cohort predictions |
| `confusion_matrix_development_external_full_support.csv` | 10 classes | Development external confusion matrix with full class support |

Image paths are relative to the split root. These files replace the earlier `predictions_vetderm_mobile_*.csv` files and the earlier `confusion_matrix_development_external.csv`, which came from a different split or had incomplete class support.
