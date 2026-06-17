# DSAN / VetDerm-Mobile Model Artifacts

This folder contains the final DSAN / VetDerm-Mobile PyTorch checkpoint and compact result summaries used for the manuscript.

Included files:

- `dsan_vetderm_mobile_best.pt`: final PyTorch checkpoint.
- `run_config.json`: final selected run configuration.
- `history.csv`: training history for the final selected run.
- `metrics_summary.csv`: validation, internal-test, and development-external metrics.
- `disease_report_*_known_species.txt`: per-class classification reports under known-species constrained decoding.
- `MOBILE_TFLITE_DEPLOYMENT_SUMMARY.md`: deployment summary for TFLite artifacts.
- `GAP_CLOSURE_SUMMARY.md`: optimisation-stage summary.

The manuscript-facing metrics should be checked against the figure source data in `results/figures_source_data/`.
