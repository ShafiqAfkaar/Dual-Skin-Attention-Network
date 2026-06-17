# Reproducibility Guide

## Environment

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

CUDA is recommended for full training and Grad-CAM/figure regeneration. CPU execution is suitable for reading code, inspecting the checkpoint structure, and running lightweight inference checks.

## Data Status

Raw images are intentionally not included in this GitHub-ready package. Full retraining or complete metric regeneration requires the private image archive or an approved public data release.

Expected local data layout after data access is granted:

```text
data/training_data_deduped_splits/seed42/
  train/
  val/
  test/

data/development_external_2026_05_02/
  Cat/
  Cattles/
  Dog/
```

The internal code uses `Cattles` as the cattle folder label for compatibility with saved artifacts; manuscript-facing text should use `Cattle`.

## Final Model

The release checkpoint is:

```text
results/models/dsan_vetderm_mobile/dsan_vetderm_mobile_best.pt
```

The final training metadata and history are included in the same folder:

```text
results/models/dsan_vetderm_mobile/run_config.json
results/models/dsan_vetderm_mobile/history.csv
results/models/dsan_vetderm_mobile/metrics_summary.csv
```

## Evaluation

The main evaluation script is:

```bash
python scripts/evaluate_checkpoint.py --help
```

Disease metrics in the manuscript use known-species constrained decoding unless otherwise stated. This means that the host species is supplied at evaluation/inference time and disease logits are restricted to classes valid for that species.

## TensorFlow Lite / LiteRT Export

The release includes the INT8 weight-only and FP16 TFLite artifacts:

```text
results/tflite/vetderm_mobile_final_single_384_int8_weightonly.tflite
results/tflite/vetderm_mobile_final_single_384_fp16.tflite
```

Export and benchmark scripts:

```bash
python scripts/export_tflite_litert.py --help
python scripts/evaluate_tflite_litert.py --help
```

The reported latency values were measured in a WSL CPU/XNNPACK engineering environment. They should be re-measured on target mobile hardware before any deployment claim.

## Figures

Publication figure panels are included under:

```text
assets/figures/
```

Figure source-data CSVs are included under:

```text
results/figures_source_data/
```

The source-data CSVs support audit of the plotted values, but raw images and private per-image source files are not part of this package.
