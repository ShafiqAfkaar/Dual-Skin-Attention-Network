# VetDerm-Mobile TFLite Deployment Summary

Generated: 2026-06-12

## Objective

This report completes the first five deployment-readiness steps for the final VetDerm-Mobile 384 px PyTorch checkpoint:

1. Export LiteRT/TFLite FP16 and INT8 variants.
2. Verify prediction agreement and accuracy on validation, internal test, and development external splits.
3. Benchmark model size, CPU latency, peak runtime RSS, and throughput.
4. Select the preferred deployment artifact.
5. Audit whether a fresh locked external cohort is available for final one-time evaluation.

## Source Checkpoint

- PyTorch checkpoint: `vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single.pt`
- Input size: `384 x 384`
- Evaluation protocol: known-species constrained disease decoding.

## Exported TFLite Artifacts

| Variant | File | Size |
|---|---|---:|
| FP32 reference | `tflite/vetderm_mobile_final_single_384_fp32.tflite` | 27.81 MiB |
| FP16 | `tflite/vetderm_mobile_final_single_384_fp16.tflite` | 15.57 MiB |
| INT8 weight-only | `tflite/vetderm_mobile_final_single_384_int8_weightonly.tflite` | 9.95 MiB |

Dynamic INT8 and static INT8 PT2E test exports were also attempted during conversion testing, but they did not reduce file size meaningfully. They should not be used as the deployment artifacts.

## Accuracy and Agreement

The table below reports known-species constrained decoding. Full details are saved in `tflite/tflite_deployment_comparison.csv`.

| Variant | Split | n | Disease accuracy | Macro-F1 | Top-3 | PyTorch disease agreement |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | Validation | 3254 | 96.65% | 96.77% | 99.39% | 100.00% |
| FP16 | Internal test | 3259 | 96.53% | 96.82% | 99.11% | 100.00% |
| FP16 | Development external | 1351 | 80.01% | 81.90% | 95.04% | 99.85% |
| INT8 weight-only | Validation | 3254 | 96.68% | 96.80% | 99.35% | 99.78% |
| INT8 weight-only | Internal test | 3259 | 96.62% | 96.98% | 99.08% | 99.69% |
| INT8 weight-only | Development external | 1351 | 79.94% | 81.87% | 95.19% | 98.82% |

Interpretation:

- FP16 is the highest-fidelity export. It exactly matches PyTorch predictions on validation and internal test, and differs on only two development-external disease predictions.
- INT8 weight-only preserves the headline metrics while reducing the file size and improving CPU latency substantially.
- The development external macro-F1 change from FP16 to INT8 weight-only is negligible: 81.90% to 81.87%.

## CPU Benchmark

Benchmarks were run with LiteRT/TFLite XNNPACK CPU execution in the current WSL environment. Peak RSS includes Python, LiteRT/TensorFlow runtime, image loading, and evaluation overhead, so it should not be interpreted as mobile-only model memory.

| Variant | Mean latency | Median latency | p95 latency | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|
| FP16 | 112.6-113.4 ms/image | 112.4-113.0 ms/image | 115.5-117.9 ms/image | 8.8 images/s | 1179.6 MiB |
| INT8 weight-only | 40.5-40.7 ms/image | 40.2-40.5 ms/image | 42.9-43.3 ms/image | 24.6-24.7 images/s | 1193.3 MiB |

## Selected Deployment Artifact

Recommended deployment artifact:

`vetderm_mobile_results/vetderm_mobile_final_single_384_v7/tflite/vetderm_mobile_final_single_384_int8_weightonly.tflite`

Reason:

- Smallest final artifact: 9.95 MiB.
- Fastest CPU latency: about 40.5 ms/image in this environment.
- Development external metrics remain effectively unchanged relative to FP16.
- Known-species disease agreement remains high: 99.78% validation, 99.69% internal test, and 98.82% development external.

Keep FP16 as the high-fidelity fallback:

`vetderm_mobile_results/vetderm_mobile_final_single_384_v7/tflite/vetderm_mobile_final_single_384_fp16.tflite`

Use FP16 if a target runtime has poor INT8 weight-only kernel support or if exact PyTorch parity is more important than size and latency.

## Locked External Cohort Audit

No genuinely fresh locked external cohort was found locally for a one-time confirmatory evaluation.

Important context:

- `PROTOCOL_FROZEN_2026_05_02.md` explicitly states that `evaluation_data/development_external_2026_05_02` is a development/exploratory external cohort, not a fresh confirmatory locked external validation set.
- The folder named `sgca_generalization_results/2026-05-02_locked_external_selected_clean_standard` contains prior selected-clean external outputs, but it is tied to the same May 2026 external development workflow and should not be rebranded as untouched confirmatory validation.
- `dog_normal_finetune_datasets/prepared/healthy_challenge` is a diagnostic challenge set used during model development, so it is also not suitable as a final locked test.

Therefore, step 5 cannot honestly be completed with the current local data. The correct next action is to create a new locked cohort and evaluate it once after freezing the model and deployment artifact.

## Required Protocol for a Future Locked Cohort

Before reporting a locked external test:

1. Collect images from sources not used in training, fine-tuning, model selection, prompt tuning, threshold tuning, or previous error analysis.
2. Confirm veterinary labels independently before model evaluation.
3. Remove exact and near duplicates against train, validation, internal test, development external, healthy-dog challenge, and any fine-tuning additions.
4. Freeze the model, TFLite artifact, preprocessing, taxonomy, species-constrained decoding rule, and all thresholds before evaluation.
5. Run the locked cohort once and report accuracy, macro-F1, top-3 accuracy, per-species metrics, per-class metrics, and confidence intervals.
6. Do not use the locked result for another round of tuning.

## Generated Files

- Export script: `vetderm_mobile_pytorch/export_tflite_litert.py`
- Evaluation script: `vetderm_mobile_pytorch/evaluate_tflite_litert.py`
- Export summary: `tflite/tflite_export_summary.json`
- Consolidated comparison: `tflite/tflite_deployment_comparison.csv`
- FP16 evaluation: `tflite/eval_fp16/`
- INT8 weight-only evaluation: `tflite/eval_int8_weightonly/`

