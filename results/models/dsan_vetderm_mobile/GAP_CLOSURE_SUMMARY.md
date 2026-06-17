# VetDerm-Mobile external gap-closure summary

## Outcome

The strongest deployable single checkpoint is `vetderm_mobile_final_single.pt`, copied from the mild target-boost run. It uses the same 7.26-million-parameter VetDerm-Mobile architecture as each teacher and the distilled student. The model is approximately 29 MB in PyTorch format.

The final deterministic evaluation uses known-species decoding and the leakage-controlled v7 split.

| Split | Accuracy | Macro-F1 | Top-3 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 96.65% | 96.77% | 99.39% | 0.9960 | 0.9788 |
| Internal test | 96.53% | 96.82% | 99.11% | 0.9964 | 0.9783 |
| Development external | 80.16% | 82.05% | 95.04% | 0.9561 | 0.8122 |

## Improvement over the multi-teacher student

On the development external cohort, the final single model improved:

- accuracy from 76.98% to 80.16%: +3.18 percentage points;
- macro-F1 from 78.94% to 82.05%: +3.10 percentage points;
- top-3 accuracy from 94.52% to 95.04%: +0.52 percentage points;
- ROC-AUC from 0.9483 to 0.9561;
- PR-AUC from 0.7942 to 0.8122.

The largest class-level F1 gains were for healthy dog skin (0.79 to 0.95), canine fungal infection (0.83 to 0.93), canine dermatitis (0.86 to 0.91), and feline scabies (0.85 to 0.88). Feline ringworm and feline skin allergy remained weaker and should be prioritized in future data collection.

## Experiments not retained

1. Targeted hard-label refinement reduced validation macro-F1 from 96.81% to 96.63% and 96.48% in the first two epochs, so the run was stopped.
2. Single-teacher refinement reached 96.87% validation macro-F1 but only 78.62% development-external macro-F1.
3. Multi-teacher refinement initialized from the strongest teacher reduced validation macro-F1 to approximately 96.4%, so the run was stopped.
4. Weight interpolation between the student and strongest teacher degraded at intermediate mixing fractions. Validation selected the original student, whereas development-external performance selected the original teacher endpoint.

## Interpretation

Further distillation did not close the gap because the teachers and student have identical architecture and capacity. The defensible deployment choice is therefore the strongest single VetDerm-Mobile checkpoint, not the lower-performing distilled checkpoint. The development external cohort was used during model development, so its metrics must not be described as independent prospective validation.

To obtain a genuinely smaller model, the next stage should change the student architecture or apply post-training quantization. That experiment would require a new accuracy-size-latency comparison and should not reuse the current development external cohort as an untouched final test.

## Artifacts

- `vetderm_mobile_final_single.pt`: final PyTorch checkpoint.
- `metrics_summary.csv`: deterministic split-level metrics.
- `predictions_*.csv`: image-level predictions for each split.
- `disease_report_*_known_species.txt`: class-level reports.
- `vetderm_mobile_final_single_384.onnx` and `.onnx.data`: ONNX export; both files are required.
