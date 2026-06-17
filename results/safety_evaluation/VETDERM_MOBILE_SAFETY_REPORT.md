# VetDerm-Mobile safety-layer evaluation

Checkpoint: `vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single.pt`
Device: `cuda`

## Calibration
| split                |    n |       ece |   accuracy |   mean_confidence |   median_confidence |   brier_top1_surrogate |
|:---------------------|-----:|----------:|-----------:|------------------:|--------------------:|-----------------------:|
| val                  | 3254 | 0.0166379 |   0.966503 |          0.979288 |            0.992674 |              0.0271607 |
| internal_test        | 3259 | 0.0157898 |   0.965327 |          0.977856 |            0.992598 |              0.0277758 |
| development_external | 1351 | 0.0951658 |   0.802369 |          0.874493 |            0.987202 |              0.140605  |

## Domain gate
Threshold calibrated at 5.0% validation false-rejection rate: `0.868390`.
| split                |    n |   rejected_n |   reject_rate |   domain_score_mean |   domain_score_median |
|:---------------------|-----:|-------------:|--------------:|--------------------:|----------------------:|
| val                  | 3254 |          163 |     0.0500922 |            0.961128 |              0.975586 |
| internal_test        | 3259 |          151 |     0.0463332 |            0.962276 |              0.975946 |
| development_external | 1351 |          391 |     0.289415  |            0.896875 |              0.951149 |
| ood                  |   75 |           66 |     0.88      |            0.732745 |              0.744167 |

## Generated figures
- `figs/fig_vetderm_mobile_reliability.png/.pdf`
- `figs/fig_vetderm_mobile_domain_gate_scores.png/.pdf`
- `figs/fig_vetderm_mobile_gradcam_correct.png/.pdf`
- `figs/fig_vetderm_mobile_gradcam_wrong.png/.pdf`

## Publication caveat
These are post-hoc safety and interpretability analyses on the frozen VetDerm-Mobile checkpoint. The development external cohort remains a development-external cohort, not a fresh locked prospective validation set.