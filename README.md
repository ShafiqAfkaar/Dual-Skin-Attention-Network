<div align="center">

# Dual Skin Attention Network

**Project website:** https://shafiqafkaar.github.io/Dual-Skin-Attention-Network/

### Lightweight multi-species veterinary dermatology classification and mobile deployment benchmark

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-DSAN-red.svg)](#model-architecture)
[![Backbone](https://img.shields.io/badge/backbone-EfficientNetV2--B0-0A7EA4.svg)](#model-architecture)
[![Status](https://img.shields.io/badge/status-research%20release-4B5563.svg)](#scientific-boundary)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**DSAN / VetDerm-Mobile** combines an EfficientNetV2-B0 backbone, sequential channel-spatial attention, global-local feature fusion, a species head, species-assisted disease decoding, and TensorFlow Lite deployment artifacts for compact veterinary dermatology image classification.

</div>

---

## Overview

This repository contains the GitHub-ready release artifacts for the **Dual Skin Attention Network (DSAN)** veterinary dermatology classifier described in the accompanying manuscript. The system classifies **21 dermatological categories** across **cats, cattle, and dogs** from RGB images.

The project is designed as an image-based research benchmark and lightweight screening prototype. It is **not** a veterinary diagnosis, treatment-prescription, or autonomous clinical decision system.

## Highlights

- Multi-species veterinary dermatology classification across cats, cattle, and dogs.
- Compact EfficientNetV2-B0 feature extractor.
- Sequential channel attention and spatial attention over final-stage image features.
- Global-local feature fusion for coarse context and localized lesion-pattern representation.
- Species head plus species-assisted disease head.
- Known-species constrained decoding to reduce biologically implausible predictions.
- PyTorch checkpoint and TensorFlow Lite INT8 weight-only deployment artifact.
- Publication figure panels and source-data CSVs for auditability.

## Model Architecture

![DSAN architecture](./assets/architecture/dsan_architecture_workflow.png?raw=true)

The implementation is in [`vetderm_mobile_pytorch/model.py`](vetderm_mobile_pytorch/model.py). In brief, DSAN uses a shared EfficientNetV2-B0 backbone followed by:

1. Channel attention over feature channels.
2. Spatial attention over lesion-relevant image regions.
3. Global-local feature fusion.
4. A 3-class species head.
5. A 21-class disease head receiving the fused image representation plus the species hidden representation.

The main manuscript disease metrics use known-species constrained decoding, where a supplied host species label restricts disease predictions to classes valid for that species.

## Final Manuscript Metrics

| Split | Disease accuracy | Macro-F1 | ROC-AUC | PR-AUC | Top-3 |
|---|---:|---:|---:|---:|---:|
| Validation | 96.65% | 96.91% | 0.996 | 0.977 | 99.21% |
| Internal test | 97.09% | 97.44% | 0.997 | 0.982 | 99.41% |
| Development external | 80.24% | 82.15% | 0.956 | 0.812 | 95.04% |

The development-external cohort is an internet-derived robustness cohort covering 10 of the 21 disease classes. It should not be interpreted as a locked prospective clinical validation set.

## Mobile Deployment Summary

The TensorFlow Lite / LiteRT INT8 weight-only artifact is included at:

```text
results/tflite/vetderm_mobile_final_single_384_int8_weightonly.tflite
```

In the tested WSL CPU/XNNPACK environment, the INT8 weight-only artifact was approximately 9.95 MiB and achieved about 40.5 ms per-image inference latency while retaining 81.87% development-external macro-F1. These values are reference engineering benchmarks, not confirmed Android-device latency or memory measurements.

## Repository Structure

```text
DSAN-main/
├── assets/
│   ├── architecture/             README architecture panel
│   └── figures/                  Publication-ready figure panels
├── data/                         Data access notes; raw images are not included
├── notebooks/                    Publication training notebook record
├── results/
│   ├── figures_source_data/      Figure source-data CSVs and figure report
│   ├── models/                   Final checkpoint and result summaries
│   └── tflite/                   TensorFlow Lite deployment artifacts
├── scripts/                      Training, evaluation, export, and plotting scripts
├── vetderm_mobile_pytorch/       Core DSAN / VetDerm-Mobile PyTorch code
├── LICENSE
├── README.md
├── REPRODUCE.md
└── requirements.txt
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The model code also depends on PyTorch, torchvision, timm, scikit-learn, pandas, numpy, Pillow, matplotlib, seaborn, and OpenCV for Grad-CAM/figure workflows. CUDA is recommended for full training and figure regeneration; CPU is sufficient for code inspection and lightweight inference tests.

## Checkpoints

The final PyTorch checkpoint is included at:

```text
results/models/dsan_vetderm_mobile/dsan_vetderm_mobile_best.pt
```

This checkpoint is approximately 29 MiB and is below GitHub's 100 MiB file limit.

## Data Availability

Raw images are **not included** in this GitHub-ready package. The internal training, validation, internal-test, and development-external images require the private image archive or a separate approved data release.

Expected local data layout for full reproduction after data access is granted:

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

The internal folder label `Cattles` is retained for checkpoint/code compatibility; manuscript-facing text uses `Cattle`.

## Reproducibility

The main notebook and scripts are:

```text
notebooks/train_dsan_vetderm_mobile.ipynb
scripts/train_vetderm_mobile.py
scripts/evaluate_checkpoint.py
scripts/export_tflite_litert.py
scripts/evaluate_tflite_litert.py
scripts/plot_vetderm_publication_figures.py
```

Additional details are provided in [REPRODUCE.md](REPRODUCE.md).

## Scientific Boundary

This repository is a research release. DSAN/VetDerm-Mobile provides image-classification outputs and ranked candidates for evaluation. It does not provide treatment plans, drug names, dosage guidance, prognosis, or definitive diagnostic statements. Veterinary review remains necessary, especially for zoonotic, contagious, herd-health, uncertain, or low-quality submissions.

## Citation

If you use this repository, please cite the accompanying manuscript when available.

```bibtex
@article{dsan_vetderm_mobile_2026,
  title   = {Optimized Dual Skin Attention Network and Multi-Species Veterinary Dermatology Classification Benchmark},
  author  = {Afkaar, Shafiq and collaborators},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## License

This repository is released under the [Apache License 2.0](LICENSE).
