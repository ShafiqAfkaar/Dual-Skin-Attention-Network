# VetDerm-Mobile PyTorch workflow

This folder replaces the old 19-class TensorFlow/Keras mobile experiments with a clean PyTorch workflow aligned to the final SGCA 21-class taxonomy.

## What is different from SGCA

VetDerm-Mobile is a lightweight deployment model, not an SGCA variant. It uses:

- EfficientNetV2-B0 backbone, trained in the current experiments at 384 x 384 input resolution
- channel attention and spatial attention over the final CNN feature map
- global/local feature fusion
- a species head for Cat/Cattle/Dog
- a species-assisted disease head using concatenated species hidden features
- 21 disease classes using the same final SGCA taxonomy

SGCA remains the stronger research model. VetDerm-Mobile should be positioned as the compact deployable model.

## Training root

The current 384-pixel training root is the final no-external-overlap dog-normal robustness split:

```text
/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly
```

This includes the dog-normal robustness images in the training protocol while keeping validation, internal test, and development external images separate.

## Train

```bash
./.venv-wsl/bin/python -m vetderm_mobile_pytorch.train_vetderm_mobile \
  --split-root "/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly" \
  --external-root evaluation_data/development_external_2026_05_02 \
  --output-dir vetderm_mobile_results/vetderm_mobile_pt_384_v7_safeadd_weighted \
  --img-size 384 \
  --train-policy internet_robust \
  --weighted-sampler \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 4
```

## Distill

The current distillation script supports class-aware multi-teacher distillation. Teacher weights are derived from validation split per-class F1, with optional manual boosts for classes where a specific teacher is known to be stronger. Do not use the development external cohort for teacher selection.

```bash
./.venv-wsl/bin/python -m vetderm_mobile_pytorch.distill_vetderm_mobile \
  --split-root "/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly" \
  --external-root evaluation_data/development_external_2026_05_02 \
  --output-dir vetderm_mobile_results/vetderm_mobile_distilled_multiteacher_384_v7 \
  --img-size 384 \
  --train-policy internet_robust \
  --weighted-sampler \
  --target-class-boost "Skin Allergy in Cat=1.8,Ringworm in Cat=1.2,Lumpy Skin=1.2" \
  --teacher-checkpoints "original=vetderm_mobile_results/vetderm_mobile_pt_384_v7_safeadd_weighted/vetderm_mobile_pt_best.pt,mild=vetderm_mobile_results/vetderm_mobile_pt_384_v7_targetboost_mild_lr5e6_es30/vetderm_mobile_pt_best.pt,heavy=vetderm_mobile_results/vetderm_mobile_pt_384_v7_targetboost_lr1e5_es30/vetderm_mobile_pt_best.pt" \
  --teacher-priors "original=1.0,mild=1.15,heavy=1.0" \
  --class-teacher-boost "Lumpy Skin=original:1.35,Skin Allergy in Cat=heavy:1.35" \
  --teacher-weight-power 3.0 \
  --teacher-weight-smoothing 0.02 \
  --temperature 3.0 \
  --alpha 0.70 \
  --epochs 25 \
  --early-stopping-patience 5 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4
```

The completed multi-teacher student checkpoint is:

```text
vetderm_mobile_results/vetderm_mobile_distilled_multiteacher_384_v7/vetderm_mobile_distilled_pt_best.pt
```

## Final single-model selection

All three teachers and the distilled student use the same 7.26-million-parameter VetDerm-Mobile architecture and are approximately 29 MB each. Distillation therefore compresses a multi-model ensemble into one checkpoint, but it does not reduce the size of an individual model.

The multi-teacher student reached 78.94% macro-F1 on the development external cohort. Additional hard-label refinement, teacher-guided refinement, and checkpoint interpolation did not close the gap without reducing validation performance. The strongest single checkpoint remained the mild target-boost model, with deterministic known-species performance of 80.16% accuracy, 82.05% macro-F1, and 95.04% top-3 accuracy on the development external cohort.

The final packaged checkpoint and evaluation artifacts are stored in:

```text
vetderm_mobile_results/vetderm_mobile_final_single_384_v7
```

Reproduce the deterministic evaluation with:

```bash
./.venv-wsl/bin/python -m vetderm_mobile_pytorch.evaluate_checkpoint \
  --checkpoint vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single.pt \
  --split-root "/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly" \
  --external-root evaluation_data/development_external_2026_05_02 \
  --output-dir vetderm_mobile_results/vetderm_mobile_final_single_384_v7 \
  --img-size 384 \
  --batch-size 32 \
  --num-workers 4
```

## Export ONNX

```bash
./.venv-wsl/bin/python -m vetderm_mobile_pytorch.export_onnx \
  --checkpoint vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single.pt \
  --output vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single_384.onnx \
  --img-size 384 \
  --opset 18
```

TFLite conversion should be done after ONNX validation through an ONNX-to-TensorFlow conversion workflow. PyTorch does not export directly to TFLite.
