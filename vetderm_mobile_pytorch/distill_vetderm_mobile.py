#!/usr/bin/env python3
"""Distill VetDerm-Mobile teachers into a compact student.

The default path supports validation-driven, class-aware multi-teacher
distillation. Per-class teacher weights are estimated on the validation split,
so the development external cohort remains a reporting set rather than a model
selection source.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from vetderm_mobile_pytorch.data import VetDermFolderDataset, build_eval_transform, build_train_transform, collate, index_split_dir
from vetderm_mobile_pytorch.metrics import constrained_predictions, prediction_frame, save_classification_report, summarize_predictions
from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES
from vetderm_mobile_pytorch.train_vetderm_mobile import collect_predictions, parse_target_class_boost


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_mobile_checkpoint(path: Path, device: torch.device, pretrained: bool = False) -> VetDermMobileNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def parse_named_paths(raw: str) -> list[tuple[str, Path]]:
    teachers: list[tuple[str, Path]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid teacher item {item!r}; expected name=/path/to/checkpoint.pt")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid teacher item {item!r}; teacher name is empty")
        teachers.append((name, Path(path.strip())))
    if not teachers:
        raise ValueError("At least one teacher checkpoint is required")
    return teachers


def parse_teacher_priors(raw: str, teacher_names: list[str]) -> np.ndarray:
    priors = np.ones(len(teacher_names), dtype=np.float32)
    if raw.strip():
        name_to_idx = {name: idx for idx, name in enumerate(teacher_names)}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid prior item {item!r}; expected teacher=weight")
            name, value = item.split("=", 1)
            name = name.strip()
            if name not in name_to_idx:
                raise ValueError(f"Unknown teacher in --teacher-priors: {name!r}")
            weight = float(value.strip())
            if weight <= 0:
                raise ValueError(f"Teacher prior must be positive for {name!r}: {weight}")
            priors[name_to_idx[name]] = weight
    return priors


def parse_class_teacher_boosts(raw: str, teacher_names: list[str]) -> dict[int, dict[int, float]]:
    """Parse "Disease=teacher:multiplier,Disease=teacher:multiplier" boosts."""
    boosts: dict[int, dict[int, float]] = {}
    if not raw.strip():
        return boosts
    teacher_to_idx = {name: idx for idx, name in enumerate(teacher_names)}
    disease_to_idx = {name: idx for idx, name in enumerate(DISEASE_NAMES)}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Invalid class-teacher boost {item!r}; expected Disease Name=teacher:multiplier")
        disease_name, rhs = item.split("=", 1)
        teacher_name, value = rhs.split(":", 1)
        disease_name = disease_name.strip()
        teacher_name = teacher_name.strip()
        if disease_name not in disease_to_idx:
            raise ValueError(f"Unknown disease in --class-teacher-boost: {disease_name!r}")
        if teacher_name not in teacher_to_idx:
            raise ValueError(f"Unknown teacher in --class-teacher-boost: {teacher_name!r}")
        multiplier = float(value.strip())
        if multiplier <= 0:
            raise ValueError(f"Class-teacher boost must be positive for {item!r}")
        boosts.setdefault(disease_to_idx[disease_name], {})[teacher_to_idx[teacher_name]] = multiplier
    return boosts


def build_loader(frame: pd.DataFrame, transform, batch_size: int, num_workers: int, shuffle: bool, weighted: bool = False, target_class_boost: dict[int, float] | None = None):
    dataset = VetDermFolderDataset(frame, transform)
    sampler = None
    target_class_boost = target_class_boost or {}
    if weighted or target_class_boost:
        counts = frame["disease"].value_counts().to_dict()
        weights = frame["disease"].map(lambda x: (1.0 / counts[int(x)]) * target_class_boost.get(int(x), 1.0)).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate)


def evaluate_split(model: VetDermMobileNet, loader: DataLoader, device: torch.device, split_name: str, out_dir: Path | None = None) -> list[dict[str, float]]:
    arrays, paths = collect_predictions(model, loader, device)
    rows = []
    for mode in ["raw", "pred_species", "known_species"]:
        row = summarize_predictions(arrays, mode=mode)
        row["split"] = split_name
        rows.append(row)
    if out_dir is not None:
        pred_df = prediction_frame(arrays, paths)
        pred_df.to_csv(out_dir / f"predictions_{split_name}.csv", index=False)
        known_pred = constrained_predictions(arrays["disease_logits"], arrays["y_species"])
        save_classification_report(arrays["y_disease"], known_pred, out_dir / f"disease_report_{split_name}_known_species.txt")
    return rows


def validation_class_f1(model: VetDermMobileNet, loader: DataLoader, device: torch.device) -> np.ndarray:
    arrays, _ = collect_predictions(model, loader, device)
    y_true = arrays["y_disease"]
    y_pred = constrained_predictions(arrays["disease_logits"], arrays["y_species"])
    f1 = np.zeros(len(DISEASE_NAMES), dtype=np.float32)
    for class_idx in range(len(DISEASE_NAMES)):
        if np.any(y_true == class_idx):
            f1[class_idx] = float(f1_score(y_true == class_idx, y_pred == class_idx, zero_division=0))
    return f1


def build_class_teacher_weights(
    teachers: list[VetDermMobileNet],
    teacher_names: list[str],
    val_loader: DataLoader,
    device: torch.device,
    priors: np.ndarray,
    class_teacher_boosts: dict[int, dict[int, float]],
    power: float,
    smoothing: float,
    out_dir: Path,
) -> torch.Tensor:
    f1_rows = []
    all_f1 = []
    for teacher_name, teacher in zip(teacher_names, teachers):
        f1 = validation_class_f1(teacher, val_loader, device)
        all_f1.append(f1)
        for class_idx, score in enumerate(f1):
            f1_rows.append({"teacher": teacher_name, "class_index": class_idx, "disease": DISEASE_NAMES[class_idx], "validation_f1": float(score)})
    f1_matrix = np.stack(all_f1, axis=1)
    raw = np.power(np.clip(f1_matrix, 0.0, 1.0) + smoothing, power) * priors[None, :]
    for class_idx, teacher_boosts in class_teacher_boosts.items():
        for teacher_idx, multiplier in teacher_boosts.items():
            raw[class_idx, teacher_idx] *= multiplier
    weights = raw / raw.sum(axis=1, keepdims=True)
    pd.DataFrame(f1_rows).to_csv(out_dir / "teacher_validation_class_f1.csv", index=False)
    weight_rows = []
    for class_idx, disease in enumerate(DISEASE_NAMES):
        row = {"class_index": class_idx, "disease": disease}
        for teacher_idx, teacher_name in enumerate(teacher_names):
            row[f"weight_{teacher_name}"] = float(weights[class_idx, teacher_idx])
            row[f"validation_f1_{teacher_name}"] = float(f1_matrix[class_idx, teacher_idx])
        weight_rows.append(row)
    pd.DataFrame(weight_rows).to_csv(out_dir / "class_teacher_weights.csv", index=False)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def blend_teacher_logits(
    teacher_outputs: list[tuple[torch.Tensor, torch.Tensor]],
    disease_labels: torch.Tensor,
    class_teacher_weights: torch.Tensor,
    teacher_global_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    species_stack = torch.stack([sp for sp, _ in teacher_outputs], dim=1)
    disease_stack = torch.stack([dis for _, dis in teacher_outputs], dim=1)
    disease_weights = class_teacher_weights[disease_labels].unsqueeze(-1)
    species_weights = teacher_global_weights.view(1, -1, 1)
    blended_species = (species_stack * species_weights).sum(dim=1)
    blended_disease = (disease_stack * disease_weights).sum(dim=1)
    return blended_species, blended_disease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", default="", help="Backward-compatible single teacher checkpoint.")
    parser.add_argument("--teacher-checkpoints", default="", help="Comma-separated name=checkpoint entries for multi-teacher distillation.")
    parser.add_argument("--teacher-priors", default="", help="Optional comma-separated teacher=weight priors before class-wise normalization.")
    parser.add_argument("--class-teacher-boost", default="", help="Optional comma-separated Disease Name=teacher:multiplier boosts.")
    parser.add_argument("--split-root", default="/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v2_general_dog")
    parser.add_argument("--external-root", default="evaluation_data/development_external_2026_05_02")
    parser.add_argument("--output-dir", default="vetderm_mobile_results/vetderm_mobile_distilled_pt_21class_dogrobust")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--train-policy", default="internet_robust", choices=["standard", "internet_robust", "internet_robust_heavy"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight of soft distillation loss")
    parser.add_argument("--disease-weight", type=float, default=1.0)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--target-class-boost", default="", help="Comma-separated Disease Name=multiplier entries applied to the weighted sampler.")
    parser.add_argument("--teacher-weight-power", type=float, default=3.0, help="Sharper values give more weight to the best validation teacher per class.")
    parser.add_argument("--teacher-weight-smoothing", type=float, default=0.02, help="Small positive smoothing for classes with low validation F1.")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", default="", help="Optional student checkpoint used to initialize refinement distillation.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_root = Path(args.split_root)
    train_df = index_split_dir(split_root / "train", SPECIES_NAMES, DISEASE_NAMES)
    val_df = index_split_dir(split_root / "val", SPECIES_NAMES, DISEASE_NAMES)
    test_df = index_split_dir(split_root / "test", SPECIES_NAMES, DISEASE_NAMES)
    external_df = index_split_dir(Path(args.external_root), SPECIES_NAMES, DISEASE_NAMES)
    target_class_boost = parse_target_class_boost(args.target_class_boost)
    train_loader = build_loader(train_df, build_train_transform(args.img_size, args.train_policy), args.batch_size, args.num_workers, True, args.weighted_sampler, target_class_boost)
    val_loader = build_loader(val_df, build_eval_transform(args.img_size), args.batch_size, args.num_workers, False)
    test_loader = build_loader(test_df, build_eval_transform(args.img_size), args.batch_size, args.num_workers, False)
    external_loader = build_loader(external_df, build_eval_transform(args.img_size), args.batch_size, args.num_workers, False)

    if args.teacher_checkpoints:
        teacher_specs = parse_named_paths(args.teacher_checkpoints)
    elif args.teacher_checkpoint:
        teacher_specs = [("teacher", Path(args.teacher_checkpoint))]
    else:
        raise ValueError("Provide --teacher-checkpoints or --teacher-checkpoint")
    teacher_names = [name for name, _ in teacher_specs]
    teachers = [load_mobile_checkpoint(path, device).eval() for _, path in teacher_specs]
    priors = parse_teacher_priors(args.teacher_priors, teacher_names)
    class_teacher_boosts = parse_class_teacher_boosts(args.class_teacher_boost, teacher_names)
    class_teacher_weights = build_class_teacher_weights(
        teachers,
        teacher_names,
        val_loader,
        device,
        priors,
        class_teacher_boosts,
        args.teacher_weight_power,
        args.teacher_weight_smoothing,
        out_dir,
    )
    teacher_global_weights = torch.tensor(priors / priors.sum(), dtype=torch.float32, device=device)

    student = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=not args.no_pretrained).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(init_ckpt.get("model_state", init_ckpt))
        print(f"initialized student from checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    hard_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0.0
        for images, sp, dis, _ in tqdm(train_loader, desc="distill", leave=False):
            images = images.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            dis = dis.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                teacher_outputs = [teacher(images) for teacher in teachers]
                t_sp, t_dis = blend_teacher_logits(teacher_outputs, dis, class_teacher_weights, teacher_global_weights)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                s_sp, s_dis = student(images)
                hard = hard_loss(s_sp, sp) + args.disease_weight * hard_loss(s_dis, dis)
                soft = kd_loss(s_sp, t_sp, args.temperature) + kd_loss(s_dis, t_dis, args.temperature)
                loss = (1.0 - args.alpha) * hard + args.alpha * soft
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(images)
        arrays, _ = collect_predictions(student, val_loader, device)
        val = summarize_predictions(arrays, mode="known_species")
        row = {"epoch": epoch, "train_loss": total / len(train_loader.dataset), **{f"val_{k}": v for k, v in val.items() if k != "mode"}}
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        scheduler.step(val["macro_f1"])
        print(f"epoch {epoch:03d} loss={row['train_loss']:.4f} val_acc={val['disease_accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} lr={row['lr']:.2e}", flush=True)
        if val["macro_f1"] > best_macro_f1:
            best_macro_f1 = val["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": student.state_dict(),
                    "species_names": SPECIES_NAMES,
                    "disease_names": DISEASE_NAMES,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "best_val_macro_f1": best_macro_f1,
                    "teacher_names": teacher_names,
                    "teacher_checkpoints": {name: str(path) for name, path in teacher_specs},
                    "distillation_config": vars(args) | {
                        "train_size": int(len(train_df)),
                        "val_size": int(len(val_df)),
                        "test_size": int(len(test_df)),
                        "external_size": int(len(external_df)),
                        "target_class_boost_by_name": {DISEASE_NAMES[idx]: boost for idx, boost in target_class_boost.items()},
                    },
                },
                out_dir / "vetderm_mobile_distilled_pt_best.pt",
            )
            print(f"saved distilled checkpoint at epoch {epoch}", flush=True)
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"early stopping at epoch {epoch}: no val_macro_f1 improvement for {epochs_without_improvement} epochs", flush=True)
                break

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    ckpt = torch.load(out_dir / "vetderm_mobile_distilled_pt_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    all_rows = []
    for split_name, loader in [("val", val_loader), ("internal_test", test_loader), ("development_external", external_loader)]:
        all_rows.extend(evaluate_split(student, loader, device, split_name, out_dir))
    pd.DataFrame(all_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
