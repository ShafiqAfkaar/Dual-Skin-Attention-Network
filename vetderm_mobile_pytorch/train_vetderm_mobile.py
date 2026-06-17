#!/usr/bin/env python3
"""Train VetDerm-Mobile-PT on the final 21-class dog-robust split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from vetderm_mobile_pytorch.data import VetDermFolderDataset, build_eval_transform, build_train_transform, collate, index_split_dir
from vetderm_mobile_pytorch.metrics import constrained_predictions, prediction_frame, save_classification_report, summarize_predictions
from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_target_class_boost(raw: str) -> dict[int, float]:
    """Parse comma-separated class boosts, e.g. "Lumpy Skin=2.0,Skin Allergy in Cat=2.5"."""
    boosts: dict[int, float] = {}
    if not raw.strip():
        return boosts
    name_to_idx = {name: idx for idx, name in enumerate(DISEASE_NAMES)}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --target-class-boost item: {item!r}. Expected Disease Name=multiplier")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in name_to_idx:
            raise ValueError(f"Unknown disease in --target-class-boost: {name!r}")
        multiplier = float(value.strip())
        if multiplier <= 0:
            raise ValueError(f"Class boost must be positive for {name!r}: {multiplier}")
        boosts[name_to_idx[name]] = multiplier
    return boosts


def build_loader(
    frame: pd.DataFrame,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    weighted: bool = False,
    target_class_boost: dict[int, float] | None = None,
):
    dataset = VetDermFolderDataset(frame, transform)
    sampler = None
    target_class_boost = target_class_boost or {}
    if weighted or target_class_boost:
        counts = frame["disease"].value_counts().to_dict()
        weights = frame["disease"].map(lambda x: (1.0 / counts[int(x)]) * target_class_boost.get(int(x), 1.0)).to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_species, y_disease, species_logits, disease_logits, paths = [], [], [], [], []
    for images, sp, dis, batch_paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        sp_logits, dis_logits = model(images)
        y_species.append(sp.numpy())
        y_disease.append(dis.numpy())
        species_logits.append(sp_logits.cpu().numpy())
        disease_logits.append(dis_logits.cpu().numpy())
        paths.extend(batch_paths)
    return {
        "y_species": np.concatenate(y_species),
        "y_disease": np.concatenate(y_disease),
        "species_logits": np.concatenate(species_logits),
        "disease_logits": np.concatenate(disease_logits),
    }, paths


def evaluate_split(model, loader, device, split_name: str, out_dir: Path | None = None):
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


def train_one_epoch(model, loader, optimizer, scaler, device, species_loss_fn, disease_loss_fn, disease_weight: float):
    model.train()
    total_loss = 0.0
    for images, sp, dis, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        sp = sp.to(device, non_blocking=True)
        dis = dis.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            sp_logits, dis_logits = model(images)
            loss = species_loss_fn(sp_logits, sp) + disease_weight * disease_loss_fn(dis_logits, dis)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * len(images)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", default="/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v2_general_dog")
    parser.add_argument("--external-root", default="evaluation_data/development_external_2026_05_02")
    parser.add_argument("--output-dir", default="vetderm_mobile_results/vetderm_mobile_pt_21class_dogrobust")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--train-policy", default="internet_robust", choices=["standard", "internet_robust", "internet_robust_heavy"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--disease-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--target-class-boost", default="", help="Comma-separated Disease Name=multiplier entries applied to the weighted sampler.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after this many epochs without validation macro-F1 improvement. 0 disables early stopping.")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=not args.no_pretrained).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = init_ckpt.get("model_state", init_ckpt)
        model.load_state_dict(state)
        print(f"initialized from checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    species_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    disease_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    config = vars(args) | {
        "species_names": SPECIES_NAMES,
        "disease_names": DISEASE_NAMES,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "external_size": int(len(external_df)),
        "device": str(device),
        "target_class_boost_by_index": target_class_boost,
        "target_class_boost_by_name": {DISEASE_NAMES[idx]: boost for idx, boost in target_class_boost.items()},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, species_loss_fn, disease_loss_fn, args.disease_weight)
        val_rows = evaluate_split(model, val_loader, device, "val")
        val_known = next(row for row in val_rows if row["mode"] == "known_species")
        scheduler.step(val_known["macro_f1"])
        row = {"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val_known.items() if k not in {"mode", "split"}}}
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch:03d} loss={loss:.4f} val_acc={val_known['disease_accuracy']:.4f} val_macro_f1={val_known['macro_f1']:.4f} lr={row['lr']:.2e}", flush=True)
        if val_known["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_known["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "species_names": SPECIES_NAMES,
                    "disease_names": DISEASE_NAMES,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "best_val_macro_f1": best_macro_f1,
                },
                out_dir / "vetderm_mobile_pt_best.pt",
            )
            print(f"saved best checkpoint at epoch {epoch} with val_macro_f1={best_macro_f1:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"early stopping at epoch {epoch}: no val_macro_f1 improvement for {epochs_without_improvement} epochs", flush=True)
                break

    ckpt = torch.load(out_dir / "vetderm_mobile_pt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    all_rows = []
    for split_name, loader in [("val", val_loader), ("internal_test", test_loader), ("development_external", external_loader)]:
        all_rows.extend(evaluate_split(model, loader, device, split_name, out_dir))
    pd.DataFrame(all_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
