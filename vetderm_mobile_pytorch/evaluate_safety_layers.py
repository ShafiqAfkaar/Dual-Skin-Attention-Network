#!/usr/bin/env python3
"""Post-hoc safety and interpretability evaluation for VetDerm-Mobile.

This script does not retrain the model. It computes:
- known-species ECE and reliability data;
- a feature-bank OOD/domain gate using fused VetDerm-Mobile features;
- Grad-CAM panels for correct and incorrect development-external predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_vetderm_mobile")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from vetderm_mobile_pytorch.data import (
    VALID_EXTS,
    VetDermFolderDataset,
    build_eval_transform,
    collate,
    index_split_dir,
)
from vetderm_mobile_pytorch.metrics import constrained_predictions
from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES, SPECIES_TO_DISEASE_INDICES


OKABE_ITO = {
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
}


class ImageOnlyDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.transform = transform
        self.paths = [
            path
            for path in sorted(self.root.rglob("*"), key=lambda p: str(p))
            if path.is_file() and path.suffix.lower() in VALID_EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def collate_image_only(batch):
    images, paths = zip(*batch)
    return torch.stack(images), list(paths)


@dataclass(frozen=True)
class SplitOutputs:
    paths: list[str]
    y_species: np.ndarray | None
    y_disease: np.ndarray | None
    species_logits: np.ndarray
    disease_logits: np.ndarray
    features: np.ndarray


def configure_style() -> None:
    for font_path in [
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]:
        if font_path.exists():
            try:
                from matplotlib import font_manager

                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass
    mpl.rcParams.update(
        {
            "figure.dpi": 900,
            "savefig.dpi": 900,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_model(checkpoint_path: Path, device: torch.device) -> VetDermMobileNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=False)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_labeled_outputs(model: VetDermMobileNet, loader: DataLoader, device: torch.device, desc: str) -> SplitOutputs:
    species_logits: list[np.ndarray] = []
    disease_logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    y_species: list[np.ndarray] = []
    y_disease: list[np.ndarray] = []
    paths: list[str] = []
    model.eval()
    for images, sp, dis, batch_paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        feat, fused = model.forward_features(images)
        sp_logits, dis_logits = model(images)
        species_logits.append(sp_logits.detach().cpu().numpy())
        disease_logits.append(dis_logits.detach().cpu().numpy())
        features.append(F.normalize(fused, dim=1).detach().cpu().numpy().astype(np.float32))
        y_species.append(sp.numpy())
        y_disease.append(dis.numpy())
        paths.extend(batch_paths)
    return SplitOutputs(
        paths=paths,
        y_species=np.concatenate(y_species),
        y_disease=np.concatenate(y_disease),
        species_logits=np.concatenate(species_logits),
        disease_logits=np.concatenate(disease_logits),
        features=np.concatenate(features),
    )


@torch.no_grad()
def collect_unlabeled_outputs(model: VetDermMobileNet, loader: DataLoader, device: torch.device, desc: str) -> SplitOutputs:
    species_logits: list[np.ndarray] = []
    disease_logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    paths: list[str] = []
    model.eval()
    for images, batch_paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        _, fused = model.forward_features(images)
        sp_logits, dis_logits = model(images)
        species_logits.append(sp_logits.detach().cpu().numpy())
        disease_logits.append(dis_logits.detach().cpu().numpy())
        features.append(F.normalize(fused, dim=1).detach().cpu().numpy().astype(np.float32))
        paths.extend(batch_paths)
    return SplitOutputs(
        paths=paths,
        y_species=None,
        y_disease=None,
        species_logits=np.concatenate(species_logits),
        disease_logits=np.concatenate(disease_logits),
        features=np.concatenate(features),
    )


def constrained_probabilities(disease_logits: np.ndarray, species_indices: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(disease_logits, dtype=np.float64)
    for i, species_idx in enumerate(species_indices.astype(int)):
        valid = np.asarray(SPECIES_TO_DISEASE_INDICES[SPECIES_NAMES[species_idx]], dtype=np.int64)
        logits = disease_logits[i, valid].astype(np.float64)
        logits -= logits.max()
        exp = np.exp(logits)
        probs[i, valid] = exp / exp.sum()
    return probs


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n:
            conf_mean = float(confidence[mask].mean())
            acc = float(correct[mask].mean())
            gap = abs(acc - conf_mean)
            ece += (n / len(confidence)) * gap
        else:
            conf_mean = float("nan")
            acc = float("nan")
            gap = float("nan")
        rows.append(
            {
                "bin": idx + 1,
                "confidence_low": lo,
                "confidence_high": hi,
                "n": n,
                "confidence_mean": conf_mean,
                "accuracy": acc,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def compute_split_calibration(split_name: str, out: SplitOutputs, n_bins: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    assert out.y_species is not None and out.y_disease is not None
    probs = constrained_probabilities(out.disease_logits, out.y_species)
    pred = probs.argmax(axis=1)
    confidence = probs[np.arange(len(pred)), pred]
    correct = pred == out.y_disease
    ece, reliability = expected_calibration_error(confidence, correct.astype(float), n_bins=n_bins)
    reliability.insert(0, "split", split_name)
    predictions = pd.DataFrame(
        {
            "path": out.paths,
            "true_species": [SPECIES_NAMES[i] for i in out.y_species],
            "true_disease": [DISEASE_NAMES[i] for i in out.y_disease],
            "pred_disease_known_species": [DISEASE_NAMES[i] for i in pred],
            "confidence_known_species": confidence,
            "correct_known_species": correct,
        }
    )
    summary = {
        "split": split_name,
        "n": int(len(pred)),
        "ece": ece,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "brier_top1_surrogate": float(np.mean((confidence - correct.astype(float)) ** 2)),
    }
    return summary, reliability, predictions


def max_similarity(query: np.ndarray, reference: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    out = np.empty(len(query), dtype=np.float32)
    reference_t = reference.T.astype(np.float32)
    for start in range(0, len(query), chunk_size):
        sims = query[start : start + chunk_size].astype(np.float32) @ reference_t
        out[start : start + chunk_size] = sims.max(axis=1)
    return out


def nearest_reference(query: np.ndarray, reference: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(query), dtype=np.float32)
    indices = np.empty(len(query), dtype=np.int64)
    reference_t = reference.T.astype(np.float32)
    for start in range(0, len(query), chunk_size):
        sims = query[start : start + chunk_size].astype(np.float32) @ reference_t
        local = sims.argmax(axis=1)
        rows = np.arange(len(local))
        scores[start : start + len(local)] = sims[rows, local]
        indices[start : start + len(local)] = local
    return scores, indices


def plot_reliability(reliability: pd.DataFrame, summary: pd.DataFrame, out_prefix: Path) -> None:
    configure_style()
    splits = ["val", "internal_test", "development_external"]
    labels = {"val": "Validation", "internal_test": "Internal test", "development_external": "Development external"}
    colors = {"val": OKABE_ITO["blue"], "internal_test": OKABE_ITO["green"], "development_external": OKABE_ITO["vermillion"]}
    fig, ax = plt.subplots(figsize=(4.3, 3.35))
    ax.plot([0, 1], [0, 1], color="#333333", linewidth=1.2, linestyle="--", label="Perfect calibration")
    for split in splits:
        data = reliability[(reliability["split"] == split) & reliability["n"].gt(0)]
        ece = float(summary.loc[summary["split"] == split, "ece"].iloc[0])
        ax.plot(
            data["confidence_mean"],
            data["accuracy"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            color=colors[split],
            label=f"{labels[split]} (ECE {ece:.3f})",
        )
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_ood_scores(score_frames: list[pd.DataFrame], threshold: float, out_prefix: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.25))
    colors = {
        "val": OKABE_ITO["blue"],
        "internal_test": OKABE_ITO["green"],
        "development_external": OKABE_ITO["orange"],
        "ood": OKABE_ITO["vermillion"],
    }
    labels = {
        "val": "Validation",
        "internal_test": "Internal test",
        "development_external": "Development external",
        "ood": "General-photo OOD",
    }
    bins = np.linspace(0.0, 1.0, 41)
    for frame in score_frames:
        split = str(frame["split"].iloc[0])
        ax.hist(
            frame["domain_score"],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            color=colors[split],
            label=labels[split],
        )
    ax.axvline(threshold, color="#111827", linewidth=1.4, linestyle="--", label=f"5% validation threshold ({threshold:.3f})")
    ax.set_xlabel("Nearest-neighbour feature similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def clean_label(label: str) -> str:
    return (
        label.replace("Foot and Mouth disease", "FMD")
        .replace("Fungal Infection in Dog", "Dog fungal")
        .replace("Dermatitis in Dog", "Dog dermatitis")
        .replace("Skin Allergy in Dog", "Dog allergy")
        .replace("Skin Allergy in Cat", "Cat allergy")
        .replace("Ringworm in Cat", "Cat ringworm")
        .replace("Normal Skin", "Cattle normal")
        .replace("Lumpy Skin", "Lumpy skin")
        .replace("Cat_normal", "Cat normal")
        .replace("Dog_normal", "Dog normal")
        .replace("scabies cat", "Cat scabies")
    )


def unnormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1).detach().cpu()
    return np.asarray(to_pil_image(image))


def gradcam_for_image(model: VetDermMobileNet, tensor: torch.Tensor, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.zero_grad(set_to_none=True)
    tensor = tensor.requires_grad_(True)
    feat, fused = model.forward_features(tensor)
    feat.retain_grad()
    species_feat = model.species_hidden(fused)
    disease_logits = model.disease_head(torch.cat([fused, species_feat], dim=1))
    score = disease_logits[0, target_idx]
    score.backward()
    grads = feat.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=1, keepdim=False))[0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    cam_np = cam.detach().cpu().numpy()
    image_np = unnormalize_tensor(tensor.detach()[0])
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((image_np.shape[1], image_np.shape[0]), Image.Resampling.BICUBIC)
    cam_np = np.asarray(cam_img).astype(np.float32) / 255.0
    return image_np, cam_np


def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    cmap = mpl.colormaps["jet"]
    heat = (cmap(cam)[..., :3] * 255).astype(np.float32)
    base = image.astype(np.float32)
    mask = np.clip((cam - 0.15) / 0.85, 0, 1)[..., None]
    overlay = base * (1 - alpha * mask) + heat * (alpha * mask)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def render_gradcam_panel(
    *,
    model: VetDermMobileNet,
    selected: pd.DataFrame,
    transform,
    device: torch.device,
    out_prefix: Path,
    pred_color: str,
    max_items: int,
) -> pd.DataFrame:
    configure_style()
    selected = selected.head(max_items).copy()
    cols = 5
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.28 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    records = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        ax = axes.flat[idx]
        image_path = Path(str(row["path"]))
        true_disease = str(row["true_disease"])
        pred_disease = str(row["pred_disease_known_species"])
        pred_idx = DISEASE_NAMES.index(pred_disease)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        image_np, cam = gradcam_for_image(model, tensor, pred_idx)
        overlay = overlay_cam(image_np, cam)
        ax.imshow(overlay)
        ax.text(
            0.5,
            1.12,
            f"True: {clean_label(true_disease)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.7,
            color=OKABE_ITO["blue"],
            fontweight="bold",
        )
        ax.text(
            0.5,
            1.03,
            f"Pred: {clean_label(pred_disease)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.7,
            color=pred_color,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.6)
            spine.set_linestyle(":")
        records.append(
            {
                "path": str(image_path),
                "true_species": row["true_species"],
                "true_disease": true_disease,
                "pred_disease_known_species": pred_disease,
                "confidence_known_species": float(row["confidence_known_species"]),
                "correct_known_species": bool(row["correct_known_species"]),
            }
        )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.035, hspace=0.24)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=900, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(records)


def pick_gradcam_examples(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct_rows = []
    for disease in sorted(predictions["true_disease"].unique()):
        rows = predictions[(predictions["true_disease"] == disease) & (predictions["correct_known_species"] == True)].copy()
        if not rows.empty:
            rows = rows.sort_values("confidence_known_species", ascending=False)
            correct_rows.append(rows.iloc[0])
    correct = pd.DataFrame(correct_rows)
    wrong = predictions[predictions["correct_known_species"] == False].copy()
    if not wrong.empty:
        wrong["priority"] = wrong["true_disease"].map({"Lumpy Skin": 0, "Skin Allergy in Cat": 1, "Cat_normal": 2}).fillna(3)
        wrong = wrong.sort_values(["priority", "confidence_known_species"], ascending=[True, False]).drop(columns=["priority"])
    return correct, wrong


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("vetderm_mobile_results/vetderm_mobile_final_single_384_v7/vetderm_mobile_final_single.pt"))
    parser.add_argument("--split-root", type=Path, default=Path("/mnt/e/ringworm detection project/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly"))
    parser.add_argument("--external-root", type=Path, default=Path("evaluation_data/development_external_2026_05_02"))
    parser.add_argument("--ood-root", type=Path, default=Path("ood_evaluation/picsum_384"))
    parser.add_argument("--output-dir", type=Path, default=Path("vetderm_mobile_results/vetderm_mobile_final_single_384_v7/safety_evaluation"))
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--false-reject-rate", type=float, default=0.05)
    parser.add_argument("--gradcam-items", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_data").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    transform = build_eval_transform(args.img_size)

    split_frames = {
        "train": index_split_dir(args.split_root / "train", SPECIES_NAMES, DISEASE_NAMES),
        "val": index_split_dir(args.split_root / "val", SPECIES_NAMES, DISEASE_NAMES),
        "internal_test": index_split_dir(args.split_root / "test", SPECIES_NAMES, DISEASE_NAMES),
        "development_external": index_split_dir(args.external_root, SPECIES_NAMES, DISEASE_NAMES),
    }
    outputs: dict[str, SplitOutputs] = {}
    for split_name, frame in split_frames.items():
        dataset = VetDermFolderDataset(frame, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
        outputs[split_name] = collect_labeled_outputs(model, loader, device, desc=f"{split_name}")

    ood_ds = ImageOnlyDataset(args.ood_root, transform)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_image_only)
    ood = collect_unlabeled_outputs(model, ood_loader, device, desc="ood")

    calibration_rows = []
    reliability_frames = []
    prediction_frames = {}
    for split_name in ["val", "internal_test", "development_external"]:
        summary, reliability, predictions = compute_split_calibration(split_name, outputs[split_name], args.ece_bins)
        calibration_rows.append(summary)
        reliability_frames.append(reliability)
        prediction_frames[split_name] = predictions
        predictions.to_csv(args.output_dir / "source_data" / f"calibrated_predictions_{split_name}.csv", index=False)

    calibration_summary = pd.DataFrame(calibration_rows)
    reliability_all = pd.concat(reliability_frames, ignore_index=True)
    calibration_summary.to_csv(args.output_dir / "source_data" / "ece_summary.csv", index=False)
    reliability_all.to_csv(args.output_dir / "source_data" / "reliability_bins.csv", index=False)
    plot_reliability(reliability_all, calibration_summary, args.output_dir / "figs" / "fig_vetderm_mobile_reliability")

    train_features = outputs["train"].features
    val_scores, val_nearest = nearest_reference(outputs["val"].features, train_features)
    threshold = float(np.quantile(val_scores, args.false_reject_rate))
    score_frames = []
    gate_summary_rows = []
    reference_species = np.asarray([SPECIES_NAMES[i] for i in outputs["train"].y_species])
    reference_disease = np.asarray([DISEASE_NAMES[i] for i in outputs["train"].y_disease])

    for split_name in ["val", "internal_test", "development_external"]:
        scores, nearest = nearest_reference(outputs[split_name].features, train_features)
        frame = pd.DataFrame(
            {
                "split": split_name,
                "path": outputs[split_name].paths,
                "domain_score": scores,
                "rejected": scores < threshold,
                "nearest_species": reference_species[nearest],
                "nearest_disease": reference_disease[nearest],
            }
        )
        frame.to_csv(args.output_dir / "source_data" / f"domain_gate_scores_{split_name}.csv", index=False)
        score_frames.append(frame[["split", "domain_score"]])
        gate_summary_rows.append(
            {
                "split": split_name,
                "n": int(len(frame)),
                "rejected_n": int(frame["rejected"].sum()),
                "reject_rate": float(frame["rejected"].mean()),
                "domain_score_mean": float(frame["domain_score"].mean()),
                "domain_score_median": float(frame["domain_score"].median()),
            }
        )

    ood_scores, ood_nearest = nearest_reference(ood.features, train_features)
    ood_frame = pd.DataFrame(
        {
            "split": "ood",
            "path": ood.paths,
            "domain_score": ood_scores,
            "rejected": ood_scores < threshold,
            "nearest_species": reference_species[ood_nearest],
            "nearest_disease": reference_disease[ood_nearest],
        }
    )
    ood_frame.to_csv(args.output_dir / "source_data" / "domain_gate_scores_ood.csv", index=False)
    score_frames.append(ood_frame[["split", "domain_score"]])
    gate_summary_rows.append(
        {
            "split": "ood",
            "n": int(len(ood_frame)),
            "rejected_n": int(ood_frame["rejected"].sum()),
            "reject_rate": float(ood_frame["rejected"].mean()),
            "domain_score_mean": float(ood_frame["domain_score"].mean()),
            "domain_score_median": float(ood_frame["domain_score"].median()),
        }
    )
    gate_summary = pd.DataFrame(gate_summary_rows)
    gate_summary.to_csv(args.output_dir / "source_data" / "domain_gate_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / "source_data" / "domain_gate_profile_frr05.npz",
        reference_features=train_features.astype(np.float16),
        reference_species=reference_species,
        reference_diseases=reference_disease,
        reference_paths=np.asarray(outputs["train"].paths, dtype=object),
        threshold=np.asarray(threshold, dtype=np.float32),
        false_reject_rate=np.asarray(args.false_reject_rate, dtype=np.float32),
        validation_scores=val_scores.astype(np.float32),
    )
    plot_ood_scores(score_frames, threshold, args.output_dir / "figs" / "fig_vetderm_mobile_domain_gate_scores")

    correct_examples, wrong_examples = pick_gradcam_examples(prediction_frames["development_external"])
    correct_source = render_gradcam_panel(
        model=model,
        selected=correct_examples,
        transform=transform,
        device=device,
        out_prefix=args.output_dir / "figs" / "fig_vetderm_mobile_gradcam_correct",
        pred_color=OKABE_ITO["green"],
        max_items=args.gradcam_items,
    )
    correct_source.to_csv(args.output_dir / "source_data" / "gradcam_correct_selected_images.csv", index=False)
    if not wrong_examples.empty:
        wrong_source = render_gradcam_panel(
            model=model,
            selected=wrong_examples,
            transform=transform,
            device=device,
            out_prefix=args.output_dir / "figs" / "fig_vetderm_mobile_gradcam_wrong",
            pred_color=OKABE_ITO["vermillion"],
            max_items=args.gradcam_items,
        )
        wrong_source.to_csv(args.output_dir / "source_data" / "gradcam_wrong_selected_images.csv", index=False)

    report = {
        "checkpoint": str(args.checkpoint),
        "split_root": str(args.split_root),
        "external_root": str(args.external_root),
        "ood_root": str(args.ood_root),
        "img_size": args.img_size,
        "device": str(device),
        "ece": calibration_summary.to_dict(orient="records"),
        "domain_gate_threshold": threshold,
        "domain_gate_target_validation_false_reject_rate": args.false_reject_rate,
        "domain_gate": gate_summary.to_dict(orient="records"),
        "gradcam_correct_n": int(len(correct_source)),
        "gradcam_wrong_n": int(len(wrong_examples.head(args.gradcam_items))),
    }
    (args.output_dir / "VETDERM_MOBILE_SAFETY_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# VetDerm-Mobile safety-layer evaluation",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Device: `{device}`",
        "",
        "## Calibration",
        calibration_summary.to_markdown(index=False),
        "",
        "## Domain gate",
        f"Threshold calibrated at {args.false_reject_rate:.1%} validation false-rejection rate: `{threshold:.6f}`.",
        gate_summary.to_markdown(index=False),
        "",
        "## Generated figures",
        "- `figs/fig_vetderm_mobile_reliability.png/.pdf`",
        "- `figs/fig_vetderm_mobile_domain_gate_scores.png/.pdf`",
        "- `figs/fig_vetderm_mobile_gradcam_correct.png/.pdf`",
        "- `figs/fig_vetderm_mobile_gradcam_wrong.png/.pdf`",
        "",
        "## Publication caveat",
        "These are post-hoc safety and interpretability analyses on the frozen VetDerm-Mobile checkpoint. "
        "The development external cohort remains a development-external cohort, not a fresh locked prospective validation set.",
    ]
    (args.output_dir / "VETDERM_MOBILE_SAFETY_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
