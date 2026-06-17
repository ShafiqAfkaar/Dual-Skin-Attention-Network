#!/usr/bin/env python3
"""Generate publication figures for the VetDerm-Mobile model.

The output mirrors the SGCA publication-figure style but uses a separate
VetDerm-Mobile palette and source-data package. The script does not train or
modify models. It loads the frozen final checkpoint, recomputes full prediction
arrays when needed, and plots verified metrics, deployment results, safety
layers, and Grad-CAM panels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_vetderm_mobile_publication")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import font_manager
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vetderm_mobile_pytorch.data import VetDermFolderDataset, build_eval_transform, collate, index_split_dir
from vetderm_mobile_pytorch.evaluate_safety_layers import (
    constrained_probabilities,
    gradcam_for_image,
    load_model,
)
from vetderm_mobile_pytorch.metrics import constrained_predictions, constrained_topk, softmax_np, summarize_predictions
from vetderm_mobile_pytorch.taxonomy import CLASS_TO_SPECIES, DISEASE_NAMES, SPECIES_NAMES


SPLITS = ["val", "internal_test", "development_external"]
SPLIT_DISPLAY = {
    "val": "Validation",
    "internal_test": "Internal test",
    "development_external": "Development external",
}
MODE_DISPLAY = {
    "raw": "Disease-only",
    "pred_species": "Predicted-species",
    "known_species": "Species-conditioned",
}

DISEASE_DISPLAY = {
    "Cat_normal": "Cat - Normal skin",
    "Dog_normal": "Dog - Normal skin",
    "Normal Skin": "Cattle - Normal skin",
    "Ringworm in Cat": "Ringworm in cat",
    "Ringworm in Dog": "Ringworm in dog",
    "Ringworm(cow)": "Ringworm in cattle",
    "Skin Allergy in Cat": "Skin allergy in cat",
    "Skin Allergy in Dog": "Skin allergy in dog",
    "Ear Mites in Cat": "Ear mites in cat",
    "Eye Infection in Cat": "Eye infection in cat",
    "Eye Infection in Dog": "Eye infection in dog",
    "Dermatitis in Dog": "Dermatitis in dog",
    "Fungal Infection in Dog": "Fungal infection in dog",
    "Hot Spots in Dog": "Hot spots in dog",
    "Mange in Dog": "Mange in dog",
    "Tick Infestation in Dog": "Tick infestation in dog",
    "Foot and Mouth disease": "Foot-and-mouth disease",
    "Lumpy Skin": "Lumpy skin disease",
    "papiloma": "Papilloma",
    "scabies cat": "Scabies in cat",
    "scabies cattle": "Scabies in cattle",
}

SHORT_LABEL = {
    "Cat_normal": "Cat normal",
    "Dog_normal": "Dog normal",
    "Normal Skin": "Cattle normal",
    "Ringworm in Cat": "Cat ringworm",
    "Ringworm in Dog": "Dog ringworm",
    "Ringworm(cow)": "Cattle ringworm",
    "Skin Allergy in Cat": "Cat allergy",
    "Skin Allergy in Dog": "Dog allergy",
    "Ear Mites in Cat": "Cat ear mites",
    "Eye Infection in Cat": "Cat eye infection",
    "Eye Infection in Dog": "Dog eye infection",
    "Dermatitis in Dog": "Dog dermatitis",
    "Fungal Infection in Dog": "Dog fungal",
    "Hot Spots in Dog": "Dog hot spots",
    "Mange in Dog": "Dog mange",
    "Tick Infestation in Dog": "Dog tick infestation",
    "Foot and Mouth disease": "FMD",
    "Lumpy Skin": "Lumpy skin",
    "papiloma": "Papilloma",
    "scabies cat": "Cat scabies",
    "scabies cattle": "Cattle scabies",
}

DISEASE_ORDER = [
    "Cat_normal",
    "Ringworm in Cat",
    "Skin Allergy in Cat",
    "scabies cat",
    "Ear Mites in Cat",
    "Eye Infection in Cat",
    "Dog_normal",
    "Dermatitis in Dog",
    "Fungal Infection in Dog",
    "Hot Spots in Dog",
    "Mange in Dog",
    "Ringworm in Dog",
    "Skin Allergy in Dog",
    "Tick Infestation in Dog",
    "Eye Infection in Dog",
    "Normal Skin",
    "Foot and Mouth disease",
    "Lumpy Skin",
    "Ringworm(cow)",
    "scabies cattle",
    "papiloma",
]

SPECIES_DISPLAY = {"Cat": "Cat", "Cattles": "Cattle", "Dog": "Dog"}

PALETTE = {
    "mobile": "#2A9D8F",
    "mobile_dark": "#264653",
    "mobile_light": "#A8DADC",
    "fp16": "#457B9D",
    "int8": "#E76F51",
    "raw": "#8D99AE",
    "pred_species": "#B56576",
    "known_species": "#2A9D8F",
    "external": "#F4A261",
    "ood": "#D62828",
    "grid": "#E5E7EB",
    "text_gray": "#4B5563",
    "black": "#111827",
    "true_blue": "#1D4ED8",
    "correct_green": "#0A7F4F",
    "wrong_red": "#C1121F",
}

PROBLEMATIC_GRADCAM_SUBSTRINGS = {
    # Manually identified from the first automatic Grad-CAM pass as containing
    # people, visible text overlays, or watermarks unsuitable for manuscript panels.
    "3f5b539d0f72",
    "415ec4276fb6",
    "9c300e3361df",
    "289b3c82d036",
    "425e757b121e",
    "f458253b3142",
    "eec213c9d43",
    "e3ec213c9d43",
    "063e9ad5fec8",
    "a32b71791f79",
    "a2fc93692725",
    "b87cba7dc510",
    "ecfd57a1496",
    "ecf5d57a1496",
    "f292d9f42ef7",
    "19baa091db3e",
    "c1d4f64de62b",
    "4c1e8bb4433a",
    "9b9431888bda",
    "cdeee1dcafd9",
    "2c228b7e9c33",
    "3da04f13e76e",
}

MANUAL_GRADCAM_SELECTION_SUBSTRINGS = {
    # Validation correct Grad-CAM curation: the highest-confidence dog tick
    # image contained visible writing. Use this clean close-up dog-fur example.
    "Tick Infestation in Dog": "5dfec71c3107__image_75.jpg",
}


def configure_style() -> None:
    for fp in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
    ]:
        if fp.exists():
            try:
                font_manager.fontManager.addfont(str(fp))
            except Exception:
                pass

    sns.set_style("white")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 900,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def disp_disease(name: str) -> str:
    return DISEASE_DISPLAY.get(str(name), str(name).replace("_", " "))


def short_label(name: str) -> str:
    return SHORT_LABEL.get(str(name), str(name).replace("_", " "))


def disp_species(name: str) -> str:
    return SPECIES_DISPLAY.get(str(name), str(name))


def order_diseases(names) -> list[str]:
    rank = {name: idx for idx, name in enumerate(DISEASE_ORDER)}
    return sorted(set(map(str, names)), key=lambda name: (rank.get(name, 999), name))


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def percent_axis(ax, axis: str = "y") -> None:
    fmt = mpl.ticker.PercentFormatter(xmax=1.0)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def save_figure(fig, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=900, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png}")


def arrays_to_npz(path: Path, paths: list[str], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray(paths, dtype=object),
        y_species=arrays["y_species"].astype(np.int64),
        y_disease=arrays["y_disease"].astype(np.int64),
        species_logits=arrays["species_logits"].astype(np.float32),
        disease_logits=arrays["disease_logits"].astype(np.float32),
    )


def load_arrays_npz(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    arrays = {
        "y_species": data["y_species"],
        "y_disease": data["y_disease"],
        "species_logits": data["species_logits"],
        "disease_logits": data["disease_logits"],
    }
    return [str(x) for x in data["paths"].tolist()], arrays


@torch.no_grad()
def collect_split_arrays(model, split_dir: Path, transform, batch_size: int, num_workers: int, device: torch.device):
    frame = index_split_dir(split_dir, SPECIES_NAMES, DISEASE_NAMES)
    dataset = VetDermFolderDataset(frame, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    y_species, y_disease, species_logits, disease_logits, paths = [], [], [], [], []
    model.eval()
    for images, sp, dis, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        sp_logits, dis_logits = model(images)
        y_species.append(sp.numpy())
        y_disease.append(dis.numpy())
        species_logits.append(sp_logits.detach().cpu().numpy())
        disease_logits.append(dis_logits.detach().cpu().numpy())
        paths.extend(batch_paths)
    arrays = {
        "y_species": np.concatenate(y_species),
        "y_disease": np.concatenate(y_disease),
        "species_logits": np.concatenate(species_logits),
        "disease_logits": np.concatenate(disease_logits),
    }
    return paths, arrays


def collect_or_load_all(args, source_dir: Path):
    arrays_by_split = {}
    paths_by_split = {}
    npz_dir = source_dir / "arrays"
    split_dirs = {
        "val": args.split_root / "val",
        "internal_test": args.split_root / "test",
        "development_external": args.external_root,
    }
    missing = [sn for sn in SPLITS if not (npz_dir / f"arrays_vetderm_mobile_{sn}.npz").exists()]
    if missing or args.force_recompute_arrays:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Collecting model outputs on {device} for {', '.join(missing or SPLITS)}")
        model = load_model(args.checkpoint, device)
        transform = build_eval_transform(args.img_size)
        for sn in SPLITS:
            npz_path = npz_dir / f"arrays_vetderm_mobile_{sn}.npz"
            if npz_path.exists() and not args.force_recompute_arrays:
                continue
            paths, arrays = collect_split_arrays(
                model=model,
                split_dir=split_dirs[sn],
                transform=transform,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            arrays_to_npz(npz_path, paths, arrays)

    for sn in SPLITS:
        paths, arrays = load_arrays_npz(npz_dir / f"arrays_vetderm_mobile_{sn}.npz")
        paths_by_split[sn] = paths
        arrays_by_split[sn] = arrays
    return paths_by_split, arrays_by_split


def predictions_from_arrays(paths: list[str], arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    y_sp = arrays["y_species"]
    y_dis = arrays["y_disease"]
    sp_logits = arrays["species_logits"]
    dis_logits = arrays["disease_logits"]
    sp_pred = sp_logits.argmax(axis=1)
    raw_pred = dis_logits.argmax(axis=1)
    pred_species_pred = constrained_predictions(dis_logits, sp_pred)
    known_pred = constrained_predictions(dis_logits, y_sp)
    known_probs = constrained_probabilities(dis_logits, y_sp)
    known_conf = known_probs[np.arange(len(known_pred)), known_pred]
    raw_probs = softmax_np(dis_logits)
    raw_conf = raw_probs[np.arange(len(raw_pred)), raw_pred]
    return pd.DataFrame(
        {
            "path": paths,
            "true_species": [SPECIES_NAMES[i] for i in y_sp],
            "pred_species": [SPECIES_NAMES[i] for i in sp_pred],
            "true_disease": [DISEASE_NAMES[i] for i in y_dis],
            "pred_disease_raw": [DISEASE_NAMES[i] for i in raw_pred],
            "pred_disease_pred_species": [DISEASE_NAMES[i] for i in pred_species_pred],
            "pred_disease_known": [DISEASE_NAMES[i] for i in known_pred],
            "known_confidence": known_conf,
            "raw_confidence": raw_conf,
            "correct_known": known_pred == y_dis,
        }
    )


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(np.unique(y_true).tolist())
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def bootstrap_metric(y_true, y_pred, metric_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.integers(0, n, n)
        vals[idx] = metric_fn(y_true[sample], y_pred[sample])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bin_one_vs_rest(y: np.ndarray, scores: np.ndarray, labels: list[int]):
    y_bin = np.zeros((len(y), len(labels)), dtype=np.int8)
    score = np.zeros_like(y_bin, dtype=float)
    for j, label in enumerate(labels):
        y_bin[:, j] = (y == label).astype(np.int8)
        score[:, j] = scores[:, label]
    return y_bin, score


def compute_auc_summary(y: np.ndarray, scores: np.ndarray):
    labels = sorted(np.unique(y).tolist())
    y_bin, score = bin_one_vs_rest(y, scores, labels)
    try:
        roc_macro = float(roc_auc_score(y_bin, score, average="macro", multi_class="ovr"))
    except Exception:
        roc_macro = float("nan")
    try:
        pr_macro = float(average_precision_score(y_bin, score, average="macro"))
    except Exception:
        pr_macro = float("nan")
    try:
        roc_micro = float(roc_auc_score(y_bin.ravel(), score.ravel()))
    except Exception:
        roc_micro = float("nan")
    try:
        pr_micro = float(average_precision_score(y_bin, score, average="micro"))
    except Exception:
        pr_micro = float("nan")
    return {
        "roc_auc_macro": roc_macro,
        "pr_auc_macro": pr_macro,
        "roc_auc_micro": roc_micro,
        "pr_auc_micro": pr_micro,
    }


def bootstrap_aucs(y: np.ndarray, scores: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    roc_vals = np.full(n_boot, np.nan)
    pr_vals = np.full(n_boot, np.nan)
    for idx in range(n_boot):
        sample = rng.integers(0, n, n)
        vals = compute_auc_summary(y[sample], scores[sample])
        roc_vals[idx] = vals["roc_auc_macro"]
        pr_vals[idx] = vals["pr_auc_macro"]
    return (
        float(np.nanpercentile(roc_vals, 2.5)),
        float(np.nanpercentile(roc_vals, 97.5)),
        float(np.nanpercentile(pr_vals, 2.5)),
        float(np.nanpercentile(pr_vals, 97.5)),
    )


def compute_summary(arrays_by_split: dict[str, dict[str, np.ndarray]], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for sn, arrays in arrays_by_split.items():
        y = arrays["y_disease"]
        sp = arrays["y_species"]
        logits = arrays["disease_logits"]
        raw_scores = softmax_np(logits)
        sp_pred = arrays["species_logits"].argmax(axis=1)
        pred_by_mode = {
            "raw": logits.argmax(axis=1),
            "pred_species": constrained_predictions(logits, sp_pred),
            "known_species": constrained_predictions(logits, sp),
        }
        for mode in ["raw", "pred_species", "known_species"]:
            row = summarize_predictions(arrays, mode=mode)
            row.update(
                {
                    "model_key": "vetderm_mobile",
                    "model_name": "VetDerm-Mobile",
                    "split": sn,
                    "split_name": SPLIT_DISPLAY[sn],
                    "mode_name": MODE_DISPLAY[mode],
                }
            )
            pred = pred_by_mode[mode]
            row["disease_accuracy_ci_low"], row["disease_accuracy_ci_high"] = bootstrap_metric(
                y, pred, accuracy_score, n_boot=n_boot, seed=seed + len(rows)
            )
            row["macro_f1_ci_low"], row["macro_f1_ci_high"] = bootstrap_metric(
                y, pred, macro_f1_present, n_boot=n_boot, seed=seed + 100 + len(rows)
            )
            aucs = compute_auc_summary(y, raw_scores)
            row.update(aucs)
            if mode == "known_species":
                roc_lo, roc_hi, pr_lo, pr_hi = bootstrap_aucs(y, raw_scores, n_boot=max(100, n_boot // 2), seed=seed + 200 + len(rows))
                row["roc_auc_macro_ci_low"] = roc_lo
                row["roc_auc_macro_ci_high"] = roc_hi
                row["pr_auc_macro_ci_low"] = pr_lo
                row["pr_auc_macro_ci_high"] = pr_hi
            else:
                row["roc_auc_macro_ci_low"] = np.nan
                row["roc_auc_macro_ci_high"] = np.nan
                row["pr_auc_macro_ci_low"] = np.nan
                row["pr_auc_macro_ci_high"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion(predictions: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    for sn, df in predictions.items():
        present = order_diseases(df["true_disease"].unique()) if sn == "development_external" else order_diseases(DISEASE_NAMES)
        labels = [disp_disease(name) for name in present]
        cm = confusion_matrix(df["true_disease"], df["pred_disease_known"], labels=present, normalize="true") * 100.0
        figsize = (7.5, 7.5) if sn == "development_external" else (12.0, 12.0)
        annot_size = 16 if sn == "development_external" else 12
        tick_size = 11 if sn == "development_external" else 10
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            ax=ax,
            cmap=sns.light_palette(PALETTE["mobile"], as_cmap=True),
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": annot_size, "weight": "bold", "family": "Arial"},
            cbar_kws={"shrink": 0.7, "pad": 0.02},
        )
        ax.set_title(f"VetDerm-Mobile - {SPLIT_DISPLAY[sn]} (species-conditioned, %)", weight="bold", fontsize=12, pad=10)
        ax.set_xlabel("Predicted disease", weight="bold")
        ax.set_ylabel("True disease", weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=tick_size, weight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_size, weight="bold")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("Row-normalised %", fontsize=10, weight="bold")
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_confusion_{sn}")


def per_class_f1_with_ci(y_true, pred, classes, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = np.zeros(len(classes), dtype=float)
    boot = np.zeros((n_boot, len(classes)), dtype=float)
    for j, cls in enumerate(classes):
        obs[j] = f1_score((y_true == cls).astype(int), (pred == cls).astype(int), zero_division=0)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y_true[idx]
        pi = pred[idx]
        for j, cls in enumerate(classes):
            boot[b, j] = f1_score((yi == cls).astype(int), (pi == cls).astype(int), zero_division=0)
    return obs, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def per_class_auc_with_ci(y_true, scores, classes, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = np.full(len(classes), np.nan)
    boot = np.full((n_boot, len(classes)), np.nan)
    for j, cls in enumerate(classes):
        yb = (y_true == cls).astype(int)
        if 0 < yb.sum() < n:
            fpr, tpr, _ = roc_curve(yb, scores[:, cls])
            obs[j] = auc(fpr, tpr)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y_true[idx]
        si = scores[idx]
        for j, cls in enumerate(classes):
            yb = (yi == cls).astype(int)
            if 0 < yb.sum() < n:
                try:
                    fpr, tpr, _ = roc_curve(yb, si[:, cls])
                    boot[b, j] = auc(fpr, tpr)
                except Exception:
                    pass
    return obs, np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def plot_forest(
    *,
    fig_dir: Path,
    name: str,
    title: str,
    metric_label: str,
    diseases: list[str],
    values: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    support: list[int],
    xlim: tuple[float, float],
    xticks: list[float],
    x_is_percent: bool,
) -> None:
    labels = [disp_disease(disease) for disease in diseases]
    h = max(3.5, 0.42 * len(diseases) + 1.2)
    fig, ax = plt.subplots(figsize=(6.4, h))
    y = np.arange(len(diseases))
    valid = ~np.isnan(values)
    ax.hlines(y[valid], lows[valid], highs[valid], color=PALETTE["mobile"], linewidth=1.7, alpha=0.9)
    ax.scatter(
        values[valid],
        y[valid],
        color=PALETTE["mobile"],
        s=46,
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label="VetDerm-Mobile",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, weight="bold")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    if x_is_percent:
        percent_axis(ax, axis="x")
    ax.set_xlabel(metric_label, weight="bold")
    ax.set_title(title, weight="bold", fontsize=11)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    ax.invert_yaxis()
    sns.despine(ax=ax)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"n = {n}" for n in support], fontsize=8, color=PALETTE["text_gray"])
    for spine in ("top", "right", "bottom", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, fig_dir, name)


def plot_per_class_figures(arrays_by_split, fig_dir: Path, source_dir: Path, n_boot: int, seed: int) -> None:
    for sn, arrays in arrays_by_split.items():
        y = arrays["y_disease"]
        known_pred = constrained_predictions(arrays["disease_logits"], arrays["y_species"])
        raw_scores = softmax_np(arrays["disease_logits"])
        present = order_diseases([DISEASE_NAMES[i] for i in np.unique(y)] if sn == "development_external" else DISEASE_NAMES)
        class_idx = np.asarray([DISEASE_NAMES.index(d) for d in present], dtype=int)
        support = [int((y == idx).sum()) for idx in class_idx]

        obs, lo, hi = per_class_f1_with_ci(y, known_pred, class_idx, n_boot=n_boot, seed=seed)
        plot_forest(
            fig_dir=fig_dir,
            name=f"fig_vetderm_mobile_per_disease_f1_{sn}",
            title=f"{SPLIT_DISPLAY[sn]} - per-disease F1",
            metric_label="Disease F1-score (95% CI)",
            diseases=present,
            values=obs,
            lows=lo,
            highs=hi,
            support=support,
            xlim=(0, 1.10),
            xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            x_is_percent=True,
        )
        pd.DataFrame(
            {
                "split": sn,
                "model": "VetDerm-Mobile",
                "disease": present,
                "f1": obs,
                "ci_low": lo,
                "ci_high": hi,
                "support": support,
            }
        ).to_csv(source_dir / f"per_disease_f1_{sn}.csv", index=False)

        obs, lo, hi = per_class_auc_with_ci(y, raw_scores, class_idx, n_boot=max(100, n_boot // 2), seed=seed + 1000)
        plot_forest(
            fig_dir=fig_dir,
            name=f"fig_vetderm_mobile_per_class_auc_{sn}",
            title=f"{SPLIT_DISPLAY[sn]} - per-class ROC-AUC",
            metric_label="One-vs-rest ROC-AUC (95% CI)",
            diseases=present,
            values=obs,
            lows=lo,
            highs=hi,
            support=support,
            xlim=(0.5, 1.05),
            xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            x_is_percent=False,
        )
        pd.DataFrame(
            {
                "split": sn,
                "model": "VetDerm-Mobile",
                "disease": present,
                "roc_auc": obs,
                "ci_low": lo,
                "ci_high": hi,
                "support": support,
            }
        ).to_csv(source_dir / f"per_class_auc_{sn}.csv", index=False)


def micro_roc_with_band(y, scores, n_boot: int, seed: int):
    labels = sorted(np.unique(y).tolist())
    y_bin, score = bin_one_vs_rest(y, scores, labels)
    fpr_obs, tpr_obs, _ = roc_curve(y_bin.ravel(), score.ravel())
    auc_obs = auc(fpr_obs, tpr_obs)
    grid = np.linspace(0, 1, 200)
    rng = np.random.default_rng(seed)
    interp = np.zeros((n_boot, len(grid)))
    aucs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, sc = bin_one_vs_rest(y[idx], scores[idx], labels)
        fpr, tpr, _ = roc_curve(yb.ravel(), sc.ravel())
        interp[b] = np.interp(grid, fpr, tpr)
        interp[b, 0] = 0.0
        aucs[b] = auc(fpr, tpr)
    return grid, np.interp(grid, fpr_obs, tpr_obs), np.percentile(interp, 2.5, axis=0), np.percentile(interp, 97.5, axis=0), float(auc_obs), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def micro_pr_with_band(y, scores, n_boot: int, seed: int):
    labels = sorted(np.unique(y).tolist())
    y_bin, score = bin_one_vs_rest(y, scores, labels)
    p_obs, r_obs, _ = precision_recall_curve(y_bin.ravel(), score.ravel())
    ap_obs = average_precision_score(y_bin, score, average="micro")
    grid = np.linspace(0, 1, 200)
    rng = np.random.default_rng(seed)
    interp = np.zeros((n_boot, len(grid)))
    aps = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, sc = bin_one_vs_rest(y[idx], scores[idx], labels)
        p, r, _ = precision_recall_curve(yb.ravel(), sc.ravel())
        order = np.argsort(r)
        interp[b] = np.interp(grid, r[order], p[order])
        aps[b] = average_precision_score(yb, sc, average="micro")
    order = np.argsort(r_obs)
    return grid, np.interp(grid, r_obs[order], p_obs[order]), np.percentile(interp, 2.5, axis=0), np.percentile(interp, 97.5, axis=0), float(ap_obs), float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def plot_roc_pr(arrays_by_split, fig_dir: Path, source_dir: Path, n_boot: int, seed: int) -> None:
    for sn, arrays in arrays_by_split.items():
        y = arrays["y_disease"]
        scores = softmax_np(arrays["disease_logits"])
        grid, tpr, lo, hi, auc_obs, auc_lo, auc_hi = micro_roc_with_band(y, scores, n_boot=n_boot, seed=seed)
        fig, ax = plt.subplots(figsize=(3.7, 3.2))
        ax.fill_between(grid, lo, hi, color=PALETTE["mobile"], alpha=0.18, linewidth=0)
        ax.plot(grid, tpr, color=PALETTE["mobile"], linewidth=1.7, label=f"VetDerm-Mobile\nAUC {auc_obs:.3f} ({auc_lo:.3f}-{auc_hi:.3f})")
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate", weight="bold")
        ax.set_ylabel("True positive rate", weight="bold")
        ax.set_title(f"{SPLIT_DISPLAY[sn]} - micro ROC", weight="bold", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, fontsize=8)
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_roc_{sn}")
        pd.DataFrame({"fpr": grid, "tpr": tpr, "tpr_ci_low": lo, "tpr_ci_high": hi}).to_csv(source_dir / f"roc_curve_{sn}.csv", index=False)

        grid, prec, lo, hi, ap, ap_lo, ap_hi = micro_pr_with_band(y, scores, n_boot=n_boot, seed=seed + 500)
        fig, ax = plt.subplots(figsize=(3.7, 3.2))
        ax.fill_between(grid, lo, hi, color=PALETTE["mobile"], alpha=0.18, linewidth=0)
        ax.plot(grid, prec, color=PALETTE["mobile"], linewidth=1.7, label=f"VetDerm-Mobile\nAP {ap:.3f} ({ap_lo:.3f}-{ap_hi:.3f})")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall", weight="bold")
        ax.set_ylabel("Precision", weight="bold")
        ax.set_title(f"{SPLIT_DISPLAY[sn]} - micro PR", weight="bold", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, fontsize=8)
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_pr_{sn}")
        pd.DataFrame({"recall": grid, "precision": prec, "precision_ci_low": lo, "precision_ci_high": hi}).to_csv(source_dir / f"pr_curve_{sn}.csv", index=False)


def reliability_bins(conf, correct, n_bins: int = 15):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(conf, bins[1:-1], right=True)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        center = (bins[b] + bins[b + 1]) / 2
        if not mask.any():
            rows.append({"bin": b, "center": center, "accuracy": np.nan, "confidence": np.nan, "n": 0})
            continue
        acc = float(correct[mask].mean())
        avg = float(conf[mask].mean())
        ece += (int(mask.sum()) / len(conf)) * abs(acc - avg)
        rows.append({"bin": b, "center": center, "accuracy": acc, "confidence": avg, "n": int(mask.sum())})
    return pd.DataFrame(rows), float(ece)


def plot_reliability(predictions: dict[str, pd.DataFrame], fig_dir: Path, source_dir: Path) -> pd.DataFrame:
    rows = []
    for sn, df in predictions.items():
        conf = df["known_confidence"].to_numpy()
        correct = df["correct_known"].astype(float).to_numpy()
        rel, ece = reliability_bins(conf, correct, n_bins=15)
        rel.insert(0, "split", sn)
        rel.to_csv(source_dir / f"reliability_bins_{sn}.csv", index=False)
        rows.append({"split": sn, "split_name": SPLIT_DISPLAY[sn], "ece": ece})

        fig, ax = plt.subplots(figsize=(3.7, 3.2))
        rel_v = rel.dropna()
        ax.plot(rel_v["confidence"], rel_v["accuracy"], marker="o", linewidth=1.7, color=PALETTE["mobile"], markersize=4, label=f"VetDerm-Mobile\nECE {ece:.3f}")
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        percent_axis(ax)
        percent_axis(ax, axis="x")
        ax.set_xlabel("Mean confidence", weight="bold")
        ax.set_ylabel("Empirical accuracy", weight="bold")
        ax.set_title(f"{SPLIT_DISPLAY[sn]} - calibration", weight="bold", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, fontsize=8)
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_reliability_{sn}")

    ece_df = pd.DataFrame(rows)
    ece_df.to_csv(source_dir / "table_ece.csv", index=False)
    return ece_df


def plot_headlines(summary: pd.DataFrame, ece_df: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    summary.to_csv(source_dir / "table_metric_summary.csv", index=False)
    sub = summary[summary["mode"] == "known_species"].copy()
    metrics = [
        ("disease_accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("roc_auc_macro", "Macro ROC-AUC"),
        ("pr_auc_macro", "Macro PR-AUC"),
    ]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        vals = [float(sub[sub["split"] == sn][metric].iloc[0]) for sn in SPLITS]
        err_lo, err_hi = [], []
        for sn in SPLITS:
            row = sub[sub["split"] == sn].iloc[0]
            lo_col = f"{metric}_ci_low"
            hi_col = f"{metric}_ci_high"
            if lo_col in sub.columns and pd.notna(row.get(lo_col, np.nan)):
                err_lo.append(float(row[metric] - row[lo_col]))
                err_hi.append(float(row[hi_col] - row[metric]))
            else:
                err_lo.append(0.0)
                err_hi.append(0.0)
        x = np.arange(len(SPLITS))
        bars = ax.bar(
            x,
            vals,
            width=0.52,
            color=PALETTE["mobile"],
            edgecolor="black",
            linewidth=0.5,
            yerr=np.vstack([err_lo, err_hi]),
            error_kw={"ecolor": "#374151", "lw": 0.7, "capsize": 2.5},
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.022, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_DISPLAY[sn] for sn in SPLITS], weight="bold")
        ax.set_ylim(0, 1.15)
        percent_axis(ax)
        ax.set_ylabel(title, weight="bold")
        ax.set_title(f"{title} - species-conditioned", weight="bold", fontsize=11)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_headline_{safe_name(metric)}")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    vals = [float(ece_df[ece_df["split"] == sn]["ece"].iloc[0]) for sn in SPLITS]
    x = np.arange(len(SPLITS))
    bars = ax.bar(x, vals, width=0.52, color=PALETTE["mobile"], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_DISPLAY[sn] for sn in SPLITS], weight="bold")
    ax.set_ylim(0, max(vals) * 1.55 + 0.005)
    ax.set_ylabel("Expected Calibration Error", weight="bold")
    ax.set_title("Calibration error - species-conditioned", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_ece_comparison")


def plot_top1_top3(summary: pd.DataFrame, fig_dir: Path) -> None:
    sub = summary[summary["mode"] == "known_species"].copy()
    labels = [SPLIT_DISPLAY[sn] for sn in SPLITS]
    top1 = [float(sub[sub["split"] == sn]["disease_accuracy"].iloc[0]) for sn in SPLITS]
    top3 = [float(sub[sub["split"] == sn]["top3_accuracy"].iloc[0]) for sn in SPLITS]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.3, 3.25))
    bars1 = ax.bar(x - width / 2, top1, width=width, color=PALETTE["mobile"], edgecolor="black", linewidth=0.5, label="Top-1")
    bars3 = ax.bar(x + width / 2, top3, width=width, color=PALETTE["mobile_light"], edgecolor="black", linewidth=0.5, hatch="///", label="Top-3")
    for bar, val in list(zip(bars1, top1)) + list(zip(bars3, top3)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.016, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight="bold")
    ax.set_ylim(0, 1.15)
    percent_axis(ax)
    ax.set_ylabel("Disease classification accuracy", weight="bold")
    ax.set_title("Top-1 vs Top-3 accuracy - species-conditioned", weight="bold", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=8)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_top1_vs_top3")


def plot_decoding(summary: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    external = summary[summary["split"] == "development_external"].copy()
    metrics = ["disease_accuracy", "macro_f1", "roc_auc_macro", "pr_auc_macro", "top3_accuracy"]
    metric_labels = ["Accuracy", "Macro-F1", "ROC-AUC", "PR-AUC", "Top-3"]
    rows = []
    for metric, label in zip(metrics, metric_labels):
        raw = float(external[external["mode"] == "raw"][metric].iloc[0])
        pred_species = float(external[external["mode"] == "pred_species"][metric].iloc[0])
        known = float(external[external["mode"] == "known_species"][metric].iloc[0])
        rows.append({"metric": label, "raw": raw, "predicted_species": pred_species, "species_conditioned": known, "gain_known_minus_raw": known - raw})
    table = pd.DataFrame(rows)
    table.to_csv(source_dir / "table_decoding_development_external.csv", index=False)

    x = np.arange(len(metric_labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.5, 3.45))
    for offset, col, label, color in [
        (-width, "raw", "Disease-only", PALETTE["raw"]),
        (0.0, "predicted_species", "Predicted-species", PALETTE["pred_species"]),
        (width, "species_conditioned", "Species-conditioned", PALETTE["known_species"]),
    ]:
        vals = table[col].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="black", linewidth=0.5, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=7.4, weight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, weight="bold")
    ax.set_ylim(0, 1.15)
    percent_axis(ax)
    ax.set_ylabel("Metric value", weight="bold")
    ax.set_title("Development external - decoding mode comparison", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_decoding_modes_development_external")

    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    gains = table["gain_known_minus_raw"].to_numpy(dtype=float) * 100.0
    bars = ax.bar(x, gains, width=0.5, color=PALETTE["mobile"], edgecolor="black", linewidth=0.5, zorder=3)
    for bar, val in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.45, f"+{val:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, weight="bold")
    ax.set_ylim(0, max(gains) * 1.35 + 1.0)
    ax.set_ylabel("Gain (percentage points)", weight="bold")
    ax.set_title("Development external - species-conditioned decoding gain", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5, zorder=0)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_decoding_gain_development_external")


def plot_species_subgroups(predictions: dict[str, pd.DataFrame], fig_dir: Path, source_dir: Path, n_boot: int, seed: int) -> None:
    rows = []
    for sn, df in predictions.items():
        for sp in SPECIES_NAMES:
            sub = df[df["true_species"] == sp]
            if sub.empty:
                continue
            y = sub["true_disease"].to_numpy()
            pred = sub["pred_disease_known"].to_numpy()
            f1m = macro_f1_present(y, pred)
            lo, hi = bootstrap_metric(y, pred, macro_f1_present, n_boot=n_boot, seed=seed + len(rows))
            rows.append(
                {
                    "split": sn,
                    "split_name": SPLIT_DISPLAY[sn],
                    "species": sp,
                    "species_display": disp_species(sp),
                    "n": int(len(sub)),
                    "accuracy": float(accuracy_score(y, pred)),
                    "macro_f1": f1m,
                    "f1_ci_low": lo,
                    "f1_ci_high": hi,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(source_dir / "table_species_subgroup.csv", index=False)
    for sn in SPLITS:
        sub = table[table["split"] == sn]
        species_present = [sp for sp in SPECIES_NAMES if sp in set(sub["species"])]
        x = np.arange(len(species_present))
        vals = np.asarray([float(sub[sub["species"] == sp]["macro_f1"].iloc[0]) for sp in species_present])
        lo = np.asarray([float(sub[sub["species"] == sp]["f1_ci_low"].iloc[0]) for sp in species_present])
        hi = np.asarray([float(sub[sub["species"] == sp]["f1_ci_high"].iloc[0]) for sp in species_present])
        err = np.vstack([vals - lo, hi - vals])
        fig, ax = plt.subplots(figsize=(4.0, 2.9))
        bars = ax.bar(x, vals, width=0.48, color=PALETTE["mobile"], edgecolor="black", linewidth=0.5, yerr=err, error_kw={"ecolor": "#374151", "lw": 0.7, "capsize": 2.5})
        for bar, val, high in zip(bars, vals, hi):
            ax.text(bar.get_x() + bar.get_width() / 2, min(max(val, high) + 0.035, 1.12), f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([disp_species(sp) for sp in species_present], weight="bold")
        ax.set_ylim(0, 1.15)
        percent_axis(ax)
        ax.set_ylabel("Macro-F1 (95% CI)", weight="bold")
        ax.set_title(f"{SPLIT_DISPLAY[sn]} - species subgroup", weight="bold", fontsize=11)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, f"fig_vetderm_mobile_species_subgroup_{sn}")


def plot_selective(predictions: dict[str, pd.DataFrame], fig_dir: Path, source_dir: Path) -> None:
    coverages = [1.0, 0.9, 0.8, 0.7]
    rows = []
    for sn, df in predictions.items():
        ranked = df.sort_values("known_confidence", ascending=False).reset_index(drop=True)
        for coverage in coverages:
            k = max(1, int(round(len(ranked) * coverage)))
            kept = ranked.iloc[:k]
            rows.append(
                {
                    "split": sn,
                    "coverage_retained": coverage,
                    "n_retained": int(k),
                    "accuracy": float(kept["correct_known"].mean()),
                    "min_confidence_retained": float(kept["known_confidence"].min()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(source_dir / "table_selective_classification.csv", index=False)
    x = np.arange(len(coverages))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.1, 3.45))
    for offset, sn, color in [
        (-width, "val", PALETTE["mobile_dark"]),
        (0.0, "internal_test", PALETTE["mobile"]),
        (width, "development_external", PALETTE["external"]),
    ]:
        sub = table[table["split"] == sn].set_index("coverage_retained").reindex(coverages)
        vals = sub["accuracy"].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="black", linewidth=0.5, label=SPLIT_DISPLAY[sn])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.016, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c * 100)}%" for c in coverages], weight="bold")
    ax.set_ylim(0, 1.15)
    percent_axis(ax)
    ax.set_xlabel("Coverage retained", weight="bold")
    ax.set_ylabel("Accuracy", weight="bold")
    ax.set_title("Selective classification at fixed coverage", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_selective_classification")


def plot_domain_gate(safety_dir: Path, fig_dir: Path, source_dir: Path) -> None:
    gate_summary_path = safety_dir / "source_data" / "domain_gate_summary.csv"
    if not gate_summary_path.exists():
        return
    summary = pd.read_csv(gate_summary_path)
    summary.to_csv(source_dir / "table_domain_gate_summary.csv", index=False)
    score_paths = {
        "val": safety_dir / "source_data" / "domain_gate_scores_val.csv",
        "internal_test": safety_dir / "source_data" / "domain_gate_scores_internal_test.csv",
        "development_external": safety_dir / "source_data" / "domain_gate_scores_development_external.csv",
        "ood": safety_dir / "source_data" / "domain_gate_scores_ood.csv",
    }
    frames = []
    for split, path in score_paths.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["split"] = split
            frames.append(frame[["split", "domain_score"]])
    if frames:
        score_df = pd.concat(frames, ignore_index=True)
        score_df.to_csv(source_dir / "domain_gate_scores_all.csv", index=False)
        profile = np.load(safety_dir / "source_data" / "domain_gate_profile_frr05.npz", allow_pickle=True)
        threshold = float(np.asarray(profile["threshold"]))
        fig, ax = plt.subplots(figsize=(4.4, 3.25))
        colors = {
            "val": PALETTE["mobile_dark"],
            "internal_test": PALETTE["mobile"],
            "development_external": PALETTE["external"],
            "ood": PALETTE["ood"],
        }
        labels = {
            "val": "Validation",
            "internal_test": "Internal test",
            "development_external": "Development external",
            "ood": "General-photo OOD",
        }
        bins = np.linspace(0, 1, 41)
        for split in ["val", "internal_test", "development_external", "ood"]:
            data = score_df[score_df["split"] == split]
            if data.empty:
                continue
            ax.hist(data["domain_score"], bins=bins, histtype="step", linewidth=1.8, density=True, color=colors[split], label=labels[split])
        ax.axvline(threshold, color=PALETTE["black"], linewidth=1.4, linestyle="--", label=f"5% validation threshold ({threshold:.3f})")
        ax.set_xlabel("Nearest-neighbour feature similarity", weight="bold")
        ax.set_ylabel("Density", weight="bold")
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8)
        ax.legend(frameon=False, loc="upper left", fontsize=7.6)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, fig_dir, "fig_vetderm_mobile_domain_gate_scores")

    fig, ax = plt.subplots(figsize=(4.4, 3.05))
    split_order = ["val", "internal_test", "development_external", "ood"]
    labels = ["Validation", "Internal test", "Development external", "General-photo OOD"]
    vals = [float(summary[summary["split"] == split]["reject_rate"].iloc[0]) for split in split_order]
    colors = [PALETTE["mobile_dark"], PALETTE["mobile"], PALETTE["external"], PALETTE["ood"]]
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.56, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", weight="bold")
    ax.set_ylim(0, max(vals) * 1.28 + 0.03)
    percent_axis(ax)
    ax.set_ylabel("Rejected by feature gate", weight="bold")
    ax.set_title("Unsupported-image gate rejection rate", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_domain_gate_rejection")


def plot_tflite(results_dir: Path, fig_dir: Path, source_dir: Path) -> None:
    tflite_path = results_dir / "tflite" / "tflite_deployment_comparison.csv"
    if not tflite_path.exists():
        return
    df = pd.read_csv(tflite_path)
    df.to_csv(source_dir / "table_tflite_deployment_comparison.csv", index=False)
    external = df[df["split"] == "development_external"].copy()
    metric_cols = ["disease_accuracy_pct", "macro_f1_pct", "top3_accuracy_pct", "known_species_disease_agreement_pct"]
    metric_labels = ["Accuracy", "Macro-F1", "Top-3", "Agreement"]
    variants = ["fp16", "int8_weightonly"]
    colors = {"fp16": PALETTE["fp16"], "int8_weightonly": PALETTE["int8"]}
    display = {"fp16": "FP16 TFLite", "int8_weightonly": "INT8 weight-only"}
    x = np.arange(len(metric_cols))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    for offset, variant in [(-width / 2, "fp16"), (width / 2, "int8_weightonly")]:
        row = external[external["variant"] == variant].iloc[0]
        vals = [float(row[col]) for col in metric_cols]
        bars = ax.bar(x + offset, vals, width=width, color=colors[variant], edgecolor="black", linewidth=0.5, label=display[variant])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8, weight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Metric (%)", weight="bold")
    ax.set_title("Development external - deployed TFLite fidelity", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_tflite_external_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.0))
    ext = external.set_index("variant").reindex(variants)
    labels = [display[v] for v in variants]
    axes[0].bar(np.arange(2), ext["size_mb"].to_numpy(float), color=[colors[v] for v in variants], edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(np.arange(2))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", weight="bold")
    axes[0].set_ylabel("Model size (MiB)", weight="bold")
    axes[0].set_title("Artifact size", weight="bold", fontsize=10)
    axes[0].grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for i, val in enumerate(ext["size_mb"].to_numpy(float)):
        axes[0].text(i, val + 0.45, f"{val:.2f}", ha="center", va="bottom", fontsize=8, weight="bold")

    axes[1].bar(np.arange(2), ext["latency_ms_median"].to_numpy(float), color=[colors[v] for v in variants], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(labels, rotation=25, ha="right", weight="bold")
    axes[1].set_ylabel("Median latency (ms/image)", weight="bold")
    axes[1].set_title("CPU latency", weight="bold", fontsize=10)
    axes[1].grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for i, val in enumerate(ext["latency_ms_median"].to_numpy(float)):
        axes[1].text(i, val + 2.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8, weight="bold")
    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_vetderm_mobile_tflite_size_latency")


def apply_sgca_gradcam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = 0.05,
    alpha: float = 0.46,
    gamma: float = 0.70,
) -> np.ndarray:
    heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0))
    heatmap_img = heatmap_img.resize((image_rgb.shape[1], image_rgb.shape[0]), resample=Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_img).astype(np.float32) / 255.0
    h_min = float(heatmap_resized.min())
    h_max = float(heatmap_resized.max())
    heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min + 1e-8)
    heatmap_norm = np.power(np.clip(heatmap_norm, 0.0, 1.0), gamma)
    if threshold > 0:
        heatmap_norm = heatmap_norm.copy()
        heatmap_norm[heatmap_norm < threshold] = 0.0
    heatmap_color = (mpl.colormaps["jet"](heatmap_norm)[..., :3] * 255.0).astype(np.uint8)
    visible = heatmap_norm[..., None] > 0
    blended = (1.0 - alpha) * image_rgb + alpha * heatmap_color
    return np.where(visible, np.clip(blended, 0, 255), image_rgb).astype(np.uint8)


def is_problematic_gradcam_path(path: str) -> bool:
    lower = str(path).lower()
    return any(token in lower for token in PROBLEMATIC_GRADCAM_SUBSTRINGS)


def resolve_gradcam_path(path: str) -> Path:
    image_path = Path(str(path))
    if image_path.exists():
        return image_path
    candidate = ROOT / image_path
    if candidate.exists():
        return candidate
    return image_path


def gradcam_path_exists(path: str) -> bool:
    return resolve_gradcam_path(path).exists()


def select_gradcam_rows(pred_df: pd.DataFrame, correct: bool, max_items: int | None) -> pd.DataFrame:
    df = pred_df[pred_df["correct_known"].eq(correct)].copy()
    df = df[df["path"].map(gradcam_path_exists)]
    if df.empty:
        return df
    selected = []
    for disease in order_diseases(df["true_disease"].unique()):
        group = df[df["true_disease"] == disease].sort_values("known_confidence", ascending=False)
        if not group.empty:
            manual_token = MANUAL_GRADCAM_SELECTION_SUBSTRINGS.get(disease)
            manual_group = group[group["path"].astype(str).str.contains(manual_token, regex=False)] if manual_token else pd.DataFrame()
            clean_group = group[~group["path"].map(is_problematic_gradcam_path)]
            if not manual_group.empty:
                chosen = manual_group.iloc[0]
                selection_note = "manual_clean_override"
            elif not clean_group.empty:
                chosen = clean_group.iloc[0]
                selection_note = "clean_path"
            elif not correct:
                continue
            else:
                chosen = group.iloc[0]
                selection_note = "fallback_problematic_path"
            chosen = chosen.copy()
            chosen["selection_note"] = selection_note
            selected.append(chosen)
    out = pd.DataFrame(selected)
    if max_items is not None and len(out) < max_items:
        extra = df.drop(index=out.index, errors="ignore").sort_values("known_confidence", ascending=False).head(max_items - len(out))
        out = pd.concat([out, extra], axis=0)
    out["order"] = out["true_disease"].map({d: i for i, d in enumerate(DISEASE_ORDER)}).fillna(999)
    out = out.sort_values(["order", "true_disease", "known_confidence"], ascending=[True, True, False]).drop(columns=["order"])
    return out if max_items is None else out.head(max_items)


def render_gradcam_panel(
    *,
    selected: pd.DataFrame,
    model,
    transform,
    device: torch.device,
    out_prefix: Path,
    pred_color: str,
    max_items: int | None,
    threshold: float,
    alpha: float,
    gamma: float,
) -> pd.DataFrame:
    selected = selected.copy() if max_items is None else selected.head(max_items).copy()
    cols = 5
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.42 * cols, 2.64 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    records = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        ax = axes.flat[idx]
        image_path = Path(str(row["path"]))
        image_path = resolve_gradcam_path(str(row["path"]))
        if not image_path.exists():
            continue
        true_disease = str(row["true_disease"])
        pred_disease = str(row["pred_disease_known"])
        pred_idx = DISEASE_NAMES.index(pred_disease)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        image_np, cam = gradcam_for_image(model, tensor, pred_idx)
        overlay = apply_sgca_gradcam_overlay(image_np, cam, threshold=threshold, alpha=alpha, gamma=gamma)
        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.7)
            spine.set_linestyle(":")
        ax.text(
            0.5,
            1.105,
            f"True: {short_label(true_disease)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.0,
            color=PALETTE["true_blue"],
            fontweight="bold",
            fontfamily="Arial",
        )
        ax.text(
            0.5,
            1.015,
            f"Pred: {short_label(pred_disease)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.0,
            color=pred_color,
            fontweight="bold",
            fontfamily="Arial",
        )
        records.append(
            {
                "panel_order": idx + 1,
                "path": str(image_path),
                "true_species": row["true_species"],
                "true_disease": true_disease,
                "pred_disease_known": pred_disease,
                "known_confidence": float(row["known_confidence"]),
                "correct_known": bool(row["correct_known"]),
                "selection_note": str(row.get("selection_note", "")),
            }
        )

    for j in range(len(selected), axes.size):
        axes.flat[j].axis("off")
    fig.subplots_adjust(left=0.006, right=0.994, top=0.982, bottom=0.018, wspace=0.0, hspace=0.175)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=900, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=900, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pd.DataFrame(records)


def copy_gradcam_alias(out_prefix: Path, alias_prefix: Path) -> None:
    for suffix in [".png", ".pdf"]:
        src = out_prefix.with_suffix(suffix)
        dst = alias_prefix.with_suffix(suffix)
        if src.exists():
            shutil.copyfile(src, dst)


def load_or_compute_internal_gradcam_predictions(args, model, transform, device: torch.device, source_dir: Path) -> pd.DataFrame:
    out_csv = source_dir / "gradcam_internal_validation_predictions.csv"
    if out_csv.exists() and not args.force_recompute_arrays:
        return pd.read_csv(out_csv)
    paths, arrays = collect_split_arrays(
        model=model,
        split_dir=args.gradcam_internal_val_root,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    df = predictions_from_arrays(paths, arrays)
    df["gradcam_source"] = str(args.gradcam_internal_val_root)
    df.to_csv(out_csv, index=False)
    return df


def plot_gradcam(predictions: dict[str, pd.DataFrame], args, fig_dir: Path, source_dir: Path) -> None:
    max_items = None if args.gradcam_items <= 0 else args.gradcam_items
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    transform = build_eval_transform(args.img_size)
    gradcam_predictions = dict(predictions)
    if not gradcam_predictions["val"]["path"].map(gradcam_path_exists).any():
        gradcam_predictions["val"] = load_or_compute_internal_gradcam_predictions(args, model, transform, device, source_dir)
    summary_rows = []
    for split in ["val", "development_external"]:
        split_df = gradcam_predictions[split]
        for correct_flag, label, color in [
            (True, "correct", PALETTE["correct_green"]),
            (False, "wrong", PALETTE["wrong_red"]),
        ]:
            selected = select_gradcam_rows(split_df, correct=correct_flag, max_items=max_items)
            if selected.empty:
                continue
            out_prefix = fig_dir / f"fig_vetderm_mobile_gradcam_{label}_{split}"
            src = render_gradcam_panel(
                selected=selected,
                model=model,
                transform=transform,
                device=device,
                out_prefix=out_prefix,
                pred_color=color,
                max_items=max_items,
                threshold=args.gradcam_threshold,
                alpha=args.gradcam_alpha,
                gamma=args.gradcam_gamma,
            )
            src_path = source_dir / f"gradcam_{label}_selected_images_{split}.csv"
            src.to_csv(src_path, index=False)
            if split == "development_external":
                alias = fig_dir / f"fig_vetderm_mobile_gradcam_{label}"
                copy_gradcam_alias(out_prefix, alias)
                src.to_csv(source_dir / f"gradcam_{label}_selected_images.csv", index=False)
            missing_classes = sorted(set(split_df["true_disease"].unique()) - set(src["true_disease"].unique()))
            summary_rows.append(
                {
                    "split": split,
                    "panel": label,
                    "available_classes": int(split_df[split_df["correct_known"].eq(correct_flag)]["true_disease"].nunique()),
                    "shown_classes": int(src["true_disease"].nunique()),
                    "shown_images": int(len(src)),
                    "missing_classes": "; ".join(missing_classes),
                    "gradcam_threshold": args.gradcam_threshold,
                    "gradcam_alpha": args.gradcam_alpha,
                    "gradcam_gamma": args.gradcam_gamma,
                    "source_note": str(split_df.get("gradcam_source", pd.Series(["saved_predictions"])).iloc[0]),
                }
            )
    pd.DataFrame(summary_rows).to_csv(source_dir / "gradcam_panel_summary.csv", index=False)


def plot_sgca_comparison(summary: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    sgca_path = ROOT / "publication_figures" / "top2_model_comparison_900dpi" / "updated_fig" / "source_data" / "table_metric_summary.csv"
    if not sgca_path.exists():
        return
    sgca = pd.read_csv(sgca_path)
    sgca_ext = sgca[
        (sgca["model_name"] == "Unified SGCA")
        & (sgca["split"] == "development_external")
        & (sgca["mode"] == "known_species")
    ]
    mobile_ext = summary[(summary["split"] == "development_external") & (summary["mode"] == "known_species")]
    if sgca_ext.empty or mobile_ext.empty:
        return
    sgca_row = sgca_ext.iloc[0]
    mobile_row = mobile_ext.iloc[0]
    rows = []
    for metric, label in [
        ("disease_accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("roc_auc_macro", "ROC-AUC"),
        ("pr_auc_macro", "PR-AUC"),
        ("top3_accuracy", "Top-3"),
    ]:
        rows.append({"model": "Unified SGCA", "metric": label, "value": float(sgca_row[metric])})
        rows.append({"model": "VetDerm-Mobile", "metric": label, "value": float(mobile_row[metric])})
    table = pd.DataFrame(rows)
    table.to_csv(source_dir / "table_sgca_vs_vetderm_external.csv", index=False)
    metrics = ["Accuracy", "Macro-F1", "ROC-AUC", "PR-AUC", "Top-3"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    for offset, model, color in [(-width / 2, "Unified SGCA", "#0072B2"), (width / 2, "VetDerm-Mobile", PALETTE["mobile"])]:
        vals = table[table["model"] == model].set_index("metric").reindex(metrics)["value"].to_numpy(float)
        bars = ax.bar(x + offset, vals, width=width, color=color, edgecolor="black", linewidth=0.5, label=model)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val * 100:.1f}", ha="center", va="bottom", fontsize=8, weight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, weight="bold")
    ax.set_ylim(0, 1.15)
    percent_axis(ax)
    ax.set_ylabel("Metric value", weight="bold")
    ax.set_title("Development external - SGCA vs VetDerm-Mobile", weight="bold", fontsize=11)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_sgca_vs_vetderm_mobile_development_external")


def write_report(out_dir: Path, fig_dir: Path, source_dir: Path, summary: pd.DataFrame, ece_df: pd.DataFrame) -> None:
    pngs = sorted(fig_dir.glob("*.png"))
    pdfs = sorted(fig_dir.glob("*.pdf"))
    csvs = sorted(source_dir.glob("*.csv"))
    external = summary[(summary["split"] == "development_external") & (summary["mode"] == "known_species")].iloc[0]
    report = {
        "output_dir": str(out_dir),
        "figures_png": len(pngs),
        "figures_pdf": len(pdfs),
        "source_csv": len(csvs),
        "development_external": {
            "accuracy": float(external["disease_accuracy"]),
            "macro_f1": float(external["macro_f1"]),
            "roc_auc_macro": float(external["roc_auc_macro"]),
            "pr_auc_macro": float(external["pr_auc_macro"]),
            "top3_accuracy": float(external["top3_accuracy"]),
            "ece": float(ece_df[ece_df["split"] == "development_external"]["ece"].iloc[0]),
        },
        "figures": [p.name for p in pngs],
    }
    (out_dir / "VETDERM_MOBILE_PUBLICATION_FIGURES_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# VetDerm-Mobile publication figures",
        "",
        f"Output directory: `{out_dir}`",
        f"PNG figures: {len(pngs)}",
        f"PDF figures: {len(pdfs)}",
        f"Source CSV files: {len(csvs)}",
        "",
        "## Development external headline",
        f"- Accuracy: {external['disease_accuracy'] * 100:.1f}%",
        f"- Macro-F1: {external['macro_f1'] * 100:.1f}%",
        f"- Macro ROC-AUC: {external['roc_auc_macro']:.3f}",
        f"- Macro PR-AUC: {external['pr_auc_macro']:.3f}",
        f"- Top-3 accuracy: {external['top3_accuracy'] * 100:.1f}%",
        f"- ECE: {float(ece_df[ece_df['split'] == 'development_external']['ece'].iloc[0]):.3f}",
        "",
        "## Figure files",
    ]
    lines.extend([f"- `{p.name}`" for p in pngs])
    (out_dir / "VETDERM_MOBILE_PUBLICATION_FIGURES_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "vetderm_mobile_results" / "vetderm_mobile_final_single_384_v7" / "vetderm_mobile_final_single.pt")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "vetderm_mobile_results" / "vetderm_mobile_final_single_384_v7")
    parser.add_argument("--split-root", type=Path, default=Path("/mnt/e/Ringworm detection project/Extra_all/dog_normal_finetune_datasets/prepared/combined_split_dirs_v7_no_external_overlap_trainonly"))
    parser.add_argument("--external-root", type=Path, default=ROOT / "evaluation_data" / "development_external_2026_05_02")
    parser.add_argument("--gradcam-internal-val-root", type=Path, default=ROOT / "training_data_deduped_splits" / "seed42" / "val")
    parser.add_argument("--safety-dir", type=Path, default=ROOT / "vetderm_mobile_results" / "vetderm_mobile_final_single_384_v7" / "safety_evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "publication_figures" / "vetderm_mobile_900dpi")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--curve-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradcam-items", type=int, default=0, help="Maximum Grad-CAM examples per panel; 0 means one per available class.")
    parser.add_argument("--gradcam-threshold", type=float, default=0.05)
    parser.add_argument("--gradcam-alpha", type=float, default=0.46)
    parser.add_argument("--gradcam-gamma", type=float, default=0.70)
    parser.add_argument("--force-recompute-arrays", action="store_true")
    args = parser.parse_args()

    configure_style()
    out_dir = args.output_dir
    fig_dir = out_dir / "figs"
    source_dir = out_dir / "source_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths_by_split, arrays_by_split = collect_or_load_all(args, source_dir)
    predictions = {}
    for sn in SPLITS:
        df = predictions_from_arrays(paths_by_split[sn], arrays_by_split[sn])
        predictions[sn] = df
        df.to_csv(source_dir / f"predictions_vetderm_mobile_{sn}.csv", index=False)

    summary = compute_summary(arrays_by_split, n_boot=args.bootstrap, seed=args.seed)
    ece_df = plot_reliability(predictions, fig_dir, source_dir)
    plot_headlines(summary, ece_df, fig_dir, source_dir)
    plot_top1_top3(summary, fig_dir)
    plot_decoding(summary, fig_dir, source_dir)
    plot_confusion(predictions, fig_dir)
    plot_per_class_figures(arrays_by_split, fig_dir, source_dir, n_boot=args.bootstrap, seed=args.seed)
    plot_roc_pr(arrays_by_split, fig_dir, source_dir, n_boot=args.curve_bootstrap, seed=args.seed)
    plot_species_subgroups(predictions, fig_dir, source_dir, n_boot=args.bootstrap, seed=args.seed)
    plot_selective(predictions, fig_dir, source_dir)
    plot_domain_gate(args.safety_dir, fig_dir, source_dir)
    plot_tflite(args.results_dir, fig_dir, source_dir)
    plot_gradcam(predictions, args, fig_dir, source_dir)
    plot_sgca_comparison(summary, fig_dir, source_dir)
    write_report(out_dir, fig_dir, source_dir, summary, ece_df)
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
