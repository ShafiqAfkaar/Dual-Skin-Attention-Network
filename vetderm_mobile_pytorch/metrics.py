"""Metrics and species-conditioned decoding for VetDerm-Mobile."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, roc_auc_score

from .taxonomy import DISEASE_NAMES, SPECIES_NAMES, SPECIES_TO_DISEASE_INDICES


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def constrained_predictions(disease_logits: np.ndarray, species_indices: np.ndarray) -> np.ndarray:
    pred = np.empty(len(disease_logits), dtype=np.int64)
    for i, sp_idx in enumerate(species_indices.astype(int)):
        valid = np.asarray(SPECIES_TO_DISEASE_INDICES[SPECIES_NAMES[sp_idx]], dtype=np.int64)
        pred[i] = valid[np.argmax(disease_logits[i, valid])]
    return pred


def constrained_topk(disease_logits: np.ndarray, species_indices: np.ndarray, k: int = 3) -> list[list[int]]:
    rows = []
    for i, sp_idx in enumerate(species_indices.astype(int)):
        valid = np.asarray(SPECIES_TO_DISEASE_INDICES[SPECIES_NAMES[sp_idx]], dtype=np.int64)
        order = valid[np.argsort(disease_logits[i, valid])[::-1][:k]]
        rows.append(order.tolist())
    return rows


def summarize_predictions(arrays: dict[str, np.ndarray], mode: str = "known_species") -> dict[str, float]:
    y_sp = arrays["y_species"]
    y_dis = arrays["y_disease"]
    sp_logits = arrays["species_logits"]
    dis_logits = arrays["disease_logits"]
    sp_pred = sp_logits.argmax(axis=1)
    if mode == "raw":
        dis_pred = dis_logits.argmax(axis=1)
        top3 = np.argsort(dis_logits, axis=1)[:, ::-1][:, :3]
    elif mode == "pred_species":
        dis_pred = constrained_predictions(dis_logits, sp_pred)
        top3 = constrained_topk(dis_logits, sp_pred, 3)
    elif mode == "known_species":
        dis_pred = constrained_predictions(dis_logits, y_sp)
        top3 = constrained_topk(dis_logits, y_sp, 3)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    labels = sorted(np.unique(y_dis).tolist())
    top3_acc = float(np.mean([int(y in row) for y, row in zip(y_dis, top3)]))
    row = {
        "mode": mode,
        "n": int(len(y_dis)),
        "species_accuracy": float(accuracy_score(y_sp, sp_pred)),
        "disease_accuracy": float(accuracy_score(y_dis, dis_pred)),
        "macro_f1": float(f1_score(y_dis, dis_pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": top3_acc,
    }
    probs = softmax_np(dis_logits)
    try:
        y_onehot = np.zeros((len(y_dis), probs.shape[1]), dtype=np.float32)
        y_onehot[np.arange(len(y_dis)), y_dis] = 1.0
        row["roc_auc_macro"] = float(roc_auc_score(y_onehot[:, labels], probs[:, labels], average="macro", multi_class="ovr"))
        row["pr_auc_macro"] = float(average_precision_score(y_onehot[:, labels], probs[:, labels], average="macro"))
    except Exception:
        row["roc_auc_macro"] = float("nan")
        row["pr_auc_macro"] = float("nan")
    return row


def prediction_frame(arrays: dict[str, np.ndarray], paths: list[str]) -> pd.DataFrame:
    y_sp = arrays["y_species"]
    y_dis = arrays["y_disease"]
    sp_pred = arrays["species_logits"].argmax(axis=1)
    raw_pred = arrays["disease_logits"].argmax(axis=1)
    known_pred = constrained_predictions(arrays["disease_logits"], y_sp)
    return pd.DataFrame(
        {
            "path": paths,
            "true_species": [SPECIES_NAMES[i] for i in y_sp],
            "pred_species": [SPECIES_NAMES[i] for i in sp_pred],
            "true_disease": [DISEASE_NAMES[i] for i in y_dis],
            "pred_disease_raw": [DISEASE_NAMES[i] for i in raw_pred],
            "pred_disease_known_species": [DISEASE_NAMES[i] for i in known_pred],
            "correct_known_species": known_pred == y_dis,
        }
    )


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, out_path, labels: list[int] | None = None) -> None:
    labels = labels or sorted(np.unique(y_true).tolist())
    names = [DISEASE_NAMES[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0)
    out_path.write_text(report, encoding="utf-8")
