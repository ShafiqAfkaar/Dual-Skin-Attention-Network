#!/usr/bin/env python3
"""Evaluate and benchmark LiteRT/TFLite VetDerm-Mobile artifacts."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import litert_torch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vetderm_mobile_pytorch.data import VetDermFolderDataset, build_eval_transform, index_split_dir
from vetderm_mobile_pytorch.metrics import constrained_predictions, prediction_frame, save_classification_report, summarize_predictions
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES


def run_model(edge_model, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    outputs = edge_model(images.numpy().astype(np.float32))
    if isinstance(outputs, tuple):
        species_logits, disease_logits = outputs
    else:
        species_logits = outputs["output_0"]
        disease_logits = outputs["output_1"]
    return np.asarray(species_logits), np.asarray(disease_logits)


def evaluate_split(
    edge_model,
    frame: pd.DataFrame,
    img_size: int,
    split_name: str,
    out_dir: Path,
    reference_dir: Path | None,
    show_progress: bool,
):
    dataset = VetDermFolderDataset(frame, build_eval_transform(img_size))
    y_species, y_disease, species_logits, disease_logits, paths = [], [], [], [], []
    latencies_ms = []

    for idx in tqdm(range(len(dataset)), desc=f"tflite {split_name}", leave=False, disable=not show_progress):
        image, sp, dis, path = dataset[idx]
        batch = image.unsqueeze(0).contiguous()
        start = time.perf_counter()
        sp_logits, dis_logits = run_model(edge_model, batch)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        y_species.append(sp)
        y_disease.append(dis)
        species_logits.append(sp_logits[0])
        disease_logits.append(dis_logits[0])
        paths.append(path)

    arrays = {
        "y_species": np.asarray(y_species, dtype=np.int64),
        "y_disease": np.asarray(y_disease, dtype=np.int64),
        "species_logits": np.asarray(species_logits, dtype=np.float32),
        "disease_logits": np.asarray(disease_logits, dtype=np.float32),
    }
    pred_df = prediction_frame(arrays, paths)
    pred_path = out_dir / f"predictions_{split_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    known_pred = constrained_predictions(arrays["disease_logits"], arrays["y_species"])
    save_classification_report(arrays["y_disease"], known_pred, out_dir / f"disease_report_{split_name}_known_species.txt")

    rows = []
    for mode in ["raw", "pred_species", "known_species"]:
        row = summarize_predictions(arrays, mode=mode)
        row["split"] = split_name
        row["latency_ms_mean"] = float(np.mean(latencies_ms))
        row["latency_ms_median"] = float(np.median(latencies_ms))
        row["latency_ms_p95"] = float(np.percentile(latencies_ms, 95))
        row["throughput_img_s"] = float(1000.0 / np.mean(latencies_ms))
        rows.append(row)

    agreement = {}
    if reference_dir is not None:
        ref_path = reference_dir / f"predictions_{split_name}.csv"
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path)
            pred_norm = pred_df.assign(path_norm=pred_df["path"].str.lower())
            ref_norm = ref_df.assign(path_norm=ref_df["path"].str.lower())
            merged = pred_norm.merge(
                ref_norm[
                    [
                        "path_norm",
                        "pred_species",
                        "pred_disease_raw",
                        "pred_disease_known_species",
                    ]
                ],
                on="path_norm",
                suffixes=("_tflite", "_torch"),
                how="inner",
            )
            agreement = {
                "split": split_name,
                "n_matched": int(len(merged)),
                "species_agreement": float((merged["pred_species_tflite"] == merged["pred_species_torch"]).mean()),
                "raw_disease_agreement": float(
                    (merged["pred_disease_raw_tflite"] == merged["pred_disease_raw_torch"]).mean()
                ),
                "known_species_disease_agreement": float(
                    (merged["pred_disease_known_species_tflite"] == merged["pred_disease_known_species_torch"]).mean()
                ),
            }

    return rows, agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite-model", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_model = litert_torch.Model.load(str(args.tflite_model))
    split_frames = {
        "val": index_split_dir(args.split_root / "val", SPECIES_NAMES, DISEASE_NAMES),
        "internal_test": index_split_dir(args.split_root / "test", SPECIES_NAMES, DISEASE_NAMES),
        "development_external": index_split_dir(args.external_root, SPECIES_NAMES, DISEASE_NAMES),
    }

    metrics_rows = []
    agreements = []
    for split_name, frame in split_frames.items():
        rows, agreement = evaluate_split(
            edge_model,
            frame,
            args.img_size,
            split_name,
            args.output_dir,
            args.reference_dir,
            show_progress=not args.no_progress,
        )
        metrics_rows.extend(rows)
        if agreement:
            agreements.append(agreement)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    if agreements:
        pd.DataFrame(agreements).to_csv(args.output_dir / "pytorch_agreement.csv", index=False)

    summary = {
        "model": str(args.tflite_model),
        "size_bytes": args.tflite_model.stat().st_size,
        "size_mb": args.tflite_model.stat().st_size / (1024 * 1024),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "metrics_csv": str(args.output_dir / "metrics_summary.csv"),
        "agreement_csv": str(args.output_dir / "pytorch_agreement.csv") if agreements else None,
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False), flush=True)
    if agreements:
        print(pd.DataFrame(agreements).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
