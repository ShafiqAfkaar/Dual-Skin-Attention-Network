#!/usr/bin/env python3
"""Evaluate a VetDerm-Mobile checkpoint on validation, internal, and development-external splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from vetderm_mobile_pytorch.data import build_eval_transform, index_split_dir
from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES
from vetderm_mobile_pytorch.train_vetderm_mobile import build_loader, evaluate_split, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=False).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))

    transform = build_eval_transform(args.img_size)
    split_frames = {
        "val": index_split_dir(args.split_root / "val", SPECIES_NAMES, DISEASE_NAMES),
        "internal_test": index_split_dir(args.split_root / "test", SPECIES_NAMES, DISEASE_NAMES),
        "development_external": index_split_dir(args.external_root, SPECIES_NAMES, DISEASE_NAMES),
    }

    rows: list[dict[str, float | int | str]] = []
    for split_name, frame in split_frames.items():
        loader = build_loader(frame, transform, args.batch_size, args.num_workers, False)
        rows.extend(evaluate_split(model, loader, device, split_name, args.output_dir))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
