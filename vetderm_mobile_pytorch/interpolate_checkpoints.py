#!/usr/bin/env python3
"""Evaluate single-model weight interpolation between two VetDerm-Mobile checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from vetderm_mobile_pytorch.data import build_eval_transform, index_split_dir
from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES
from vetderm_mobile_pytorch.train_vetderm_mobile import build_loader, evaluate_split, seed_everything


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("model_state", checkpoint)


def interpolate_states(
    student_state: dict[str, torch.Tensor],
    teacher_state: dict[str, torch.Tensor],
    teacher_fraction: float,
) -> dict[str, torch.Tensor]:
    if student_state.keys() != teacher_state.keys():
        raise ValueError("Checkpoint state dictionaries do not have identical parameter keys")

    result: dict[str, torch.Tensor] = {}
    for name, student_value in student_state.items():
        teacher_value = teacher_state[name]
        if student_value.shape != teacher_value.shape:
            raise ValueError(f"Checkpoint tensor shape mismatch for {name}: {student_value.shape} vs {teacher_value.shape}")
        if torch.is_floating_point(student_value):
            result[name] = student_value.lerp(teacher_value, teacher_fraction)
        else:
            result[name] = teacher_value.clone() if teacher_fraction >= 0.5 else student_value.clone()
    return result


def save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    teacher_fraction: float,
    args: argparse.Namespace,
    selection: str,
) -> None:
    torch.save(
        {
            "model_state": state,
            "species_names": SPECIES_NAMES,
            "disease_names": DISEASE_NAMES,
            "img_size": args.img_size,
            "selection": selection,
            "teacher_fraction": teacher_fraction,
            "student_checkpoint": str(args.student_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-fractions", default="0,0.25,0.5,0.75,1")
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
    fractions = [float(value.strip()) for value in args.teacher_fractions.split(",")]
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("Teacher fractions must be between 0 and 1")

    val_frame = index_split_dir(args.split_root / "val", SPECIES_NAMES, DISEASE_NAMES)
    external_frame = index_split_dir(args.external_root, SPECIES_NAMES, DISEASE_NAMES)
    transform = build_eval_transform(args.img_size)
    val_loader = build_loader(val_frame, transform, args.batch_size, args.num_workers, False)
    external_loader = build_loader(external_frame, transform, args.batch_size, args.num_workers, False)

    student_state = load_state(args.student_checkpoint)
    teacher_state = load_state(args.teacher_checkpoint)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=False).to(device)

    rows: list[dict[str, float | int | str]] = []
    states: dict[float, dict[str, torch.Tensor]] = {}
    for fraction in fractions:
        state = interpolate_states(student_state, teacher_state, fraction)
        states[fraction] = state
        model.load_state_dict(state)
        for split_name, loader in (("val", val_loader), ("development_external", external_loader)):
            metrics = next(row for row in evaluate_split(model, loader, device, split_name) if row["mode"] == "known_species")
            rows.append({"teacher_fraction": fraction, **metrics})
            print(
                f"teacher_fraction={fraction:.2f} split={split_name} "
                f"accuracy={metrics['disease_accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "interpolation_metrics.csv", index=False)
    val_rows = results[results["split"] == "val"]
    external_rows = results[results["split"] == "development_external"]
    val_fraction = float(val_rows.loc[val_rows["macro_f1"].idxmax(), "teacher_fraction"])
    external_fraction = float(external_rows.loc[external_rows["macro_f1"].idxmax(), "teacher_fraction"])

    save_checkpoint(
        args.output_dir / "vetderm_mobile_validation_selected.pt",
        states[val_fraction],
        val_fraction,
        args,
        "validation_macro_f1",
    )
    save_checkpoint(
        args.output_dir / "vetderm_mobile_development_selected.pt",
        states[external_fraction],
        external_fraction,
        args,
        "development_external_macro_f1",
    )
    summary = {
        "validation_selected_teacher_fraction": val_fraction,
        "development_selected_teacher_fraction": external_fraction,
        "student_checkpoint": str(args.student_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
    }
    (args.output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
