#!/usr/bin/env python3
"""Export VetDerm-Mobile checkpoints to LiteRT/TFLite variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import litert_torch
import torch
from litert_torch.generative.quantize import quant_recipes

from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES


def load_model(checkpoint_path: Path) -> VetDermMobileNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=False)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    return model


def convert_variant(model: torch.nn.Module, sample_args: tuple[torch.Tensor], variant: str):
    quant_config = None
    if variant == "fp16":
        quant_config = quant_recipes.full_fp16_recipe()
    elif variant == "int8_weightonly":
        quant_config = quant_recipes.full_weight_only_recipe()
    elif variant != "fp32":
        raise ValueError(f"Unsupported variant: {variant}")

    return litert_torch.convert(
        model,
        sample_args,
        quant_config=quant_config,
        strict_export=False,
        lightweight_conversion=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--variants", default="fp32,fp16,int8_weightonly")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint)
    sample_args = (torch.randn(1, 3, args.img_size, args.img_size),)

    rows = []
    for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
        out_path = args.output_dir / f"vetderm_mobile_final_single_384_{variant}.tflite"
        edge_model = convert_variant(model, sample_args, variant)
        edge_model.export(str(out_path))
        rows.append(
            {
                "variant": variant,
                "path": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "size_mb": out_path.stat().st_size / (1024 * 1024),
            }
        )
        print(f"{variant}: {out_path} ({rows[-1]['size_mb']:.2f} MiB)", flush=True)

    (args.output_dir / "tflite_export_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
