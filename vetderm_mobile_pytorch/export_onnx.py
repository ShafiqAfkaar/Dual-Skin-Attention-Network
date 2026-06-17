#!/usr/bin/env python3
"""Export a VetDerm-Mobile PyTorch checkpoint to ONNX for downstream TFLite conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vetderm_mobile_pytorch.model import VetDermMobileNet
from vetderm_mobile_pytorch.taxonomy import DISEASE_NAMES, SPECIES_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = VetDermMobileNet(n_species=len(SPECIES_NAMES), n_diseases=len(DISEASE_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["image"],
        output_names=["species_logits", "disease_logits"],
        dynamic_axes={"image": {0: "batch"}, "species_logits": {0: "batch"}, "disease_logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"saved ONNX model to {output}")


if __name__ == "__main__":
    main()
