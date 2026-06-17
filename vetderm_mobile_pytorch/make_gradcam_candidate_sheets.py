#!/usr/bin/env python3
"""Create contact sheets for manual VetDerm-Mobile Grad-CAM image curation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def make_sheet(predictions: pd.DataFrame, disease: str, out_path: Path, top_n: int) -> None:
    sub = (
        predictions[(predictions["true_disease"] == disease) & (predictions["correct_known"] == True)]
        .sort_values("known_confidence", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    thumbs = []
    for idx, row in sub.iterrows():
        path = Path(str(row["path"]))
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((220, 180))
        tile = Image.new("RGB", (240, 230), "white")
        tile.paste(image, ((240 - image.width) // 2, 5))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 188), f"{idx}: {float(row['known_confidence']):.3f}", fill="black")
        draw.text((6, 205), path.name[:31], fill="black")
        thumbs.append(tile)

    cols = 4
    rows = max(1, (len(thumbs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * 240, rows * 230), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 240, (idx // cols) * 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("diseases", nargs="+")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    for disease in args.diseases:
        out_path = args.output_dir / f"candidates_{disease.replace(' ', '_')}.jpg"
        make_sheet(predictions, disease, out_path, args.top_n)
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
