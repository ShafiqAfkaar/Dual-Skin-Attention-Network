"""Dataset and transform helpers for the PyTorch VetDerm-Mobile workflow."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .taxonomy import DISEASE_NAMES, SPECIES_NAMES

ImageFile.LOAD_TRUNCATED_IMAGES = True
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _random_jpeg_compress(image: Image.Image, quality_range: tuple[int, int] = (35, 92)) -> Image.Image:
    buffer = io.BytesIO()
    quality = random.randint(*quality_range)
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def _random_downsample_upsample(image: Image.Image, scale_range: tuple[float, float] = (0.55, 0.90)) -> Image.Image:
    width, height = image.size
    scale = random.uniform(*scale_range)
    low_width = max(16, int(width * scale))
    low_height = max(16, int(height * scale))
    down = image.resize((low_width, low_height), resample=Image.BICUBIC)
    return down.resize((width, height), resample=Image.BICUBIC).convert("RGB")


def _random_jpeg_compress_heavy(image: Image.Image) -> Image.Image:
    return _random_jpeg_compress(image, quality_range=(28, 92))


def _random_downsample_upsample_heavy(image: Image.Image) -> Image.Image:
    return _random_downsample_upsample(image, scale_range=(0.42, 0.90))


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def index_split_dir(split_dir: Path, species_names: list[str] | None = None, disease_names: list[str] | None = None) -> pd.DataFrame:
    """Index a split with layout split/species/disease/image.*.

    Path.is_file() follows symlinks, which is required for the dog-normal combined split dirs.
    """
    species_names = species_names or SPECIES_NAMES
    disease_names = disease_names or DISEASE_NAMES
    sp2i = {name: idx for idx, name in enumerate(species_names)}
    dis2i = {name: idx for idx, name in enumerate(disease_names)}
    rows = []
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    for species_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if species_dir.name not in sp2i:
            continue
        for disease_dir in sorted([p for p in species_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            if disease_dir.name not in dis2i:
                continue
            for image_path in iter_image_files(disease_dir):
                rows.append(
                    {
                        "path": str(image_path),
                        "species_name": species_dir.name,
                        "disease_name": disease_dir.name,
                        "species": sp2i[species_dir.name],
                        "disease": dis2i[disease_dir.name],
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No indexed images found under {split_dir}")
    return frame


class VetDermFolderDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = Image.open(row.path).convert("RGB")
        image = self.transform(image)
        return image, int(row.species), int(row.disease), row.path


def build_train_transform(img_size: int = 224, policy: str = "internet_robust"):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    policy = policy.lower()
    if policy == "standard":
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                T.ToTensor(),
                normalize,
            ]
        )
    if policy == "internet_robust":
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.78, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(18),
                T.RandomPerspective(distortion_scale=0.08, p=0.25),
                T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05)], p=0.85),
                T.RandomAutocontrast(p=0.25),
                T.RandomAdjustSharpness(sharpness_factor=1.7, p=0.25),
                T.RandomApply([T.Lambda(_random_jpeg_compress)], p=0.55),
                T.RandomApply([T.Lambda(_random_downsample_upsample)], p=0.35),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.4))], p=0.22),
                T.ToTensor(),
                normalize,
            ]
        )
    if policy == "internet_robust_heavy":
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.62, 1.0), ratio=(0.78, 1.28)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(22),
                T.RandomPerspective(distortion_scale=0.12, p=0.35),
                T.RandomApply([T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.35, hue=0.06)], p=0.90),
                T.RandomAutocontrast(p=0.35),
                T.RandomAdjustSharpness(sharpness_factor=1.9, p=0.30),
                T.RandomApply([T.Lambda(_random_jpeg_compress_heavy)], p=0.65),
                T.RandomApply([T.Lambda(_random_downsample_upsample_heavy)], p=0.45),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.8))], p=0.30),
                T.ToTensor(),
                T.RandomErasing(p=0.12, scale=(0.015, 0.065), ratio=(0.4, 2.2), value="random"),
                normalize,
            ]
        )
    raise ValueError(f"Unknown transform policy: {policy}")


def build_eval_transform(img_size: int = 224):
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def collate(batch):
    images, species, diseases, paths = zip(*batch)
    return torch.stack(images), torch.tensor(species, dtype=torch.long), torch.tensor(diseases, dtype=torch.long), list(paths)
