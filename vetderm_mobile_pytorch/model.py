"""PyTorch VetDerm-Mobile model.

This is intentionally different from SGCA. It is a compact dual-attention multi-task CNN:
EfficientNetV2-B0 -> channel/spatial attention -> global/local fusion -> species head ->
species-assisted disease head.
"""

from __future__ import annotations

import torch
from torch import nn
import timm


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.channel(x).unsqueeze(-1).unsqueeze(-1)
        x = x * gate
        avg = x.mean(dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True).values
        spatial_gate = self.spatial(torch.cat([avg, maxv], dim=1))
        return x * spatial_gate


class VetDermMobileNet(nn.Module):
    def __init__(
        self,
        n_species: int = 3,
        n_diseases: int = 21,
        backbone_name: str = "tf_efficientnetv2_b0.in1k",
        pretrained: bool = True,
        local_filters: int = 64,
        global_units: int = 128,
        species_units: int = 64,
        disease_units: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
        channels = int(self.backbone.num_features)
        self.feature_channels = channels
        self.attention = ChannelSpatialAttention(channels)
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, local_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(local_filters),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_filters, local_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(local_filters),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, global_units),
            nn.ReLU(inplace=True),
        )
        fused_units = global_units + local_filters
        self.fused_norm = nn.BatchNorm1d(fused_units)
        self.fused_dropout = nn.Dropout(dropout)

        self.species_hidden = nn.Sequential(
            nn.Linear(fused_units, species_units),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(species_units),
            nn.Dropout(dropout),
        )
        self.species_head = nn.Linear(species_units, n_species)
        self.disease_head = nn.Sequential(
            nn.Linear(fused_units + species_units, disease_units),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(disease_units),
            nn.Dropout(dropout),
            nn.Linear(disease_units, n_diseases),
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x)
        feat = self.attention(feat)
        local = self.local_branch(feat)
        global_feat = self.global_branch(feat)
        fused = torch.cat([global_feat, local], dim=1)
        fused = self.fused_dropout(self.fused_norm(fused))
        return feat, fused

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, fused = self.forward_features(x)
        species_feat = self.species_hidden(fused)
        species_logits = self.species_head(species_feat)
        disease_logits = self.disease_head(torch.cat([fused, species_feat], dim=1))
        return species_logits, disease_logits
