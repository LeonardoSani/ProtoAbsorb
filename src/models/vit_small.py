from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTSmall(nn.Module):
    """ViT-S/16 (timm) with return_features=True interface. Expects 224×224 input."""

    feat_dim = 384

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0
        )
        self.head = nn.Linear(384, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits


def build_vit_small(num_classes: int = 10) -> ViTSmall:
    return ViTSmall(num_classes=num_classes)
