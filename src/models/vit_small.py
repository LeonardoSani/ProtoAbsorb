from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.base import BackboneClassifier


class ViTSmall(BackboneClassifier):
    """ViT-S/16 (timm) conforming to the BackboneClassifier ABC.

    Expects 224x224 input.
    """

    feat_dim: int = 384

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0
        )
        self.head = nn.Linear(384, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


def build_vit_small(num_classes: int = 10) -> ViTSmall:
    return ViTSmall(num_classes=num_classes)
