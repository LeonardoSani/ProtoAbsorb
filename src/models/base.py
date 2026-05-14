"""Abstract base class for image classifier backbones.

Subclasses must implement `features`, `classify`, and provide a `feat_dim`
(usually a class attribute). The concrete `forward` is inherited and keeps the
existing `(x, return_features=False)` contract used by the rest of the codebase.
"""

from __future__ import annotations

import abc
from typing import Tuple, Union

import torch
import torch.nn as nn


class BackboneClassifier(nn.Module, abc.ABC):
    """Common contract for ResNet-18 / ResNet-50 / ViT-Small backbones."""

    @property
    @abc.abstractmethod
    def feat_dim(self) -> int:
        """Penultimate-layer embedding dimension."""

    @abc.abstractmethod
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding of shape `(B, feat_dim)`."""

    @abc.abstractmethod
    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        """Project an embedding to class logits, shape `(B, num_classes)`."""

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feats = self.features(x)
        logits = self.classify(feats)
        if return_features:
            return logits, feats
        return logits
