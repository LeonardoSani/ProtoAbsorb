"""Build a pycls R-26 (ResNet, depth=26) and load a local checkpoint.

The TENT paper's CIFAR-10/100 source models are R-26 trained with pycls's
ResNet (`basic_transform` block). RobustBench does NOT host these
checkpoints; we train them with `train_R26.py` and load them here.

Arch keys understood:
    Local_R26_cifar10   → ResNet(num_classes=10)  ← ckpt/local/R26_cifar10.pt
    Local_R26_cifar100  → ResNet(num_classes=100) ← ckpt/local/R26_cifar100.pt

The returned module is wrapped with a ``Normalize`` layer so it accepts
images in the [0, 1] range produced by ``robustbench.data.load_cifar*c``
(matching how RobustBench's own ``Standard`` baseline is wrapped). The
training script ``train_R26.py`` already applies the same per-channel mean
and std via ``torchvision.transforms.Normalize`` so the R-26 weights see
identical statistics in both train and test.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from pycls.core.config import cfg as pycfg
from pycls.models.resnet import ResNet


CKPT_DIR_DEFAULT = Path("ckpt/local")

# Per-channel mean / std for [0,1]-scaled CIFAR images. Identical to the
# values used in train_R26.py.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class NormalizedModel(nn.Module):
    """Wraps a CIFAR backbone with a leading ``(x - mean) / std`` layer.

    The mean/std are registered as buffers so they move with ``.to(device)``
    and are saved/loaded transparently. The backbone is exposed as
    ``.backbone`` so TENT/Norm setup can iterate its BatchNorm2d modules
    (the wrapper itself has none, so ``collect_params`` still finds only
    the backbone's BN affine parameters).
    """

    def __init__(self, mean: tuple[float, float, float],
                 std: tuple[float, float, float], backbone: nn.Module):
        super().__init__()
        self.register_buffer(
            "_norm_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer(
            "_norm_std", torch.tensor(std).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self._norm_mean) / self._norm_std)


def build_R26(num_classes: int) -> nn.Module:
    """Construct a pycls R-26 with BatchNorm2d (required for TENT)."""
    pycfg.MODEL.TYPE = "resnet"
    pycfg.MODEL.DEPTH = 26
    pycfg.MODEL.NUM_CLASSES = num_classes
    pycfg.RESNET.TRANS_FUN = "basic_transform"
    pycfg.TRAIN.DATASET = "cifar10"
    pycfg.TEST.DATASET = "cifar10"
    pycfg.TRAIN.IM_SIZE = 32
    pycfg.TEST.IM_SIZE = 32
    return ResNet()


def _resolve(arch_key: str) -> tuple[int, Path, tuple, tuple]:
    if arch_key == "Local_R26_cifar10":
        return 10, CKPT_DIR_DEFAULT / "R26_cifar10.pt", CIFAR10_MEAN, CIFAR10_STD
    if arch_key == "Local_R26_cifar100":
        return 100, CKPT_DIR_DEFAULT / "R26_cifar100.pt", CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unknown local arch key: {arch_key!r}")


def load_local_model(arch_key: str, ckpt_path: str | None = None) -> nn.Module:
    """Build R-26, load its weights, and wrap with input normalization.

    The returned module accepts [0, 1]-scaled images (RobustBench convention)
    and internally applies ``(x - mean) / std`` before the backbone so the
    R-26 affine weights see the same statistics they were trained on.
    """
    num_classes, default_path, mean, std = _resolve(arch_key)
    path = Path(ckpt_path) if ckpt_path else default_path
    if not path.is_file():
        raise FileNotFoundError(
            f"R-26 checkpoint missing: {path}. "
            f"Run experiments/tent_replication/train_R26.py first.")
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    backbone = build_R26(num_classes)
    backbone.load_state_dict(weights, strict=True)
    return NormalizedModel(mean, std, backbone)
