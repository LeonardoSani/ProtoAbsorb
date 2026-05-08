"""ResNet-18 adapted for 32x32 CIFAR images, with feature-extraction hooks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18(nn.Module):
    """ResNet-18 with the 'CIFAR' stem (3x3 conv, no maxpool)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.feat_dim = 512 * BasicBlock.expansion

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.fc(feats)
        if return_features:
            return logits, feats
        return logits


def build_resnet18(num_classes: int = 10) -> ResNet18:
    return ResNet18(num_classes=num_classes)


def _replace_bn_with_in(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            in_layer = nn.InstanceNorm2d(
                child.num_features, affine=True, track_running_stats=False
            )
            if child.affine:
                in_layer.weight.data.copy_(child.weight.data)
                in_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, in_layer)
        else:
            _replace_bn_with_in(child)


def bn_to_in(model: nn.Module) -> nn.Module:
    """Return deep copy of model with every BatchNorm2d replaced by InstanceNorm2d.

    Affine parameters (weight/bias) are copied from BN to IN so the initial
    forward pass is close to the BN model's output. IN computes per-sample
    statistics regardless of batch composition, removing the shared-statistics
    contamination pathway that BN introduces.
    """
    import copy
    model = copy.deepcopy(model)
    _replace_bn_with_in(model)
    return model


class ViTSmall(nn.Module):
    """ViT-S/16 (timm) with return_features=True interface. Expects 224×224 input."""

    feat_dim = 384

    def __init__(self, num_classes: int = 10):
        super().__init__()
        import timm
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


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: Optional[str] = None) -> nn.Module:
    state = torch.load(ckpt_path, map_location=map_location or "cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model
