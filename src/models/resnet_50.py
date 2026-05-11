import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ResNet50Wrapper(nn.Module):
    """torchvision ResNet-50 with return_features=True interface.

    Exposes the same (logits, feats) forward signature as ResNet18 / ViTSmall
    so all existing _common.py helpers work unchanged.
    """

    feat_dim = 2048

    def __init__(self):
        super().__init__()

        r = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = r.conv1
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.avgpool = r.avgpool
        self.fc = r.fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.fc(feats)
        if return_features:
            return logits, feats
        return logits


def build_resnet50() -> ResNet50Wrapper:
    return ResNet50Wrapper()
