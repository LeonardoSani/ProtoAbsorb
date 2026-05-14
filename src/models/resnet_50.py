import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from models.base import BackboneClassifier


class ResNet50Wrapper(BackboneClassifier):
    """torchvision ResNet-50 conforming to the BackboneClassifier ABC."""

    feat_dim: int = 2048

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

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


def build_resnet50() -> ResNet50Wrapper:
    return ResNet50Wrapper()
