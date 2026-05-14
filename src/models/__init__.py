"""Public surface for the `models` package.

Importers that previously did `from proto_absorb.models import …` should now
do `from models import …`. The re-exports below cover every symbol the
codebase previously consumed from the old location.
"""

from .base import BackboneClassifier
from .models import BasicBlock, load_checkpoint
from .resnet_18 import ResNet18, build_resnet18, bn_to_in
from .resnet_50 import ResNet50Wrapper, build_resnet50
from .vit_small import ViTSmall, build_vit_small

__all__ = [
    "BackboneClassifier",
    "BasicBlock",
    "ResNet18",
    "ResNet50Wrapper",
    "ViTSmall",
    "bn_to_in",
    "build_resnet18",
    "build_resnet50",
    "build_vit_small",
    "load_checkpoint",
]
