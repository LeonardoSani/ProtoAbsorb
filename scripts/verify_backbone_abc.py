"""Verification script for the BackboneClassifier ABC refactor.

Run from anywhere:

    uv run python scripts/verify_backbone_abc.py

The script puts both the project root and `src/` on `sys.path` so that
`models.*`, `proto_absorb.*`, and `experiments.*` all resolve regardless of
whether the project has been pip-installed.

Exits with code 0 on success, non-zero on failure. Designed to be
self-contained — no pytest dependency.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import torch  # noqa: E402  (path setup must happen first)


def _ok(msg: str) -> None:
    print(f"OK   {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise AssertionError(msg)


def check_abc_unmet_raises() -> None:
    from models.base import BackboneClassifier

    try:
        BackboneClassifier()  # type: ignore[abstract]
    except TypeError:
        _ok("BackboneClassifier() raises TypeError when instantiated directly")
        return
    _fail("BackboneClassifier() did not raise TypeError")


def check_resnet18() -> None:
    from models.base import BackboneClassifier
    from models import build_resnet18

    m = build_resnet18()
    assert isinstance(m, BackboneClassifier), "ResNet18 must subclass BackboneClassifier"
    assert isinstance(m.feat_dim, int) and m.feat_dim == 512, f"feat_dim {m.feat_dim}"

    m.eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 32, 32)
        logits = m(x)
        assert logits.shape == (2, 10), f"logits {logits.shape}"
        logits2, feats = m(x, return_features=True)
        assert torch.equal(logits, logits2), "return_features path mismatch"
        assert feats.shape == (2, m.feat_dim), f"feats {feats.shape}"
    _ok("ResNet18 contract")


def check_resnet50() -> None:
    from models.base import BackboneClassifier
    from models import build_resnet50

    m = build_resnet50()
    assert isinstance(m, BackboneClassifier), "ResNet50Wrapper must subclass BackboneClassifier"
    assert m.feat_dim == 2048, f"feat_dim {m.feat_dim}"

    m.eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 224, 224)
        logits = m(x)
        assert logits.shape == (2, 1000), f"logits {logits.shape}"
        logits2, feats = m(x, return_features=True)
        assert torch.equal(logits, logits2)
        assert feats.shape == (2, 2048)
    _ok("ResNet50Wrapper contract")


def check_vit_small() -> None:
    from models.base import BackboneClassifier
    from models import build_vit_small

    m = build_vit_small()
    assert isinstance(m, BackboneClassifier), "ViTSmall must subclass BackboneClassifier"
    assert m.feat_dim == 384, f"feat_dim {m.feat_dim}"

    m.eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 224, 224)
        logits = m(x)
        assert logits.shape == (2, 10), f"logits {logits.shape}"
        logits2, feats = m(x, return_features=True)
        assert torch.equal(logits, logits2)
        assert feats.shape == (2, 384)
    _ok("ViTSmall contract")


def check_centroids_import() -> None:
    """The stale centroids.py:18 import must now resolve and the function
    must accept any BackboneClassifier."""

    import proto_absorb.centroids as c  # noqa: F401
    _ok("proto_absorb.centroids imports without ImportError")


def check_experiments_imports() -> None:
    """Every experiment script that previously imported from the removed
    proto_absorb.models path must now import cleanly."""

    for mod in (
        "experiments._common",
        "experiments.exp_bn_ablation",
        "experiments.exp_imagenet_baseline",
        "experiments.exp_imagenet_scale",
        "experiments.exp_step1_sanity",
        "experiments.exp_vit_backbone",
    ):
        __import__(mod)
        _ok(f"{mod} imports without ImportError")


def main() -> int:
    checks = [
        check_abc_unmet_raises,
        check_resnet18,
        check_resnet50,
        check_vit_small,
        check_centroids_import,
        check_experiments_imports,
    ]
    failed = 0
    for check in checks:
        try:
            check()
        except Exception:
            traceback.print_exc()
            failed += 1
    if failed:
        print(f"\n{failed} check(s) failed")
        return 1
    print("\nAll checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
