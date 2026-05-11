"""Console-script entry points for proto-absorb."""

from __future__ import annotations


def train_main() -> None:
    from ..proto_absorb.train import cli

    cli()


def centroids_main() -> None:
    from ..proto_absorb.compute_centroids import cli

    cli()


def exp1_main() -> None:
    from experiments.exp1_failure_mode import cli

    cli()


def exp2_main() -> None:
    from experiments.exp2_geometric_mechanism import cli

    cli()


def exp3_main() -> None:
    from experiments.exp3_fixes import cli

    cli()


def exp7_main() -> None:
    from experiments.exp7_vector_field import cli

    cli()


# ---------------------------------------------------------------------------
# Reframed step-N entry points (canonical going forward, see doc/reframe.md).
# ---------------------------------------------------------------------------


def step1_main() -> None:
    from experiments.exp_step1_sanity import cli

    cli()


def step2_main() -> None:
    from experiments.exp_step2_contamination import cli

    cli()


def step3_main() -> None:
    from experiments.exp_step3_detector_breadth import cli

    cli()


def step4_main() -> None:
    from experiments.exp_step4_mechanism import cli

    cli()


def step5_main() -> None:
    from experiments.exp_step5_generalization import cli

    cli()


def step6_main() -> None:
    from experiments.exp_step6_fix_a import cli

    cli()


def train_vit_main() -> None:
    from ..proto_absorb.train_vit import cli

    cli()


def vit_backbone_main() -> None:
    from experiments.exp_vit_backbone import cli

    cli()
