"""Console-script entry points for proto-absorb."""

from __future__ import annotations


def train_main() -> None:
    from .train import cli
    cli()


def centroids_main() -> None:
    from .compute_centroids import cli
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
