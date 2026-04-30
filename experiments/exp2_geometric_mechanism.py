"""Experiment 2 — Characterize the geometric mechanism.

Per-step we record, for every OOD sample in the batch, its L2 distance to the
nearest centroid (under both **frozen** and **dynamic** centroid regimes).
We then produce:

  2.1 Mean OOD-to-centroid distance vs steps (frozen solid, dynamic dashed).
  2.2 Distance distribution histograms at t in {0, 5, 10, 20}, ID vs OOD.
  2.3 Per-class absorption bar chart (which classes attract OOD?).
  2.4 Frozen vs dynamic centroid comparison (per-alpha panel).
  2.5 2D UMAP visualization with frames at t in {0, T/2, T}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    ALPHAS,
    CentroidBank,
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    TentConfig,
    ensure_dir,
    evaluate_batch,
    fresh_tent_model,
    get_device,
    get_logger,
    make_sampler,
    nearest_centroid_distances,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
    update_centroids_dynamic,
)


def run_single_run(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_steps: int, n_batches: int, batch_size: int, ood_name: str,
    severity: int, seed: int, lr: float, centroid_mode: str,
    base_centroids: CentroidBank, device: torch.device,
) -> dict:
    """Run TENT and record per-step distance traces under one centroid regime."""
    assert centroid_mode in {"frozen", "dynamic"}
    rng = np.random.default_rng(seed)
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)

    ood_dist_per_step: list[list[np.ndarray]] = [[] for _ in range(n_steps + 1)]
    id_dist_per_step: list[list[np.ndarray]] = [[] for _ in range(n_steps + 1)]
    absorbed_class_per_step: list[list[np.ndarray]] = [[] for _ in range(n_steps + 1)]

    for b in range(n_batches):
        cfg = TentConfig(lr=lr)
        model, opt = fresh_tent_model(ckpt_path, device, cfg)
        batch = sampler.next_batch(rng).to(device)
        is_ood = batch.is_ood.cpu().numpy().astype(bool)

        bank = base_centroids.clone()
        for t in range(n_steps + 1):
            ev = evaluate_batch(model, batch)
            feats = torch.from_numpy(ev["feats"]).float()
            cents = bank.centroids
            d, argmin = nearest_centroid_distances(feats, cents)
            d_np, am_np = d.cpu().numpy(), argmin.cpu().numpy()

            ood_dist_per_step[t].append(d_np[is_ood])
            id_dist_per_step[t].append(d_np[~is_ood])
            absorbed_class_per_step[t].append(am_np[is_ood])

            if t < n_steps:
                tent_step(model, opt, batch.images, cfg)
                if centroid_mode == "dynamic":
                    ev2 = evaluate_batch(model, batch)
                    f2 = torch.from_numpy(ev2["feats"]).float().to(device)
                    lbl = batch.labels
                    id_mask = ~batch.is_ood
                    if id_mask.any():
                        bank = update_centroids_dynamic(
                            bank, f2[id_mask], lbl[id_mask], momentum=0.9
                        )

    ood_distances = [np.concatenate(x) if x else np.array([]) for x in ood_dist_per_step]
    id_distances = [np.concatenate(x) if x else np.array([]) for x in id_dist_per_step]
    absorbed = [np.concatenate(x) if x else np.array([], dtype=np.int64)
                for x in absorbed_class_per_step]

    return {
        "ood_distances": ood_distances,         # list[T+1] of arrays
        "id_distances": id_distances,
        "absorbed_class": absorbed,
        "mean_ood": np.array([d.mean() if len(d) else np.nan for d in ood_distances]),
        "std_ood": np.array([d.std() if len(d) else np.nan for d in ood_distances]),
        "mean_id": np.array([d.mean() if len(d) else np.nan for d in id_distances]),
    }


def plot_mean_distance(results_by_alpha: dict, out_dir: Path) -> None:
    """Plot 2.1 + 2.4: mean OOD->centroid distance, frozen solid vs dynamic dashed."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results_by_alpha)))
    for color, (alpha, modes) in zip(colors, sorted(results_by_alpha.items())):
        if "frozen" in modes:
            m = modes["frozen"]["mean_ood"]; s = modes["frozen"]["std_ood"]
            steps = np.arange(len(m))
            ax.plot(steps, m, color=color, ls="-", lw=2,
                    label=fr"$\alpha={alpha}$ (frozen)")
            ax.fill_between(steps, m - s, m + s, color=color, alpha=0.10)
        if "dynamic" in modes:
            m = modes["dynamic"]["mean_ood"]
            steps = np.arange(len(m))
            ax.plot(steps, m, color=color, ls="--", lw=2,
                    label=fr"$\alpha={alpha}$ (dynamic)")
    ax.set_xlabel("Adaptation step $t$")
    ax.set_ylabel(r"$\bar d(t)$ — mean $L_2$ OOD $\rightarrow$ nearest centroid")
    ax.set_title("Plot 2.1 / 2.4 — Mean OOD-to-centroid distance over time")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_2.1_mean_distance_vs_steps.png")
    plt.close(fig)


def plot_distance_histograms(results: dict, out_dir: Path,
                             snapshot_steps: list[int]) -> None:
    """Plot 2.2: distance histograms (ID vs OOD) at selected timesteps."""
    import matplotlib.pyplot as plt
    n = len(snapshot_steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, snapshot_steps):
        if t >= len(results["ood_distances"]):
            continue
        ood = results["ood_distances"][t]
        idd = results["id_distances"][t]
        bins = 30
        if len(ood) and len(idd):
            lo = min(ood.min(), idd.min())
            hi = max(ood.max(), idd.max())
            edges = np.linspace(lo, hi, bins + 1)
            ax.hist(idd, bins=edges, color="seagreen", alpha=0.6, label="ID", density=True)
            ax.hist(ood, bins=edges, color="firebrick", alpha=0.6, label="OOD", density=True)
        ax.set_title(f"t = {t}")
        ax.set_xlabel(r"$\|\phi(x) - \mu_{c^*}\|_2$")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    fig.suptitle("Plot 2.2 — Distance distributions over time (frozen centroids)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_2.2_distance_histograms.png")
    plt.close(fig)


def plot_per_class_absorption(results: dict, out_dir: Path,
                              num_classes: int = 10) -> None:
    """Plot 2.3: bar chart of OOD->class absorption at t=0 vs t=T."""
    import matplotlib.pyplot as plt
    absorbed = results["absorbed_class"]
    if len(absorbed) == 0:
        return
    counts_t0 = np.bincount(absorbed[0], minlength=num_classes)
    counts_tT = np.bincount(absorbed[-1], minlength=num_classes)
    x = np.arange(num_classes)
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - w/2, counts_t0, width=w, color="steelblue", label="t = 0")
    ax.bar(x + w/2, counts_tT, width=w, color="coral",
           label=f"t = {len(absorbed) - 1}")
    ax.set_xticks(x)
    ax.set_xlabel("CIFAR-10 class index")
    ax.set_ylabel("# OOD samples nearest to this centroid")
    ax.set_title("Plot 2.3 — Per-class absorption (which classes attract OOD?)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_2.3_per_class_absorption.png")
    plt.close(fig)


def umap_snapshots(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_steps: int, batch_size: int, ood_name: str, severity: int,
    seed: int, lr: float, base_centroids: CentroidBank, device: torch.device,
    out_dir: Path,
) -> None:
    """Plot 2.5: 2D UMAP embedding at t=0, T/2, T."""
    try:
        import umap  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[warn] UMAP unavailable ({e}); skipping Plot 2.5.")
        return
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    cfg = TentConfig(lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batch = sampler.next_batch(rng).to(device)
    is_ood = batch.is_ood.cpu().numpy().astype(bool)
    labels = batch.labels.cpu().numpy()

    snaps = sorted({0, n_steps // 2, n_steps})
    feats_at: dict[int, np.ndarray] = {}
    for t in range(n_steps + 1):
        if t in snaps:
            ev = evaluate_batch(model, batch)
            feats_at[t] = ev["feats"]
        if t < n_steps:
            tent_step(model, opt, batch.images, cfg)

    fit_basis = np.concatenate(
        [feats_at[t] for t in snaps] + [base_centroids.centroids.numpy()], axis=0
    )
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
    reducer.fit(fit_basis)

    fig, axes = plt.subplots(1, len(snaps), figsize=(4.0 * len(snaps), 3.8), sharex=True, sharey=True)
    if len(snaps) == 1:
        axes = [axes]
    for ax, t in zip(axes, snaps):
        emb = reducer.transform(feats_at[t])
        cent_emb = reducer.transform(base_centroids.centroids.numpy())
        sc = ax.scatter(emb[~is_ood, 0], emb[~is_ood, 1],
                        c=labels[~is_ood], cmap="tab10", s=14,
                        vmin=0, vmax=9, alpha=0.85, label="ID")
        ax.scatter(emb[is_ood, 0], emb[is_ood, 1],
                   c="black", s=14, marker="x", alpha=0.85, label="OOD")
        ax.scatter(cent_emb[:, 0], cent_emb[:, 1], marker="*", s=140,
                   edgecolor="white", linewidth=0.8,
                   c=np.arange(10), cmap="tab10", vmin=0, vmax=9, label="centroid")
        ax.set_title(f"t = {t}")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(rf"Plot 2.5 — UMAP snapshots ($\alpha={alpha}$, {corruption})")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_2.5_umap_snapshots.png")
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/exp2")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hist-alpha", type=float, default=0.5,
                        help="Alpha used for histogram, per-class, and UMAP plots.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("exp2")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    base = CentroidBank.load(args.centroids)
    log.info(f"Loaded centroids: shape={tuple(base.centroids.shape)}, "
             f"cov={base.cov is not None}")

    if args.smoke:
        args.steps = min(args.steps, 6)
        args.batches = min(args.batches, 2)

    results_by_alpha: dict[float, dict] = {}
    for a in args.alphas:
        results_by_alpha[a] = {}
        for mode in ("frozen", "dynamic"):
            log.info(f"alpha={a} mode={mode}")
            res = run_single_run(
                args.ckpt, args.data_root, args.corruption, a,
                args.steps, args.batches, args.batch_size, args.ood,
                args.severity, args.seed + int(a * 1000) + (1 if mode == "dynamic" else 0),
                args.lr, mode, base, device,
            )
            results_by_alpha[a][mode] = res
            log.info(f"  mean_ood: {res['mean_ood'][0]:.3f} -> {res['mean_ood'][-1]:.3f}")

    plot_mean_distance(results_by_alpha, out_dir)

    hist_alpha = min(args.alphas, key=lambda a: abs(a - args.hist_alpha))
    chosen = results_by_alpha[hist_alpha]["frozen"]
    snapshots = sorted({0, min(5, args.steps), min(10, args.steps), args.steps})
    plot_distance_histograms(chosen, out_dir, snapshots)
    plot_per_class_absorption(chosen, out_dir)

    log.info("Generating UMAP snapshots ...")
    umap_snapshots(
        args.ckpt, args.data_root, args.corruption, hist_alpha,
        args.steps, args.batch_size, args.ood, args.severity,
        args.seed + 7777, args.lr, base, device, out_dir,
    )

    summary = {}
    for a, modes in results_by_alpha.items():
        summary[str(a)] = {
            mode: {"mean_ood": r["mean_ood"].tolist(),
                   "std_ood": r["std_ood"].tolist(),
                   "mean_id": r["mean_id"].tolist()}
            for mode, r in modes.items()
        }
    save_json(summary, out_dir / "summary.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
