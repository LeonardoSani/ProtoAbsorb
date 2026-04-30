"""Experiment 7 (Optional) — Latent space vector field.

Captures the per-step representation residual
    Delta_phi(x) = phi_{theta_{t+k}}(x) - phi_{theta_t}(x)

and produces:
  7.1 Quiver plot in 2D UMAP space (per-sample drift arrows).
  7.2 Drift magnitude vs distance to nearest centroid (scatter).
  7.3 Cosine alignment between Delta_phi and the direction toward the nearest
      centroid (histogram).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
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
)


def collect_step_features(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_steps: int, batch_size: int, ood_name: str, severity: int,
    seed: int, lr: float, device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Run TENT for ``n_steps`` and return (features_per_step, is_ood, labels)."""
    rng = np.random.default_rng(seed)
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    cfg = TentConfig(lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    batch = sampler.next_batch(rng).to(device)

    feats_per_step: list[np.ndarray] = []
    for t in range(n_steps + 1):
        ev = evaluate_batch(model, batch)
        feats_per_step.append(ev["feats"])
        if t < n_steps:
            tent_step(model, opt, batch.images, cfg)
    return feats_per_step, batch.is_ood.cpu().numpy().astype(bool), batch.labels.cpu().numpy()


def quiver_plot(
    feats_per_step: list[np.ndarray], is_ood: np.ndarray,
    labels: np.ndarray, centroids: np.ndarray,
    t: int, k: int, seed: int, out_path: Path,
) -> None:
    try:
        import umap  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[warn] UMAP unavailable ({e}); skipping quiver plot.")
        return
    import matplotlib.pyplot as plt
    if t + k >= len(feats_per_step):
        return
    f_t = feats_per_step[t]
    f_tk = feats_per_step[t + k]
    fit = np.concatenate([f_t, f_tk, centroids], axis=0)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed).fit(fit)
    p_t = reducer.transform(f_t)
    p_tk = reducer.transform(f_tk)
    p_c = reducer.transform(centroids)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(p_t[~is_ood, 0], p_t[~is_ood, 1], c=labels[~is_ood], cmap="tab10",
               s=14, alpha=0.85, vmin=0, vmax=9, label="ID")
    ax.scatter(p_t[is_ood, 0], p_t[is_ood, 1], c="black", s=14, marker="x",
               alpha=0.85, label="OOD")
    dx, dy = p_tk[:, 0] - p_t[:, 0], p_tk[:, 1] - p_t[:, 1]
    color = np.where(is_ood, "firebrick", "steelblue")
    ax.quiver(p_t[:, 0], p_t[:, 1], dx, dy,
              color=color, angles="xy", scale_units="xy", scale=1.0,
              width=0.003, alpha=0.55)
    ax.scatter(p_c[:, 0], p_c[:, 1], marker="*", s=180, c=np.arange(len(centroids)),
               cmap="tab10", vmin=0, vmax=9, edgecolor="white", linewidth=0.7,
               label="centroid")
    ax.set_title(rf"Plot 7.1 — Quiver in UMAP space  ($t={t}\to{t+k}$)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def drift_magnitude_scatter(
    feats_per_step: list[np.ndarray], is_ood: np.ndarray,
    centroids: np.ndarray, t: int, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    if t + 1 >= len(feats_per_step):
        return
    feats_t = torch.from_numpy(feats_per_step[t]).float()
    delta = feats_per_step[t + 1] - feats_per_step[t]
    drift_mag = np.linalg.norm(delta, axis=1)
    cents = torch.from_numpy(centroids).float()
    d, _ = nearest_centroid_distances(feats_t, cents)
    d_np = d.numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(d_np[~is_ood], drift_mag[~is_ood], c="steelblue", s=18, alpha=0.7, label="ID")
    ax.scatter(d_np[is_ood], drift_mag[is_ood], c="firebrick", s=22,
               alpha=0.85, marker="x", label="OOD")
    ax.set_xlabel(r"$\|\phi(x) - \mu_{c^*}\|_2$ at step $t$")
    ax.set_ylabel(r"$\|\Delta\phi^{(t \to t+1)}\|_2$")
    ax.set_title(f"Plot 7.2 — Drift magnitude vs centroid distance (t={t})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cosine_alignment_plot(
    feats_per_step: list[np.ndarray], is_ood: np.ndarray,
    centroids: np.ndarray, t: int, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    if t + 1 >= len(feats_per_step):
        return
    feats_t = torch.from_numpy(feats_per_step[t]).float()
    delta = torch.from_numpy(feats_per_step[t + 1] - feats_per_step[t]).float()
    cents = torch.from_numpy(centroids).float()
    _, argmin = nearest_centroid_distances(feats_t, cents)
    target_dir = cents[argmin] - feats_t
    cos = torch.nn.functional.cosine_similarity(delta, target_dir, dim=1).numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(cos[~is_ood], bins=bins, alpha=0.6, color="steelblue",
            label="ID", density=True)
    ax.hist(cos[is_ood], bins=bins, alpha=0.6, color="firebrick",
            label="OOD", density=True)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"cos($\Delta\phi$, $\mu_{c^*} - \phi(x)$)")
    ax.set_ylabel("density")
    ax.set_title(f"Plot 7.3 — Cosine alignment of drift with attraction direction (t={t})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/exp7")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 5, 10, 15])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("exp7")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    base = CentroidBank.load(args.centroids)
    centroids_np = base.centroids.numpy()

    if args.smoke:
        args.steps = max(args.steps, 6)
        args.snapshots = [0, args.steps // 2]
        args.ks = [1]

    feats_per_step, is_ood, labels = collect_step_features(
        args.ckpt, args.data_root, args.corruption, args.alpha,
        args.steps, args.batch_size, args.ood, args.severity,
        args.seed, args.lr, device,
    )

    for t in args.snapshots:
        if t > args.steps:
            continue
        for k in args.ks:
            if t + k > args.steps:
                continue
            log.info(f"quiver: t={t} k={k}")
            quiver_plot(feats_per_step, is_ood, labels, centroids_np,
                        t=t, k=k, seed=args.seed,
                        out_path=out_dir / f"plot_7.1_quiver_t{t}_k{k}.png")

    for t in args.snapshots:
        if t + 1 > args.steps:
            continue
        log.info(f"drift mag: t={t}")
        drift_magnitude_scatter(feats_per_step, is_ood, centroids_np, t,
                                out_dir / f"plot_7.2_drift_magnitude_t{t}.png")
        log.info(f"cosine alignment: t={t}")
        cosine_alignment_plot(feats_per_step, is_ood, centroids_np, t,
                              out_dir / f"plot_7.3_cosine_alignment_t{t}.png")

    save_json({"snapshots": args.snapshots, "ks": args.ks,
               "alpha": args.alpha, "corruption": args.corruption,
               "n_steps": args.steps},
              out_dir / "config.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
