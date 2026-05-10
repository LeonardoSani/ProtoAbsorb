"""Step 10 — Streaming validation (state-carryover).

Tests whether the paired contamination gap persists under sequential
adaptation with no model reset between batches.

Run:
    uv run python -m experiments.exp_step10_streaming \
        --ckpt checkpoints/resnet18_cifar10.pt \
        --data-root data --out results/step10 --seeds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    MixedBatch,
    MixedBatchSampler,
    TentConfig,
    TentVariant,
    EataConfig,
    EataState,
    auroc_from_eval,
    build_id_pool,
    build_ood_pool,
    ensure_dir,
    evaluate_batch,
    eata_step,
    fresh_tent_model,
    fpr95_from_eval,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    make_sampler,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
)

CONDITIONS = ["no_tta", "id_only", "mixed"]
METHODS    = ["tent", "eta"]
OODS       = ["svhn", "dtd"]
ALPHAS     = [0.9, 0.5]
N_BATCHES  = 100
N_STEPS    = 10          # adaptation steps per batch (same as rest of paper)
CORRUPTION = "gaussian_noise"


def draw_stream(data_root: str, corruption: str, alpha: float,
                ood_name: str, batch_size: int, n_batches: int,
                seed: int, severity: int = 5) -> list[MixedBatch]:
    """Draw a fixed sequence of mixed batches (same sequence for all conditions)."""
    sampler = make_sampler(data_root, corruption, alpha, ood_name,
                           batch_size, severity)
    rng_seq = np.random.default_rng(seed)
    return [sampler.next_batch(rng_seq) for _ in range(n_batches)]


def run_streaming_condition(
    ckpt_path: str,
    device: torch.device,
    batches: list[MixedBatch],
    condition: str,
    n_steps: int,
    lr: float,
    tta_method: str = "tent",
) -> dict:
    """Adapt sequentially across all batches WITHOUT resetting model state."""
    cfg = TentConfig(lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg)

    eata_state: EataState | None = None
    eata_cfg: EataConfig | None = None
    if tta_method in {"eta", "eata"}:
        eata_state = EataState()
        eata_cfg = EataConfig(lr=lr)
        from proto_absorb.tent import collect_bn_params
        params, _ = collect_bn_params(model)
        opt = torch.optim.SGD(params, lr=eata_cfg.lr, momentum=eata_cfg.momentum)

    auroc_per_batch: list[float] = []
    id_acc_per_batch: list[float] = []

    for batch in batches:
        batch = batch.to(device)

        if condition != "no_tta":
            for _ in range(n_steps):
                if condition == "id_only":
                    id_mask = ~batch.is_ood
                    if id_mask.any():
                        imgs = batch.images[id_mask]
                        if tta_method in {"eta", "eata"}:
                            eata_step(model, opt, imgs, eata_state, eata_cfg)
                        else:
                            tent_step(model, opt, imgs, cfg)
                else:  # mixed
                    if tta_method in {"eta", "eata"}:
                        eata_step(model, opt, batch.images, eata_state, eata_cfg)
                    else:
                        tent_step(model, opt, batch.images, cfg)

        ev = evaluate_batch(model, batch)
        auroc_per_batch.append(auroc_from_eval(ev, "msp"))
        id_acc_per_batch.append(id_accuracy_from_eval(ev))

    return {
        "auroc_per_batch": auroc_per_batch,
        "id_acc_per_batch": id_acc_per_batch,
        "auroc_mean": float(np.mean(auroc_per_batch)),
        "id_acc_mean": float(np.mean(id_acc_per_batch)),
        "auroc_final": auroc_per_batch[-1],
    }


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    ood_name: str, tta_method: str, seeds: list[int],
    n_batches: int, n_steps: int, batch_size: int, lr: float,
    device: torch.device,
) -> dict:
    """Run one cell (method × OOD × alpha) across all seeds."""
    seed_results: list[dict] = []

    for seed in seeds:
        set_seed(seed)
        batches = draw_stream(data_root, corruption, alpha, ood_name,
                              batch_size, n_batches, seed)

        cond_out: dict[str, dict] = {}
        for cond in CONDITIONS:
            cond_out[cond] = run_streaming_condition(
                ckpt_path, device, batches, cond, n_steps, lr, tta_method
            )
        seed_results.append({"seed": seed, "conditions": cond_out})

    agg: dict[str, dict] = {}
    for cond in CONDITIONS:
        auroc_matrix = np.array([
            r["conditions"][cond]["auroc_per_batch"] for r in seed_results
        ])  # (n_seeds, n_batches)
        agg[cond] = {
            "auroc_mean_over_seeds": auroc_matrix.mean(axis=0).tolist(),
            "auroc_std_over_seeds": auroc_matrix.std(axis=0).tolist(),
            "time_avg_auroc_mean": float(auroc_matrix.mean()),
            "time_avg_auroc_std": float(auroc_matrix.std()),
            "final_auroc_mean": float(auroc_matrix[:, -1].mean()),
            "final_auroc_std": float(auroc_matrix[:, -1].std()),
        }

    id_only_mat = np.array([r["conditions"]["id_only"]["auroc_per_batch"] for r in seed_results])
    mixed_mat   = np.array([r["conditions"]["mixed"]["auroc_per_batch"] for r in seed_results])
    gap_mat = id_only_mat - mixed_mat

    gap_per_seed = gap_mat.mean(axis=1)  # (n_seeds,)
    zero_vec = np.zeros(len(seeds))
    t_stat, p_val = paired_t_test(gap_per_seed.tolist(), zero_vec.tolist())

    return {
        "conditions": agg,
        "paired_gap": {
            "mean_per_batch": gap_mat.mean(axis=0).tolist(),
            "time_avg_mean": float(gap_mat.mean()),
            "time_avg_std": float(gap_mat.std()),
            "t": t_stat,
            "p": p_val,
            "n_seeds": len(seeds),
        },
        "config": {
            "corruption": corruption, "alpha": alpha, "ood": ood_name,
            "method": tta_method, "n_batches": n_batches, "n_steps": n_steps,
        },
    }


def make_trajectory_figure(all_results: dict, out_path: Path) -> None:
    """Trajectory plot: AUROC vs batch index for representative cell.
    Representative: tent, svhn, alpha=0.9, gaussian_noise.
    """
    setup_matplotlib()
    import matplotlib.pyplot as plt

    key = ("tent", "svhn", 0.9)
    if key not in all_results:
        return

    r = all_results[key]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    colors = {"no_tta": "tab:gray", "id_only": "tab:blue", "mixed": "tab:red"}
    n_batches = len(r["conditions"]["no_tta"]["auroc_mean_over_seeds"])
    xs = list(range(1, n_batches + 1))

    ax = axes[0]
    for cond in CONDITIONS:
        ys = r["conditions"][cond]["auroc_mean_over_seeds"]
        sd = r["conditions"][cond]["auroc_std_over_seeds"]
        ax.plot(xs, ys, color=colors[cond], label=cond, lw=1.5)
        ax.fill_between(xs,
                        [y - s for y, s in zip(ys, sd)],
                        [y + s for y, s in zip(ys, sd)],
                        color=colors[cond], alpha=0.15)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("MSP-AUROC")
    ax.set_title("AUROC trajectory (TENT, SVHN, α=0.9)")
    ax.legend()

    ax = axes[1]
    gap_ys = r["paired_gap"]["mean_per_batch"]
    ax.plot(xs, gap_ys, color="tab:purple", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Paired gap (id_only − mixed)")
    ax.set_title(f"Gap persists? (p={r['paired_gap']['p']:.3f})")

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory figure: {out_path}")


def make_summary_table(all_results: dict, out_path: Path) -> None:
    """Print 8-cell summary table and save as JSON."""
    rows = []
    for (method, ood, alpha), r in sorted(all_results.items()):
        row = {
            "method": method,
            "ood": ood,
            "alpha": alpha,
            "no_tta_auroc": r["conditions"]["no_tta"]["time_avg_auroc_mean"],
            "id_only_auroc": r["conditions"]["id_only"]["time_avg_auroc_mean"],
            "mixed_auroc": r["conditions"]["mixed"]["time_avg_auroc_mean"],
            "paired_gap_mean": r["paired_gap"]["time_avg_mean"],
            "paired_gap_p": r["paired_gap"]["p"],
            "n_seeds": r["paired_gap"]["n_seeds"],
        }
        rows.append(row)

    save_json(rows, out_path)
    print("\n=== Streaming Summary (8 cells) ===")
    print(f"{'Method':8} {'OOD':6} {'α':4}  {'NoTTA':6}  {'IDonly':6}  {'Mixed':6}  {'Gap':6}  p")
    for r in rows:
        print(f"{r['method']:8} {r['ood']:6} {r['alpha']:4.1f}  "
              f"{r['no_tta_auroc']:.3f}   {r['id_only_auroc']:.3f}   "
              f"{r['mixed_auroc']:.3f}   {r['paired_gap_mean']:.3f}  "
              f"{r['paired_gap_p']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/resnet18_cifar10.pt")
    p.add_argument("--data-root",  default="data")
    p.add_argument("--out",        default="results/step10")
    p.add_argument("--seeds",      type=int, default=5)
    p.add_argument("--batches",    type=int, default=N_BATCHES)
    p.add_argument("--steps",      type=int, default=N_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()

    device = get_device()
    log    = get_logger("step10")
    out    = Path(args.out)
    ensure_dir(out)
    seeds  = list(range(args.seeds))

    all_results: dict = {}

    for method in METHODS:
        for ood in OODS:
            for alpha in ALPHAS:
                key = (method, ood, alpha)
                log.info(f"Running: method={method} ood={ood} alpha={alpha}")
                result = run_cell(
                    ckpt_path=args.ckpt,
                    data_root=args.data_root,
                    corruption=CORRUPTION,
                    alpha=alpha,
                    ood_name=ood,
                    tta_method=method,
                    seeds=seeds,
                    n_batches=args.batches,
                    n_steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                )
                all_results[key] = result
                cell_file = out / f"{method}_{ood}_alpha{alpha}.json"
                save_json(result, cell_file)
                log.info(f"  gap={result['paired_gap']['time_avg_mean']:.3f} "
                         f"p={result['paired_gap']['p']:.4f}")

    json_results = {f"{m}_{o}_{a}": v for (m, o, a), v in all_results.items()}
    save_json(json_results, out / "streaming_results.json")

    make_summary_table(all_results, out / "streaming_summary.json")
    make_trajectory_figure(all_results, out / "streaming_trajectory.png")
    log.info(f"Done. Results in {out}/")


if __name__ == "__main__":
    main()
