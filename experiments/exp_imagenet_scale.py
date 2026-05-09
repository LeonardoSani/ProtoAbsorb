"""ImageNet-Scale Validation — 24-cell paired protocol (Steps 2–4).

Replicates the CIFAR-10 contamination result at ImageNet scale:
  ResNet-50 (torchvision pretrained), ImageNet-C (ID), NINCO / DTD (OOD).

Run matrix:
  OOD:         ninco, dtd
  Methods:     tent, eata
  alpha:       0.9, 0.5
  Corruptions: gaussian_noise, fog, jpeg_compression  (sev-5)
  n_batches:   30
  Metrics:     MSP-AUROC + FPR95

Total cells: 2 × 2 × 2 × 3 = 24, ~3–5 GPU-hours on a single A100.

Run:
  python -m experiments.exp_imagenet_scale \\
    --data-root data --out results/imagenet_scale --batches 30
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    MixedBatchSampler,
    bootstrap_ci,
    draw_paired_batches,
    ensure_dir,
    fpr95_from_eval,
    auroc_from_eval,
    evaluate_batch,
    get_device,
    get_logger,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    run_condition_on_batch,
    EataState, EataConfig,
    TentConfig, TentVariant,
    collect_bn_params, configure_tent_model, make_optimizer,
)
from proto_absorb.data import ImageNetC, get_ood_dataset_224
from proto_absorb.models import build_resnet50


CONDITIONS = ("no_tta", "id_only", "mixed")

# EATA entropy threshold for 1000-class ImageNet: 0.4 * ln(1000)
IMAGENET_EATA_E0 = 0.4 * float(np.log(1000))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_resnet50_factory(
    device: torch.device,
    lr: float,
    tta_method: str = "tent",
    eata_e0: float = IMAGENET_EATA_E0,
):
    """Return a factory callable that yields a fresh (model, optimizer) pair.

    Loads weights once; deep-copies per call to avoid re-downloading.
    """
    _base = build_resnet50()  # loaded from torchvision cache once

    def factory():
        model = copy.deepcopy(_base).to(device)
        configure_tent_model(model)
        params, _ = collect_bn_params(model)
        if tta_method == "eata":
            opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
        else:
            cfg = TentConfig(lr=lr)
            opt = make_optimizer(params, cfg)
        return model, opt

    return factory


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    data_root: str, corruption: str, alpha: float, ood_name: str,
    n_batches: int, n_steps: int, batch_size: int,
    tta_method: str, master_seed: int, lr: float,
    device: torch.device,
) -> dict:
    """Run all three conditions on shared batches; return per-batch summaries."""
    id_ds = ImageNetC(data_root, corruption=corruption, severity=5)
    ood_ds = get_ood_dataset_224(ood_name, data_root)
    sampler = MixedBatchSampler(
        id_dataset=id_ds, ood_dataset=ood_ds,
        alpha=alpha, batch_size=batch_size,
    )
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    factory = make_resnet50_factory(
        device=device, lr=lr, tta_method=tta_method,
        eata_e0=IMAGENET_EATA_E0,
    )

    per_cond: dict[str, dict[str, list]] = {
        c: {
            "auroc_0": [], "auroc_T": [], "delta_auroc": [],
            "fpr95_0": [], "fpr95_T": [], "delta_fpr95": [],
            "id_acc_0": [], "id_acc_T": [],
        }
        for c in CONDITIONS
    }

    for batch in batches:
        for cond in CONDITIONS:
            res = run_condition_on_batch(
                ckpt_path="",           # unused when model_factory is provided
                device=device,
                batch=batch,
                condition=cond,
                n_steps=n_steps,
                lr=lr,
                tta_method=tta_method,
                model_factory=factory,
                eata_e0=IMAGENET_EATA_E0,
            )
            a0, aT = float(res["msp_auroc"][0]), float(res["msp_auroc"][-1])
            f0, fT = float(res["msp_fpr95"][0]),  float(res["msp_fpr95"][-1])
            i0, iT = float(res["id_acc"][0]),      float(res["id_acc"][-1])
            per_cond[cond]["auroc_0"].append(a0)
            per_cond[cond]["auroc_T"].append(aT)
            per_cond[cond]["delta_auroc"].append(aT - a0)
            per_cond[cond]["fpr95_0"].append(f0)
            per_cond[cond]["fpr95_T"].append(fT)
            per_cond[cond]["delta_fpr95"].append(fT - f0)
            per_cond[cond]["id_acc_0"].append(i0)
            per_cond[cond]["id_acc_T"].append(iT)

    summary: dict = {}
    for cond in CONDITIONS:
        d = per_cond[cond]
        m_auroc, lo_a, hi_a = bootstrap_ci(d["delta_auroc"], seed=master_seed)
        m_fpr95, lo_f, hi_f = bootstrap_ci(d["delta_fpr95"], seed=master_seed + 1)
        summary[cond] = {
            "auroc_T_mean":     float(np.mean(d["auroc_T"])),
            "delta_auroc_mean": m_auroc, "delta_auroc_lo": lo_a, "delta_auroc_hi": hi_a,
            "fpr95_T_mean":     float(np.mean(d["fpr95_T"])),
            "delta_fpr95_mean": m_fpr95, "delta_fpr95_lo": lo_f, "delta_fpr95_hi": hi_f,
            "id_acc_T_mean":    float(np.mean(d["id_acc_T"])),
            "n_batches":        len(d["delta_auroc"]),
            "delta_auroc_per_batch": d["delta_auroc"],
            "delta_fpr95_per_batch": d["delta_fpr95"],
        }

    t_stat, p_val = paired_t_test(
        per_cond["mixed"]["delta_auroc"], per_cond["id_only"]["delta_auroc"]
    )
    summary["paired_test_mixed_vs_id_only"] = {"t": t_stat, "p": p_val}
    return summary


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict, log) -> None:
    """Print a compact summary: AUROC(T) and FPR95(T) per cell."""
    header = f"{'corruption':20s} {'ood':8s} {'method':6s} {'alpha':5s} {'cond':8s}  AUROC(T)  FPR95(T)  Δ-AUROC  p"
    log.info(header)
    log.info("-" * len(header))
    for key, cell in all_results.items():
        corr, ood, method, alpha_str = key.split("|")
        p = cell.get("paired_test_mixed_vs_id_only", {}).get("p", float("nan"))
        for cond in CONDITIONS:
            s = cell[cond]
            log.info(
                f"{corr:20s} {ood:8s} {method:6s} {alpha_str:5s} {cond:8s}  "
                f"{s['auroc_T_mean']:.4f}    {s['fpr95_T_mean']:.4f}    "
                f"{s['delta_auroc_mean']:+.4f}  "
                + (f"p={p:.3g}" if cond == "mixed" else "")
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/imagenet_scale")
    parser.add_argument("--corruptions", nargs="+",
                        default=["fog", "jpeg_compression"])
    parser.add_argument("--oods", nargs="+", default=["ninco"])
    parser.add_argument("--methods", nargs="+", default=["tent", "eata"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.9, 0.5])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 3 batches, 2 steps.")
    args = parser.parse_args()

    log = get_logger("imagenet_scale")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps = 2
        # only override if user didn't explicitly set these
        if args.corruptions == ["fog", "jpeg_compression"]:
            args.corruptions = ["fog"]
        if args.oods == ["ninco"]:
            pass  # already single OOD
        if args.methods == ["tent", "eata"]:
            args.methods = ["tent"]
        if args.alphas == [0.9, 0.5]:
            args.alphas = [0.9]

    total_cells = (len(args.corruptions) * len(args.oods)
                   * len(args.methods) * len(args.alphas))
    log.info(
        f"Matrix: {len(args.corruptions)} corruptions × {len(args.oods)} OODs × "
        f"{len(args.methods)} methods × {len(args.alphas)} alphas = {total_cells} cells"
    )
    log.info(f"  n_batches={args.batches}, n_steps={args.steps}, batch_size={args.batch_size}")

    all_results: dict = {}
    cell_idx = 0

    for corruption in args.corruptions:
        for ood_name in args.oods:
            for method in args.methods:
                for alpha in args.alphas:
                    cell_idx += 1
                    key = f"{corruption}|{ood_name}|{method}|{alpha}"
                    log.info(
                        f"[{cell_idx}/{total_cells}] "
                        f"corruption={corruption}  ood={ood_name}  "
                        f"method={method}  alpha={alpha}"
                    )
                    master_seed = (
                        args.seed * 7919
                        + abs(hash(corruption)) % 997
                        + abs(hash(ood_name)) % 997
                        + abs(hash(method)) % 997
                        + int(alpha * 1000)
                    )
                    cell = run_cell(
                        data_root=args.data_root,
                        corruption=corruption,
                        alpha=alpha,
                        ood_name=ood_name,
                        n_batches=args.batches,
                        n_steps=args.steps,
                        batch_size=args.batch_size,
                        tta_method=method,
                        master_seed=master_seed,
                        lr=args.lr,
                        device=device,
                    )
                    all_results[key] = cell

                    # Quick per-cell log
                    p = cell["paired_test_mixed_vs_id_only"]["p"]
                    for cond in CONDITIONS:
                        s = cell[cond]
                        log.info(
                            f"  {cond:8s}  AUROC(T)={s['auroc_T_mean']:.4f}  "
                            f"FPR95(T)={s['fpr95_T_mean']:.4f}  "
                            f"Δ-AUROC={s['delta_auroc_mean']:+.4f} "
                            f"[{s['delta_auroc_lo']:+.4f}, {s['delta_auroc_hi']:+.4f}]"
                            + (f"  p={p:.3g}" if cond == "mixed" else "")
                        )

    print_summary_table(all_results, log)

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "imagenet_scale_results.json")
    log.info(f"Saved to {out_dir / 'imagenet_scale_results.json'}")

    # Count how many "mixed worse than id_only" cells and p<0.05 cells
    n_mixed_worse = 0
    n_sig = 0
    for cell in all_results.values():
        delta_mixed = cell["mixed"]["delta_auroc_mean"]
        delta_id    = cell["id_only"]["delta_auroc_mean"]
        p = cell["paired_test_mixed_vs_id_only"]["p"]
        if delta_mixed < delta_id:
            n_mixed_worse += 1
        if p < 0.05 and delta_mixed < delta_id:
            n_sig += 1
    log.info(
        f"\nSummary: mixed worse than id_only in {n_mixed_worse}/{len(all_results)} cells; "
        f"p<0.05 in {n_sig}/{len(all_results)} cells."
    )


if __name__ == "__main__":
    cli()
