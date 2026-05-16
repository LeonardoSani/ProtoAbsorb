"""Driver for CIFAR-10-C / CIFAR-100-C TENT replication.

Mirrors the upstream cifar10c.py but dispatches on cfg.CORRUPTION.DATASET so
the same code reproduces both Table 2 rows. Writes one JSON per
(config, severity, corruption) under <out_dir>.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robustbench.data import load_cifar10c, load_cifar100c  # noqa: E402
from robustbench.model_zoo.enums import ThreatModel  # noqa: E402
from robustbench.utils import clean_accuracy as accuracy  # noqa: E402
from robustbench.utils import load_model  # noqa: E402

import tent_official as tent  # noqa: E402
import norm_official as norm  # noqa: E402
from conf import cfg, load_cfg_fom_args, reset_cfg  # noqa: E402
from pycls_loader import load_local_model  # noqa: E402

logger = logging.getLogger(__name__)


def _setup_source(model):
    model.eval()
    return model


def _setup_norm(model):
    return norm.Norm(model)


def _setup_optimizer(params):
    if cfg.OPTIM.METHOD == "Adam":
        return optim.Adam(
            params, lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD,
        )
    if cfg.OPTIM.METHOD == "SGD":
        return optim.SGD(
            params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM,
            dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WD,
            nesterov=cfg.OPTIM.NESTEROV,
        )
    raise NotImplementedError(cfg.OPTIM.METHOD)


def _setup_tent(model):
    model = tent.configure_model(model)
    params, _ = tent.collect_params(model)
    optimizer = _setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC)


def _loader_for(dataset: str):
    if dataset == "cifar10":
        return load_cifar10c
    if dataset == "cifar100":
        return load_cifar100c
    raise ValueError(f"unknown dataset: {dataset}")


def run_from_cfg(cfg_path: str, out_dir: str) -> dict:
    reset_cfg()
    cfg.defrost()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.MODEL.ARCH.startswith("Local_"):
        base = load_local_model(cfg.MODEL.ARCH).to(device)
    else:
        base = load_model(
            cfg.MODEL.ARCH, cfg.CKPT_DIR,
            cfg.CORRUPTION.DATASET, ThreatModel.corruptions,
        ).to(device)

    adapt = cfg.MODEL.ADAPTATION
    if adapt == "source":
        model = _setup_source(base)
    elif adapt == "norm":
        model = _setup_norm(base)
    elif adapt == "tent":
        model = _setup_tent(base)
    else:
        raise ValueError(f"unknown adaptation: {adapt}")

    loader = _loader_for(cfg.CORRUPTION.DATASET)
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    cells = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption in cfg.CORRUPTION.TYPE:
            try:
                model.reset()
            except Exception:
                pass
            x, y = loader(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption])
            x, y = x.to(device), y.to(device)
            acc = accuracy(model, x, y, cfg.TEST.BATCH_SIZE)
            err = 1.0 - acc
            cell = {
                "dataset": cfg.CORRUPTION.DATASET,
                "adaptation": adapt,
                "arch": cfg.MODEL.ARCH,
                "severity": int(severity),
                "corruption": corruption,
                "num_ex": int(cfg.CORRUPTION.NUM_EX),
                "err": float(err),
            }
            cells.append(cell)
            (out_path / f"{adapt}_{cfg.CORRUPTION.DATASET}_sev{severity}_{corruption}.json"
             ).write_text(json.dumps(cell, indent=2))
            logger.info("err [%s sev%d %s]: %.2f%%", adapt, severity, corruption, err * 100)

    return {"cells": cells, "out_dir": str(out_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="path to YAML config")
    p.add_argument("--out", default="results/tent_replication/run", help="output dir")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_from_cfg(args.cfg, args.out)


if __name__ == "__main__":
    main()
