"""Train pycls R-26 (ResNet-26, 6n+2 with n=4) on CIFAR-10 / CIFAR-100.

This trains the source model used in the TENT paper's Table 2 (Wang et al.,
ICLR 2021). The paper's Models paragraph specifies "residual networks (He et
al., 2016) with 26 layers (R-26) on CIFAR-10/100", trained with the pycls
library (Radosavovic et al., 2019). We reuse pycls's `ResNet` model
definition verbatim and a canonical training recipe (SGD m=0.9 wd=5e-4
nesterov, lr=0.1 cosine, 200 epochs, BS=128 effective, std augs).

Multi-GPU is handled by torch's DDP via torchrun:
    torchrun --standalone --nproc-per-node=4 train_R26.py \\
        --dataset cifar100 --out ckpt/local/R26_cifar100.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pycls.core.config import cfg as pycfg
from pycls.models.resnet import ResNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_R26(num_classes: int) -> nn.Module:
    pycfg.MODEL.TYPE = "resnet"
    pycfg.MODEL.DEPTH = 26
    pycfg.MODEL.NUM_CLASSES = num_classes
    pycfg.RESNET.TRANS_FUN = "basic_transform"
    pycfg.TRAIN.DATASET = "cifar10"
    pycfg.TEST.DATASET = "cifar10"
    pycfg.TRAIN.IM_SIZE = 32
    pycfg.TEST.IM_SIZE = 32
    return ResNet()


def build_loaders(dataset: str, data_root: str, per_gpu_batch: int,
                  world_size: int, rank: int, num_workers: int):
    if dataset == "cifar10":
        mean, std, ds_cls, num_classes = CIFAR10_MEAN, CIFAR10_STD, torchvision.datasets.CIFAR10, 10
    elif dataset == "cifar100":
        mean, std, ds_cls, num_classes = CIFAR100_MEAN, CIFAR100_STD, torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(dataset)

    tr_tx = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    ev_tx = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    tr_ds = ds_cls(root=data_root, train=True, download=(rank == 0), transform=tr_tx)
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        tr_ds = ds_cls(root=data_root, train=True, download=False, transform=tr_tx)
    ev_ds = ds_cls(root=data_root, train=False, download=False, transform=ev_tx)

    tr_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) \
        if world_size > 1 else None
    tr_loader = DataLoader(
        tr_ds, batch_size=per_gpu_batch, sampler=tr_sampler,
        shuffle=(tr_sampler is None), num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    ev_loader = DataLoader(
        ev_ds, batch_size=200, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return tr_loader, ev_loader, tr_sampler, num_classes


@torch.no_grad()
def measure_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def refresh_bn_stats(model: nn.Module, loader: DataLoader, device: torch.device,
                     num_samples: int) -> None:
    """Reset running BN stats then accumulate over `num_samples` train images.

    Mirrors pycls's BN.USE_PRECISE_STATS=True (NUM_SAMPLES_PRECISE) behaviour.
    """
    inner = model.module if isinstance(model, DDP) else model
    for m in inner.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over passed batches
    inner.train()
    seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = inner(x)
        seen += x.size(0)
        if seen >= num_samples:
            break
    inner.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128, help="GLOBAL batch size (split across GPUs)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precise-bn-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    per_gpu = max(1, args.batch_size // max(1, world_size))
    tr_loader, ev_loader, tr_sampler, num_classes = build_loaders(
        args.dataset, args.data_root, per_gpu, world_size, rank, args.num_workers)

    model = build_R26(num_classes).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        print(f"[R-26 train] dataset={args.dataset} world_size={world_size} "
              f"per_gpu_bs={per_gpu} effective_bs={per_gpu*max(1,world_size)} "
              f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay}")
        n = sum(p.numel() for p in model.parameters())
        print(f"[R-26 train] params={n/1e6:.3f}M num_classes={num_classes}")

    for epoch in range(1, args.epochs + 1):
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        scheduler.step()
        train_loss = running / max(n, 1)
        if rank == 0 and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
            acc = measure_acc(model, ev_loader, device)
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"test_acc={acc*100:.2f}%  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    if rank == 0:
        print("[R-26 train] refreshing BN running stats over "
              f"{args.precise_bn_samples} samples (pycls precise BN)")
    refresh_bn_stats(model, tr_loader, device, args.precise_bn_samples)

    # Eval + save must run on the unwrapped module on rank 0 only. Calling
    # measure_acc on the DDP-wrapped model triggers DDP's buffer broadcast
    # (a collective op), which deadlocks when the other ranks have already
    # fallen through to dist.barrier().
    inner = model.module if isinstance(model, DDP) else model
    if rank == 0:
        final_acc = measure_acc(inner, ev_loader, device)
        print(f"[R-26 train] FINAL test_acc={final_acc*100:.2f}%")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": inner.state_dict(),
            "dataset": args.dataset,
            "num_classes": num_classes,
            "depth": 26,
            "arch": "pycls_resnet_R26_basic_transform",
            "test_acc": final_acc,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }, out_path)
        print(f"[R-26 train] saved {out_path} (test_acc={final_acc*100:.2f}%)")

    # Skip dist.barrier() at the end: rank 0 does measure_acc/save which can
    # take a few extra seconds, and ranks 1-3 reach the barrier first; the
    # NCCL watchdog can be confused into a 10-min hang waiting for rank 0 to
    # rejoin. Each rank has nothing left to do after this point, so a plain
    # destroy_process_group is enough to clean up the NCCL communicator.
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
