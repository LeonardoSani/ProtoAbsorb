"""Fine-tune ViT-S/16 on CIFAR-10 at 224x224."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.data import cifar10_test_224, cifar10_train_224, make_dataloader
from ..models.models import build_vit_small
from ..utils.utils import ensure_dir, get_device, get_logger, set_seed


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train_vit_small(
    data_root: str,
    out_path: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    log = get_logger("train_vit")
    set_seed(seed)
    dev = torch.device(device) if device else get_device()
    log.info(f"Device: {dev}")

    train_set = cifar10_train_224(data_root, augment=True)
    test_set = cifar10_test_224(data_root)
    train_loader = make_dataloader(train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    test_loader = make_dataloader(test_set, batch_size=256, shuffle=False,
                                  num_workers=num_workers)

    model = build_vit_small(num_classes=10).to(dev)
    log.info(f"ViT-S/16 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # separate LRs: backbone lower, head higher
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history: list[dict] = []
    ensure_dir(Path(out_path).parent)

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = running / max(n, 1)
        scheduler.step()

        test_acc = evaluate(model, test_loader, dev)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "test_acc": test_acc, "lr": scheduler.get_last_lr()[0]})
        log.info(f"epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                 f"test_acc={test_acc*100:.2f}%  ({elapsed:.1f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "test_acc": test_acc}, out_path)

    log.info(f"Best test accuracy: {best_acc*100:.2f}% (saved to {out_path})")
    return {"best_acc": best_acc, "history": history}


def cli() -> None:
    p = argparse.ArgumentParser(description="Fine-tune ViT-S/16 on CIFAR-10 at 224x224")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints/vit_small_cifar10.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train_vit_small(
        data_root=args.data_root, out_path=args.out, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    cli()
