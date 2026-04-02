from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: Dict[str, Any], metrics: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
