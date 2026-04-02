from typing import Optional

import torch


class NormalizeFrames:
    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean or [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor(std or [0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        assert frames.ndim == 4, f"Expected [T,C,H,W], got {frames.shape}"
        return (frames - self.mean.to(frames.device)) / self.std.to(frames.device)


def build_transforms(mode: str) -> Optional[NormalizeFrames]:
    if mode in {"train", "val", "test", "eval"}:
        return NormalizeFrames()
    return None
