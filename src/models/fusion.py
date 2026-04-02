import torch
import torch.nn as nn


class NoneFusion(nn.Module):
    def forward(self, rgb: torch.Tensor, extras: list[torch.Tensor]) -> torch.Tensor:
        return rgb


class ConcatFusion(nn.Module):
    def __init__(self, input_dim: int, num_streams: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim * num_streams, out_dim)

    def forward(self, rgb: torch.Tensor, extras: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([rgb] + extras, dim=-1)
        return self.fc(x)


class GatedFusion(nn.Module):
    def __init__(self, dim: int, num_streams: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_streams, num_streams),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, extras: list[torch.Tensor]) -> torch.Tensor:
        streams = [rgb] + extras
        x = torch.cat(streams, dim=-1)
        gates = self.gate(x)
        out = 0.0
        for i, s in enumerate(streams):
            out = out + gates[..., i : i + 1] * s
        return out


def build_fusion(kind: str, dim: int, num_streams: int):
    if kind in {"none", "late"}:
        return NoneFusion()
    if kind == "concat":
        return ConcatFusion(input_dim=dim, num_streams=num_streams, out_dim=dim)
    if kind == "gated":
        return GatedFusion(dim=dim, num_streams=num_streams)
    raise ValueError(f"Unsupported fusion: {kind}")
