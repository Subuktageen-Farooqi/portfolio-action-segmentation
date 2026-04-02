import torch
import torch.nn as nn


class TemporalIdentity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TemporalTCN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,T,D]
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return y


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, nhead: int = 8, nlayers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_temporal(kind: str, dim: int) -> nn.Module:
    if kind == "none":
        return TemporalIdentity()
    if kind == "tcn":
        return TemporalTCN(dim)
    if kind == "transformer":
        return TemporalTransformer(dim)
    raise ValueError(f"Unsupported temporal module: {kind}")
