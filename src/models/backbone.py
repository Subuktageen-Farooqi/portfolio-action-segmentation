import torch
import torch.nn as nn
import torchvision.models as tvm


class FrameBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", out_dim: int = 256):
        super().__init__()
        if name != "resnet18":
            raise ValueError("Only resnet18 is implemented currently")
        net = tvm.resnet18(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W]
        b, t, c, h, w = x.shape
        feats = self.net(x.view(b * t, c, h, w))
        feats = self.proj(feats)
        return feats.view(b, t, -1)
