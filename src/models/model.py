from typing import Dict, List

import torch
import torch.nn as nn

from models.backbone import FrameBackbone
from models.fusion import build_fusion
from models.temporal import build_temporal


class LightweightStructEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,*]
        if x.ndim > 3:
            x = x.flatten(start_dim=2)
        return self.net(x)


class ActionSegmentationModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        hidden_dim = int(cfg["model"].get("hidden_dim", 256))
        self.backbone = FrameBackbone(cfg["model"]["backbone"], out_dim=hidden_dim)

        mods = cfg["modalities"]
        self.encoders = nn.ModuleDict()
        self.active_mods: List[str] = []

        if mods.get("use_keypoints", False):
            self.encoders["keypoints"] = LightweightStructEncoder(in_dim=34, out_dim=hidden_dim)
            self.active_mods.append("keypoints")
        if mods.get("use_boxes", False):
            self.encoders["boxes"] = LightweightStructEncoder(in_dim=4, out_dim=hidden_dim)
            self.active_mods.append("boxes")
        if mods.get("use_masks", False):
            self.encoders["masks"] = LightweightStructEncoder(in_dim=128, out_dim=hidden_dim)
            self.active_mods.append("masks")

        num_streams = 1 + len(self.active_mods)
        fusion_kind = cfg["model"].get("fusion", "none")
        if not self.active_mods:
            fusion_kind = "none"
        self.fusion = build_fusion(fusion_kind, dim=hidden_dim, num_streams=num_streams)
        self.temporal = build_temporal(cfg["model"]["temporal"], hidden_dim)
        self.classifier = nn.Linear(hidden_dim, cfg["model"]["num_classes"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"]
        assert frames.ndim == 5, f"frames expected [B,T,C,H,W], got {frames.shape}"

        rgb_feat = self.backbone(frames)

        extras = []
        for mod in self.active_mods:
            x = batch.get(mod)
            if x is None:
                raise ValueError(f"Missing required modality '{mod}' in batch")
            extras.append(self.encoders[mod](x))

        fused = self.fusion(rgb_feat, extras)
        temporal_out = self.temporal(fused)
        logits = self.classifier(temporal_out)
        assert logits.ndim == 3, f"logits expected [B,T,C], got {logits.shape}"
        return logits
