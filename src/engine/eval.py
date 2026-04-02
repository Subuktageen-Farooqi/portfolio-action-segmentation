from typing import Dict

import numpy as np
import torch

from engine.metrics import compute_metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch)  # [B,T,C]
        labels = batch["labels"]  # [B,T]
        valid_mask = batch.get("valid_mask")

        if valid_mask is not None:
            flat_mask = valid_mask.reshape(-1)
            flat_logits = logits.reshape(-1, logits.size(-1))[flat_mask]
            flat_labels = labels.reshape(-1)[flat_mask]
        else:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)

        all_logits.append(flat_logits.cpu().numpy())
        all_labels.append(flat_labels.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return compute_metrics(logits_np, labels_np, num_classes=num_classes)
