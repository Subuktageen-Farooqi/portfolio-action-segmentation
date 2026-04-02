from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine.eval import evaluate
from utils.checkpoint import save_checkpoint


def train_one_epoch(model, loader, optimizer, device, num_classes: int, label_smoothing: float = 0.0):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        logits = model(batch)
        labels = batch["labels"]
        assert logits.shape[:2] == labels.shape[:2], f"logits/labels temporal mismatch {logits.shape} vs {labels.shape}"

        loss = F.cross_entropy(
            logits.reshape(-1, num_classes),
            labels.reshape(-1),
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    cfg: Dict,
    out_dir: Path,
    device: torch.device,
):
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("patience", 7))
    num_classes = int(cfg["model"]["num_classes"])
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))

    history = {"train_loss": [], "val_mAP": [], "val_F1": [], "val_accuracy": []}
    best_score = -1.0
    bad_epochs = 0
    best_path = out_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, num_classes, label_smoothing)
        val_metrics = evaluate(model, val_loader, device, num_classes)

        history["train_loss"].append(train_loss)
        history["val_mAP"].append(val_metrics["mAP"])
        history["val_F1"].append(val_metrics["F1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        score = val_metrics["mAP"]
        if score > best_score:
            best_score = score
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, cfg, val_metrics)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    plot_training_history(history, out_dir / "plots" / f"{cfg['experiment']['name']}_history.png")
    return best_path, history


def plot_training_history(history: Dict, plot_path: Path):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_mAP"], label="val_mAP")
    plt.plot(history["val_F1"], label="val_F1")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
