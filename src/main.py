import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import ActionDataset, build_splits, load_metadata
from engine.eval import evaluate
from engine.train import fit
from models.model import ActionSegmentationModel
from transforms import build_transforms
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.visualization import plot_metric_bar


def collate_batch(samples):
    keys = samples[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            if vals[0].ndim > 0:
                t_len = vals[0].shape[0]
                for i, v in enumerate(vals):
                    assert v.shape[0] == t_len, f"Inconsistent sequence length in batch for key '{k}' at {i}"
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def save_run_context(cfg: Dict, split_info, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": cfg["training"]["seed"],
                "train_actors": split_info.train_actors,
                "val_actors": split_info.val_actors,
                "test_actors": split_info.test_actors,
            },
            f,
            indent=2,
        )


def append_results_csv(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(cfg: Dict, mode: str, checkpoint: str | None = None):
    out_root = Path("outputs")
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "logs" / f"{cfg['experiment']['name']}.log")
    set_seed(int(cfg["training"]["seed"]))

    metadata = load_metadata(cfg["data"]["root"], cfg["data"].get("metadata_file", "metadata.csv"))
    split_info = build_splits(metadata, seed=int(cfg["training"]["seed"]))

    train_df = metadata.iloc[split_info.train_idx]
    val_df = metadata.iloc[split_info.val_idx]
    test_df = metadata.iloc[split_info.test_idx]

    save_run_context(cfg, split_info, out_root / "logs" / cfg["experiment"]["name"])
    logger.info("Split actors | train=%s val=%s test=%s", split_info.train_actors, split_info.val_actors, split_info.test_actors)

    train_set = ActionDataset(
        train_df,
        Path(cfg["data"]["root"]),
        cfg["modalities"],
        transform=build_transforms("train"),
        num_classes=int(cfg["model"]["num_classes"]),
    )
    val_set = ActionDataset(
        val_df,
        Path(cfg["data"]["root"]),
        cfg["modalities"],
        transform=build_transforms("val"),
        num_classes=int(cfg["model"]["num_classes"]),
    )
    test_set = ActionDataset(
        test_df,
        Path(cfg["data"]["root"]),
        cfg["modalities"],
        transform=build_transforms("test"),
        num_classes=int(cfg["model"]["num_classes"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionSegmentationModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    best_ckpt = Path(checkpoint) if checkpoint else out_root / "checkpoints" / "best.pt"
    if mode == "train":
        best_ckpt, _ = fit(model, train_loader, val_loader, optimizer, cfg, out_root, device)

    load_checkpoint(best_ckpt, model, optimizer=None, map_location=device)
    metrics = evaluate(model, test_loader, device, num_classes=int(cfg["model"]["num_classes"]))
    logger.info("Test metrics: %s", metrics)

    plot_metric_bar(metrics, out_root / "plots" / f"{cfg['experiment']['name']}_metrics.png", title=cfg["experiment"]["name"])
    append_results_csv(
        out_root / "results.csv",
        {
            "experiment": cfg["experiment"]["name"],
            "config": cfg.get("_config_path", ""),
            "seed": cfg["training"]["seed"],
            "split_strategy": cfg["data"]["split_strategy"],
            **metrics,
            "checkpoint": str(best_ckpt),
        },
    )


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = path
    return cfg


def run_all_configs(configs_dir: str):
    files = sorted(Path(configs_dir).glob("exp_*.yaml"))
    for file in files:
        cfg = load_config(str(file))
        run_experiment(cfg, mode="train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp_rgb.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_all", type=str, default=None)
    args = parser.parse_args()

    if args.run_all:
        run_all_configs(args.run_all)
        return

    cfg = load_config(args.config)
    run_experiment(cfg, mode=args.mode, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
