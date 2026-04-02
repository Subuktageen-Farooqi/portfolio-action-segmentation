import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


@dataclass
class SplitInfo:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    train_actors: List[int]
    val_actors: List[int]
    test_actors: List[int]


def derive_per_frame_labels(num_frames: int, segments: List[Dict], background_class: int = 0) -> np.ndarray:
    labels = np.full((num_frames,), background_class, dtype=np.int64)
    if not segments:
        return labels
    segments = sorted(segments, key=lambda x: x["start"])
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        cls = int(seg["label"])
        start = max(0, min(start, num_frames - 1))
        end = max(start + 1, min(end, num_frames))
        labels[start:end] = cls  # latest-start rule wins naturally due to overwrite
    return labels


def _load_array(path: Path, expected_len: Optional[int] = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing modality file: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        arr = np.load(path)["arr_0"]
    else:
        raise ValueError(f"Unsupported file format for {path}")
    if expected_len is not None:
        assert len(arr) == expected_len, f"Sequence length mismatch for {path}: {len(arr)} != {expected_len}"
    return arr


class ActionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, modalities: Dict, transform=None, num_classes: int = 11):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.modalities = modalities
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        frames = _load_array(self.root / row["frames_path"])
        assert frames.ndim == 4, f"frames must be [T,C,H,W], got {frames.shape}"
        t = len(frames)

        labels = None
        if isinstance(row.get("labels_path"), str) and row["labels_path"]:
            labels = _load_array(self.root / row["labels_path"], expected_len=t).astype(np.int64)
        elif isinstance(row.get("segments_path"), str) and row["segments_path"]:
            seg_path = self.root / row["segments_path"]
            with open(seg_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            labels = derive_per_frame_labels(t, segments, background_class=0)
        else:
            raise ValueError(f"No labels_path or segments_path for sequence_id={row['sequence_id']}")

        assert len(labels) == len(frames), "Label-frame mismatch"

        sample = {
            "frames": torch.tensor(frames, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "actor_id": int(row["actor_id"]),
            "sequence_id": str(row["sequence_id"]),
        }

        if self.modalities.get("use_keypoints", False):
            kp = _load_array(self.root / row["keypoints_path"], expected_len=t)
            sample["keypoints"] = torch.tensor(kp, dtype=torch.float32)

        if self.modalities.get("use_boxes", False):
            boxes = _load_array(self.root / row["boxes_path"], expected_len=t)
            sample["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        if self.modalities.get("use_masks", False):
            masks = _load_array(self.root / row["masks_path"], expected_len=t)
            sample["masks"] = torch.tensor(masks, dtype=torch.float32)

        if self.transform is not None:
            sample["frames"] = self.transform(sample["frames"])

        return sample


def build_splits(metadata_df: pd.DataFrame, seed: int) -> SplitInfo:
    groups = metadata_df["actor_id"].values
    indices = np.arange(len(metadata_df))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(gss1.split(indices, groups=groups))

    holdout_groups = groups[holdout_idx]
    unique_holdout_actors = np.unique(holdout_groups)
    if len(unique_holdout_actors) >= 2:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        rel_val_idx, rel_test_idx = next(gss2.split(holdout_idx, groups=holdout_groups))
        val_idx = holdout_idx[rel_val_idx]
        test_idx = holdout_idx[rel_test_idx]
    else:
        val_idx = np.array([], dtype=holdout_idx.dtype)
        test_idx = holdout_idx

    train_actors = sorted(metadata_df.iloc[train_idx]["actor_id"].unique().tolist())
    val_actors = sorted(metadata_df.iloc[val_idx]["actor_id"].unique().tolist())
    test_actors = sorted(metadata_df.iloc[test_idx]["actor_id"].unique().tolist())

    assert set(train_actors).isdisjoint(val_actors)
    assert set(train_actors).isdisjoint(test_actors)
    assert set(val_actors).isdisjoint(test_actors)

    return SplitInfo(
        train_idx=train_idx.tolist(),
        val_idx=val_idx.tolist(),
        test_idx=test_idx.tolist(),
        train_actors=train_actors,
        val_actors=val_actors,
        test_actors=test_actors,
    )


def load_metadata(data_root: str, metadata_file: str) -> pd.DataFrame:
    path = Path(data_root) / metadata_file
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError("metadata file must be csv/json/jsonl")

    required = {"sequence_id", "actor_id", "frames_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    return df
