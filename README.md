# Portfolio Action Segmentation

A leakage-safe, actor-split action segmentation training/evaluation scaffold with configurable modality fusion.

## Setup

```bash
pip install -r requirements.txt
```

## Data expectations

Put your dataset under `data/` and provide a metadata table (CSV/JSON) containing:

- `sequence_id`
- `actor_id`
- `frames_path` (required)
- `labels_path` (optional, per-frame labels)
- `segments_path` (optional, timestamp segments used to derive per-frame labels)
- modality paths when enabled: `keypoints_path`, `boxes_path`, `masks_path`

Frames should be stored as numpy arrays `[T, C, H, W]` (`.npy` / `.npz`).

## Split policy (strict)

- Actor-based grouped split (`group_actor`) only.
- No frame-level split.
- One fixed split reused across all experiments (E1–E6).
- Split actor IDs + seed are logged in `outputs/logs/<exp>/split_info.json`.

## Commands

```bash
# train
python src/main.py --config configs/exp_rgb.yaml --mode train

# evaluate
python src/main.py --config configs/exp_rgb.yaml --mode eval --checkpoint outputs/checkpoints/best.pt

# run experiment suite
python src/main.py --run_all configs/
```

## Experiment plan

- E1: RGB baseline (`configs/exp_rgb.yaml`)
- E2: RGB + keypoints (`configs/exp_rgb_keypoints.yaml`)
- E3: RGB + boxes (`configs/exp_rgb_boxes.yaml`)
- E4: RGB + masks (`configs/exp_rgb_masks.yaml`)
- E5: best fusion strategy (derive from E2-E4 comparison)
- E6: best + tuning (`configs/exp_best.yaml`)

Results are appended to `outputs/results.csv`.

## Guardrails implemented

- Hard assertions for shape/misalignment.
- Hard fail on missing enabled modality files.
- Label/frame mismatch check with fallback from timestamp segments.
- Gap policy: assign background class.
- Overlap policy: latest-start segment wins.

## Outputs

- Logs: `outputs/logs/`
- Checkpoints: `outputs/checkpoints/` (best checkpoint at `best.pt`)
- Plots: `outputs/plots/`
- Table: `outputs/results.csv`

## Limitations

- Assumes pre-extracted frame/modality arrays.
- `mAP` is macro AP over per-frame multiclass one-vs-rest.
- Uses simple lightweight structured-modality encoders by design to reduce param-count confounding.
