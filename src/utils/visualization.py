from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_bar(metrics: dict, out_path: Path, title: str = "Evaluation Metrics"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]

    plt.figure(figsize=(6, 4))
    plt.bar(keys, vals)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
