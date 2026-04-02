from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def compute_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    # logits: [N,C], labels: [N]
    pred = logits.argmax(axis=1)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average="macro", zero_division=0)

    labels_onehot = np.eye(num_classes)[labels]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    try:
        m_ap = average_precision_score(labels_onehot, probs, average="macro")
    except ValueError:
        m_ap = 0.0

    return {
        "accuracy": float(acc),
        "F1": float(f1),
        "mAP": float(m_ap),
    }
