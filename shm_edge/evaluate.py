"""Reconstruction-error anomaly scoring and detection metrics.

A window is flagged as anomalous when its reconstruction MSE exceeds a
threshold set at a percentile of the errors on the *healthy validation* set.
No anomalous data is used to choose the threshold.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


@torch.no_grad()
def reconstruction_errors(model, x: torch.Tensor, batch_size: int = 256, device: str = "cpu") -> torch.Tensor:
    """Per-window mean squared reconstruction error, shape ``(n_windows,)``."""
    model.eval().to(device)
    errors = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        recon = model(batch).float()
        errors.append(F.mse_loss(recon, batch, reduction="none").mean(dim=-1).cpu())
    return torch.cat(errors)


def threshold_from_errors(valid_errors: torch.Tensor, quantile: float) -> float:
    return torch.quantile(valid_errors, quantile).item()


def detection_metrics(err_normal: torch.Tensor, err_anomaly: torch.Tensor, threshold: float) -> dict:
    """Binary metrics with healthy = 0 and anomalous = 1."""
    scores = torch.cat([err_normal, err_anomaly]).numpy()
    y_true = np.r_[np.zeros(len(err_normal)), np.ones(len(err_anomaly))]
    y_pred = (scores > threshold).astype(int)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mdr": 1 - recall,                                      # missed-detection rate
        "far": float((err_normal > threshold).float().mean()),  # false-alarm rate
        "pr_auc": average_precision_score(y_true, scores),
    }


def evaluate_model(model, splits, quantile: float, batch_size: int = 256, device: str = "cpu",
                   threshold: float = None) -> dict:
    """Threshold on healthy validation errors, then score the healthy test set and all anomaly levels.

    Pass ``threshold`` to reuse a fixed value (e.g. the FP32 threshold when
    evaluating a quantized model).
    """
    err = lambda x: reconstruction_errors(model, x, batch_size, device)
    if threshold is None:
        threshold = threshold_from_errors(err(splits.valid), quantile)
    err_test = err(splits.test)
    err_levels = {name: err(x) for name, x in splits.anomalies.items()}
    err_anomaly = torch.cat(list(err_levels.values()))
    return {
        "threshold": threshold,
        "quantile": quantile,
        "metrics": detection_metrics(err_test, err_anomaly, threshold),
        "detection_rate_per_level": {k: float((e > threshold).float().mean()) for k, e in err_levels.items()},
    }


def format_report(result: dict) -> str:
    m = result["metrics"]
    rows = [
        f"threshold   : {result['threshold']:.6f} ({result['quantile'] * 100:.0f}th percentile of healthy validation)",
        f"accuracy    : {m['accuracy'] * 100:.2f} %",
        f"ROC-AUC     : {m['roc_auc'] * 100:.2f}",
        f"precision   : {m['precision'] * 100:.2f} %",
        f"recall      : {m['recall'] * 100:.2f} %",
        f"F1          : {m['f1'] * 100:.2f} %",
        f"MDR         : {m['mdr']:.4f}",
        f"FAR         : {m['far']:.4f}",
        f"PR-AUC      : {m['pr_auc'] * 100:.2f}",
        "detection rate per anomaly level:",
    ]
    rows += [f"  {k:<10}: {v * 100:6.2f} %" for k, v in result["detection_rate_per_level"].items()]
    return "\n".join(rows)
