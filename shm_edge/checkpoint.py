"""Saving and restoring a trained model together with its input band and threshold."""

import torch

from .data import band_from_dict, band_to_dict
from .models import build_model


def save_checkpoint(path, model_name: str, model, band, threshold: float, quantile: float, **extra):
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "band": band_to_dict(band),
        "threshold": threshold,
        "quantile": quantile,
        **extra,
    }, path)


def load_checkpoint(path, device: str = "cpu"):
    """Return ``(model, checkpoint_dict)``; ``checkpoint_dict["band"]`` is a ``Band``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt["band"] = band_from_dict(ckpt["band"])
    model = build_model(ckpt["model_name"], ckpt["band"].input_dim)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
