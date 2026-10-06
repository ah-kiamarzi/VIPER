"""Reconstruction training loop for the autoencoders."""

import copy
import time

import torch
import torch.nn.functional as F


def _run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x), y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def fit(model, train_loader, valid_loader, epochs: int, lr: float, betas=(0.9, 0.999),
        device: str = "cpu", log=print):
    """Train ``model`` with Adam and an MSE reconstruction loss.

    Returns a dict with the per-epoch losses, the best validation loss, the
    weights at that epoch (``best_state``) and the training time. The model
    itself is left at its last-epoch weights.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-8)
    history = {"train_loss": [], "valid_loss": [], "best_valid_loss": float("inf"), "best_epoch": 0}
    best_state = None
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        valid_loss = _run_epoch(model, valid_loader, device)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        if valid_loss < history["best_valid_loss"]:
            history["best_valid_loss"], history["best_epoch"] = valid_loss, epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        log(f"epoch {epoch:3d}/{epochs}  train {train_loss:.6f}  valid {valid_loss:.6f}")
    history["train_time_s"] = time.time() - start
    history["best_state"] = best_state
    return history
