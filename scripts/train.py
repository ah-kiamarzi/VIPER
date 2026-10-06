#!/usr/bin/env python3
"""Train an autoencoder on healthy FRF data and evaluate it.

Example:
    python scripts/train.py --model fc_ae --frf-dir data/FRF
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from shm_edge.checkpoint import save_checkpoint
from shm_edge.data import load_frf_splits, make_loader
from shm_edge.evaluate import evaluate_model, format_report
from shm_edge.models import MODELS
from shm_edge.train import fit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--frf-dir", type=Path, required=True, help="folder with the FRF .mat files")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--band", choices=["peaks", "full"], default="peaks",
                   help="input bins: band around the first peaks, or all bins (default: peaks)")
    p.add_argument("--n-peaks", type=int, default=2)
    p.add_argument("--pad-bins", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, help="default: per-model value in shm_edge.models.MODELS")
    p.add_argument("--quantile", type=float, help="threshold percentile in [0, 1]; default: per-model value")
    p.add_argument("--restore-best", action=argparse.BooleanOptionalAction,
                   help="evaluate best-validation weights instead of the last epoch; default: per-model value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    spec = MODELS[args.model]
    lr = args.lr or spec.lr
    quantile = args.quantile if args.quantile is not None else spec.quantile
    restore_best = spec.restore_best if args.restore_best is None else args.restore_best

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits = load_frf_splits(args.frf_dir, spec.input_size, band=args.band,
                             n_peaks=args.n_peaks, pad_bins=args.pad_bins)
    print(splits.summary())

    model = spec.build(splits.band.input_dim)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters")

    run_dir = args.out_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    history = fit(model,
                  make_loader(splits.train, args.batch_size, shuffle=True),
                  make_loader(splits.valid, args.batch_size),
                  epochs=args.epochs, lr=lr, betas=spec.betas, device=args.device)
    if restore_best:
        model.load_state_dict(history["best_state"])

    # Att-CNN-AE attends across the batch, so it is evaluated with the training batch size.
    eval_bs = args.batch_size if args.model == "att_cnn_ae" else 256
    result = evaluate_model(model, splits, quantile, batch_size=eval_bs, device=args.device)
    print(format_report(result))

    model.cpu()
    save_checkpoint(run_dir / "model.pt", args.model, model, splits.band, result["threshold"], quantile,
                    eval_batch_size=eval_bs)
    history.pop("best_state")
    with open(run_dir / "results.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "lr": lr,
                   "history": history, **result}, f, indent=2)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
