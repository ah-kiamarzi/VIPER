#!/usr/bin/env python3
"""Evaluate a trained checkpoint in FP32, FP16 and/or INT8.

The threshold is computed once on the FP32 healthy validation errors and
reused for the reduced-precision models.

Example:
    python scripts/evaluate.py --checkpoint runs/fc_ae/model.pt --frf-dir data/FRF --precision fp32 fp16 int8
"""

import argparse
from pathlib import Path

from shm_edge.checkpoint import load_checkpoint
from shm_edge.data import load_frf_splits
from shm_edge.evaluate import evaluate_model, format_report
from shm_edge.quantize import FP16Model, to_int8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frf-dir", type=Path, required=True)
    p.add_argument("--precision", nargs="+", choices=["fp32", "fp16", "int8"], default=["fp32"])
    p.add_argument("--quantile", type=float, help="override the threshold percentile stored in the checkpoint")
    p.add_argument("--n-calib", type=int, default=256, help="training windows used for INT8 calibration")
    return p.parse_args()


def main():
    args = parse_args()
    model, ckpt = load_checkpoint(args.checkpoint)
    splits = load_frf_splits(args.frf_dir, input_size=None, band=ckpt["band"])
    quantile = args.quantile if args.quantile is not None else ckpt["quantile"]
    bs = ckpt.get("eval_batch_size", 256)

    fp32 = evaluate_model(model, splits, quantile, batch_size=bs)
    variants = {
        "fp32": lambda: model,
        "fp16": lambda: FP16Model(model),
        "int8": lambda: to_int8(model, splits.train[:args.n_calib]),
    }
    for name in args.precision:
        result = fp32 if name == "fp32" else evaluate_model(variants[name](), splits, quantile, batch_size=bs,
                                                            threshold=fp32["threshold"])
        print(f"\n== {ckpt['model_name']} [{name}] ==")
        print(format_report(result))


if __name__ == "__main__":
    main()
