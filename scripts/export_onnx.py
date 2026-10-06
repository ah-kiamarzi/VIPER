#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX (FP32) for GAP9 deployment with NNTool.

Example:
    python scripts/export_onnx.py --checkpoint runs/fc_ae/model.pt --output fc_ae.onnx
"""

import argparse
from pathlib import Path

import torch

from shm_edge.checkpoint import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    dummy = torch.randn(1, ckpt["band"].input_dim)
    torch.onnx.export(model, dummy, args.output, opset_version=args.opset,
                      input_names=["frf"], output_names=["reconstruction"])
    print(f"exported {ckpt['model_name']} (input {tuple(dummy.shape)}) to {args.output}")


if __name__ == "__main__":
    main()
