#!/usr/bin/env python3
"""Cut the raw recordings into normalized windows, one file per split.

The windows are the input of the ARMA system identification step that
produces the FRF files.

Example:
    python scripts/prepare_windows.py --data-dir data/raw --out-dir data/windows --format mat
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

from shm_edge.data import SPLITS, load_split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="folder with exp_<n>.feather or exp_<n>.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--length", type=int, default=2000)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--axis", default="x")
    p.add_argument("--no-normalize", action="store_true", help="skip per-window detrending and z-scoring")
    p.add_argument("--format", choices=["npy", "mat"], default="npy")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        w = load_split(args.data_dir, split, args.length, args.stride, args.axis, not args.no_normalize)
        out = args.out_dir / f"{split}.{args.format}"
        if args.format == "npy":
            np.save(out, w.astype(np.float32))
        else:
            savemat(out, {"windows": w.T})   # (length, n_windows), one column per window
        print(f"{split:<10}: {w.shape[0]:6d} windows -> {out}")


if __name__ == "__main__":
    main()
