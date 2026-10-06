# VIPER

**VI**bration monitoring on **P**arallel **E**nergy-efficient **R**ISC-V microcontrollers.

VIPER is the code for the unsupervised part of the paper *"Energy-Efficient Deep Learning–Based Vibration Monitoring on Parallel Ultra-Low-Power RISC-V Microcontrollers"*.

The repository provides:

- **Models**: the fully connected autoencoder (FC-AE) and the 1D convolutional autoencoder (CNN-AE) proposed in the paper, plus a re-implementation of the attention-convolutional autoencoder (Att-CNN-AE, [Teng et al., 2024](https://doi.org/10.1016/j.engstruct.2023.117278)) used as the baseline.
- **Data loaders** for the System Identification (SysId) Frequency Response Function (FRF) features and for the raw accelerometer recordings of the steel-frame dataset.
- **Scripts** to train, evaluate (FP32 / FP16 / INT8), and export the models to ONNX for deployment on GAP9 with NNTool.

The dataset is **not** included in this repository. See [Dataset](#dataset).

## Pipeline

```
raw acceleration ──► windows ──► ARMA(20,20) SysId ──► |FRF| (512 bins, dB) ──► band selection ──► autoencoder ──► MSE ──► threshold
   (x axis)        (2000 samples)                                                                                     healthy / anomalous
```

1. The longitudinal (`x`) acceleration is cut into overlapping windows.
2. An ARMA(20,20) model is identified for each window, and the magnitude of its FRF is evaluated on 512 frequency bins. This step produces the `.mat` FRF files used for training. The SysId step itself is not part of this repository; an implementation parallelized for GAP9 is described in [Kiamarzi et al., 2025](https://arxiv.org/abs/2504.04884).
3. The FRF is converted to dB. By default the input is the frequency band around the first two structural modes. The band is located on the mean healthy training FRF and zero-padded to a length the model accepts. Use `--band full` to feed all 512 bins instead.
4. The autoencoder is trained **only on healthy data** to reconstruct its input.
5. A window is flagged as anomalous when its reconstruction MSE exceeds a percentile of the errors on the healthy validation set. No anomalous data is used to set the threshold.

## Repository layout

```
shm_edge/
  models/
    fc_ae.py        FC-AE: Linear-Sigmoid-BatchNorm stack, halving down to a 32-d latent
    cnn_ae.py       CNN-AE: 4 residual ConvBlocks / 4 TransConvBlocks
    att_cnn_ae.py   Att-CNN-AE baseline
    __init__.py     model registry (MODELS) with per-model defaults
  data/
    frf.py          FRF .mat loading, band selection, DataLoader helper
    raw.py          raw recordings: split definition, windowing, normalization
  train.py          training loop (Adam + MSE)
  evaluate.py       reconstruction errors, threshold, detection metrics
  quantize.py       FP16 and INT8 post-training quantization (PyTorch, CPU)
  checkpoint.py     save/load model together with its input band and threshold
scripts/
  train.py          train + evaluate one model
  evaluate.py       evaluate a checkpoint in FP32 / FP16 / INT8
  export_onnx.py    export a checkpoint to ONNX
  prepare_windows.py  window the raw recordings (input of the SysId step)
```

## Installation

Requires Python ≥ 3.9.

```bash
git clone <repository-url>
cd VIPER
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add [onnx] to also install the ONNX exporter dependency
```

## Dataset

The dataset was recorded on a four-storey steel frame with a MEMS accelerometer (IIS2ICLX) on the top floor, under ambient excitation only. It contains 8 hours of healthy data and nine anomaly levels, which were produced by adding increasing masses to the top floor. Download it from Zenodo: <https://zenodo.org/records/17988554>.

| Experiment | Condition | Added mass [kg] | Split |
|---|---|---|---|
| `exp_1` | healthy | 0 | test |
| `exp_2` – `exp_4` | healthy | 0 | train |
| `exp_5` | healthy | 0 | validation |
| `exp_6` – `exp_14` | anomaly level 1 – 9 | 1.5, 2.8, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6 | test |

Each experiment is a table with columns `ts, x, y` (`exp_<n>.feather` or `exp_<n>.csv`).

Keep the data outside version control. `data/` is in `.gitignore`, and a symlink works well:

```bash
ln -s /path/to/dataset data
```

Expected layout:

```
data/
  raw/                                   exp_1.feather … exp_14.feather (or .csv)
  FRF/
    FRF_512_trainARMA_AR20_MA20.mat      healthy, training
    FRF_512_valARMA_AR20_MA20.mat        healthy, validation (threshold)
    FRF_512_testARMA_AR20_MA20.mat       healthy, test
    FRF_512_anARMA_AR20_MA20_<k>.mat     anomaly level k = 1 … 9
```

Each FRF file stores a complex array under the key `FRF` with shape `(512, n_windows)`, with one column per window. Other file names can be passed through the `files` argument of `load_frf_splits`.

## Usage

### Python API: models and data loaders

```python
from shm_edge.data import load_frf_splits, make_loader
from shm_edge.models import MODELS
from shm_edge.train import fit
from shm_edge.evaluate import evaluate_model, format_report

spec = MODELS["fc_ae"]                          # "fc_ae", "cnn_ae" or "att_cnn_ae"

# Load all splits and cut them to the band around the first two modes.
splits = load_frf_splits("data/FRF", spec.input_size, band="peaks")
print(splits.summary())                          # band and number of windows per split

model = spec.build(splits.band.input_dim)        # e.g. FCAutoEncoder(input_dim=128, latent_dim=32)

history = fit(model,
              make_loader(splits.train, batch_size=16, shuffle=True),
              make_loader(splits.valid, batch_size=16),
              epochs=100, lr=spec.lr, betas=spec.betas)

result = evaluate_model(model, splits, quantile=spec.quantile)
print(format_report(result))
```

The models can also be used on their own. All of them take and return tensors of shape `(batch, length)`:

```python
import torch
from shm_edge.models import FCAutoEncoder, CNNAutoEncoder, AttCNNAutoEncoder

x = torch.randn(8, 512)
FCAutoEncoder(input_dim=512, latent_dim=32)(x).shape   # (8, 512); input_dim = 32 * 2^k
CNNAutoEncoder()(x).shape                              # (8, 512); length must be a multiple of 16
AttCNNAutoEncoder()(x).shape                           # (8, 512)
```

To work from the raw recordings instead:

```python
from shm_edge.data import load_split

windows = load_split("data/raw", "train", length=2000, stride=200)   # (n_windows, 2000), detrended + z-scored
```

### Command line

Train and evaluate a model. The checkpoint, metrics and loss history are written to `runs/<model>/`:

```bash
python scripts/train.py --model fc_ae      --frf-dir data/FRF
python scripts/train.py --model cnn_ae     --frf-dir data/FRF
python scripts/train.py --model att_cnn_ae --frf-dir data/FRF
```

Useful options: `--band {peaks,full}`, `--epochs`, `--batch-size`, `--lr`, `--quantile`, `--seed`, `--device`.

Evaluate a checkpoint at reduced precision. The FP32 threshold is reused for the FP16 and INT8 models:

```bash
python scripts/evaluate.py --checkpoint runs/fc_ae/model.pt --frf-dir data/FRF --precision fp32 fp16 int8
```

Export to ONNX for GAP9:

```bash
python scripts/export_onnx.py --checkpoint runs/fc_ae/model.pt --output fc_ae.onnx
```

Window the raw recordings, e.g. as input to your own SysId implementation:

```bash
python scripts/prepare_windows.py --data-dir data/raw --out-dir data/windows --format mat
```

## GAP9 deployment

1. Export the trained model with `scripts/export_onnx.py`.
2. Import the ONNX graph with **NNTool** from the GAP SDK (GreenWaves Technologies).
3. Quantize it to INT8 for the NE16 accelerator, or to FP16 for the 8-core RISC-V cluster.
4. Generate the C code with **AutoTiler**.

`scripts/evaluate.py --precision fp16 int8` estimates the accuracy impact of quantization in PyTorch before deployment. The GAP9 application code is not part of this repository.

## Citation

If you use this code or the dataset, please cite:

```bibtex
@article{shm_edge_gap9,
  title   = {Energy-Efficient Deep Learning--Based Vibration Monitoring on Parallel Ultra-Low-Power RISC-V Microcontrollers},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## License

This project is released under the GNU General Public License v3.0. See [LICENSE](LICENSE).

