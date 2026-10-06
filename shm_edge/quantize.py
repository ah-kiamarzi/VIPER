"""Post-training quantization used to check accuracy before GAP9 deployment.

These PyTorch quantized models only estimate the accuracy impact of
reduced precision. On GAP9 the ONNX model (``scripts/export_onnx.py``) is
quantized and compiled by NNTool.
"""

import copy
import sys

import torch
from torch import nn


def _replicate_to_zero_padding(model: nn.Module) -> None:
    # Replication padding has no FP16 or quantized CPU kernel.
    for m in model.modules():
        if isinstance(m, nn.Conv1d) and m.padding_mode == "replicate":
            m.padding_mode = "zeros"


class FP16Model(nn.Module):
    """Runs a copy of ``model`` in float16; takes and returns float32."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = copy.deepcopy(model).half().eval()
        _replicate_to_zero_padding(self.model)

    def forward(self, x):
        return self.model(x.half()).float()


def to_int8(model: nn.Module, calib_data: torch.Tensor, batch_size: int = 16) -> nn.Module:
    """INT8 post-training quantization (CPU).

    Models made only of ``nn.Linear`` layers use dynamic quantization.
    Convolutional models use static FX-graph quantization with per-tensor
    symmetric weights, calibrated on ``calib_data`` (healthy training data).
    """
    model = copy.deepcopy(model).cpu().eval()
    has_conv = any(isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) for m in model.modules())
    if not has_conv:
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    from torch.ao.quantization import MinMaxObserver, QConfig, QConfigMapping, quantize_fx

    torch.backends.quantized.engine = "qnnpack" if sys.platform == "darwin" else "fbgemm"
    _replicate_to_zero_padding(model)
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        # per-channel weights are not supported for ConvTranspose1d
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )
    prepared = quantize_fx.prepare_fx(model, QConfigMapping().set_global(qconfig),
                                      example_inputs=(calib_data[:1],))
    with torch.no_grad():
        for i in range(0, len(calib_data), batch_size):
            prepared(calib_data[i:i + batch_size])
    return quantize_fx.convert_fx(prepared)
