"""Model zoo and per-model defaults.

Each entry in ``MODELS`` holds the model class, the rule used to pad the FRF
band to a valid input length, and the training/evaluation defaults used for
the results in the paper.
"""

from dataclasses import dataclass, field

from .att_cnn_ae import AttCNNAutoEncoder
from .cnn_ae import CNNAutoEncoder
from .fc_ae import FCAutoEncoder


def pad_pow2(length: int, base: int = 32) -> int:
    """Smallest ``base * 2**k`` (k >= 1) that is >= ``length``."""
    size = 2 * base
    while size < length:
        size *= 2
    return size


def pad_multiple(length: int, multiple: int = 16) -> int:
    """Smallest multiple of ``multiple`` that is >= ``length``."""
    return -(-length // multiple) * multiple


@dataclass(frozen=True)
class ModelSpec:
    cls: type
    input_size: callable          # band length -> valid model input length
    lr: float
    betas: tuple = (0.9, 0.999)
    quantile: float = 0.5         # threshold percentile of healthy validation errors
    restore_best: bool = False    # evaluate best-validation weights instead of last epoch
    kwargs: dict = field(default_factory=dict)

    def build(self, input_dim: int):
        if self.cls is FCAutoEncoder:
            return self.cls(input_dim=input_dim, **self.kwargs)
        return self.cls(**self.kwargs)


MODELS = {
    "fc_ae": ModelSpec(FCAutoEncoder, pad_pow2, lr=5e-3, betas=(0.9, 0.98),
                       quantile=0.5, kwargs={"latent_dim": 32}),
    "cnn_ae": ModelSpec(CNNAutoEncoder, pad_multiple, lr=5e-3, betas=(0.9, 0.98),
                        quantile=0.5),
    "att_cnn_ae": ModelSpec(AttCNNAutoEncoder, pad_pow2, lr=1e-3,
                            quantile=0.7, restore_best=True),
}


def build_model(name: str, input_dim: int):
    return MODELS[name].build(input_dim)


__all__ = ["FCAutoEncoder", "CNNAutoEncoder", "AttCNNAutoEncoder", "MODELS", "ModelSpec",
           "build_model", "pad_pow2", "pad_multiple"]
