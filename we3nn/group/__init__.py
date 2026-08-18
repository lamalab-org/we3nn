"""Generic finite-group representations and neural-network operations."""

from we3nn import *
from we3nn import __all__ as _extension_all
from we3nn import embeddings, nn
from we3nn.nn import PointActiv, SphericalKernelTensorProduct, WELinear, KernelTensorProduct
from . import utils

__all__ = [
    *_extension_all,
    "PointActiv",
    "SphericalKernelTensorProduct",
    "WELinear",
    "KernelTensorProduct",
    "embeddings",
    "nn",
    "utils",
]
