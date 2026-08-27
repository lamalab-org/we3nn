"""Equivariant PyTorch tensor modules for finite groups."""

from .field_type import FieldType
from .linear import WELinear
from .nonlinearities import ELU, ReLU, PointActiv, PointwiseNonLinearity
from .sequential import SequentialModule
from .tensor_product import (
    FullTensorProduct,
    FullyConnectedTensorProduct,
    MultiplicityChunk,
    TensorProduct,
    TensorProductInstruction,
)
from .kernel_tensor_product import SphericalKernelTensorProduct, KernelTensorProduct
from .representation_tensor import (
    MissingRepresentationMetadataWarning,
    RepresentationTensor,
)

__all__ = [
    "ELU",
    "FieldType",
    "FullyConnectedTensorProduct",
    "FullTensorProduct",
    "MissingRepresentationMetadataWarning",
    "MultiplicityChunk",
    "WELinear",
    "PointwiseNonLinearity",
    "PointActiv",
    "ReLU",
    "RepresentationTensor",
    "SequentialModule",
    "TensorProduct",
    "TensorProductInstruction",
    "KernelTensorProduct",
    "SphericalKernelTensorProduct",
]

# Kept here as a lazy import target to avoid a package import cycle.
def __getattr__(name):
    if name in {"CircularHarmonics", "RestrictedSphericalHarmonics"}:
        from ..harmonics import CircularHarmonics, RestrictedSphericalHarmonics

        return {"CircularHarmonics": CircularHarmonics, "RestrictedSphericalHarmonics": RestrictedSphericalHarmonics}[name]
    raise AttributeError(name)
