"""Typed equivariant PyTorch modules for finite groups."""

from .field_type import FieldType
from .geometric_tensor import GeometricTensor
from .linear import Linear
from .nonlinearities import ELU, ReLU, PointwiseActivation, PointwiseNonLinearity
from .sequential import SequentialModule
from .tensor_product import FullTensorProduct, FullyConnectedTensorProduct, TensorProduct, TensorProductInstruction
from .wigner_eckart import RestrictedWignerEckartTensorProduct, WignerEckartTensorProduct

__all__ = [
    "ELU",
    "FieldType",
    "GeometricTensor",
    "FullyConnectedTensorProduct",
    "FullTensorProduct",
    "Linear",
    "PointwiseNonLinearity",
    "PointwiseActivation",
    "ReLU",
    "SequentialModule",
    "TensorProduct",
    "TensorProductInstruction",
    "WignerEckartTensorProduct",
    "RestrictedWignerEckartTensorProduct",
]

# Kept here as a lazy import target to avoid a package import cycle.
def __getattr__(name):
    if name in {"CircularHarmonics", "RestrictedSphericalHarmonics"}:
        from ..harmonics import CircularHarmonics, RestrictedSphericalHarmonics

        return {"CircularHarmonics": CircularHarmonics, "RestrictedSphericalHarmonics": RestrictedSphericalHarmonics}[name]
    raise AttributeError(name)
