from __future__ import annotations

import torch

from .field_type import FieldType


class GeometricTensor:
    """A tensor bundled with the transformation law of its last axis."""

    def __init__(self, tensor: torch.Tensor, type: FieldType):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if tensor.ndim < 1 or tensor.shape[-1] != type.size:
            raise ValueError(f"expected last dimension {type.size}, got shape {tuple(tensor.shape)}")
        self.tensor = tensor
        self.type = type

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def device(self):
        return self.tensor.device

    @property
    def dtype(self):
        return self.tensor.dtype

    def transform_fibers(self, element) -> "GeometricTensor":
        return GeometricTensor(self.type.transform_fibers(self.tensor, element), self.type)

    def transform(self, element) -> "GeometricTensor":
        return self.transform_fibers(element)

    def to(self, *args, **kwargs) -> "GeometricTensor":
        return GeometricTensor(self.tensor.to(*args, **kwargs), self.type)

    def clone(self) -> "GeometricTensor":
        return GeometricTensor(self.tensor.clone(), self.type)

    def __add__(self, other: "GeometricTensor") -> "GeometricTensor":
        if not isinstance(other, GeometricTensor) or other.type != self.type:
            raise TypeError("GeometricTensors can only be added when their FieldTypes match")
        return GeometricTensor(self.tensor + other.tensor, self.type)

    def __repr__(self) -> str:
        return f"GeometricTensor(shape={tuple(self.shape)}, type={self.type!r})"
