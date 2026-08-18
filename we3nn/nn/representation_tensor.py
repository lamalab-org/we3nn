"""Optional tensor wrapper carrying finite-group representation metadata."""

from __future__ import annotations

from collections.abc import Iterable
import warnings

import torch

from ..representations import Representation
from .field_type import FieldType, as_field_type


class MissingRepresentationMetadataWarning(UserWarning):
    """A tensor product received a raw tensor whose representation is unknown."""


class RepresentationTensor:
    """Pair a :class:`torch.Tensor` with the ``FieldType`` of its last axis.

    This is an opt-in safety wrapper, not a tensor subclass. Use :attr:`tensor`
    whenever an ordinary PyTorch tensor is needed. Representation-aware we3nn
    modules accept this wrapper directly and preserve metadata on their output.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        field_type: FieldType | Representation,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("RepresentationTensor.tensor must be a torch.Tensor")
        normalized = normalize_field_type(field_type)
        if tensor.ndim == 0 or tensor.shape[-1] != normalized.size:
            raise ValueError(
                f"tensor's last dimension must be {normalized.size}, "
                f"got {tuple(tensor.shape)}"
            )
        self.tensor = tensor
        self.field_type = normalized

    @property
    def representation(self) -> Representation:
        return self.field_type.representation

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, *args, **kwargs) -> "RepresentationTensor":
        return RepresentationTensor(self.tensor.to(*args, **kwargs), self.field_type)

    def clone(self, *args, **kwargs) -> "RepresentationTensor":
        return RepresentationTensor(self.tensor.clone(*args, **kwargs), self.field_type)

    def detach(self) -> "RepresentationTensor":
        return RepresentationTensor(self.tensor.detach(), self.field_type)

    def transform_fibers(self, element) -> "RepresentationTensor":
        return RepresentationTensor(
            self.field_type.transform_fibers(self.tensor, element),
            self.field_type,
        )

    def __repr__(self) -> str:
        return (
            f"RepresentationTensor(shape={tuple(self.tensor.shape)}, "
            f"field_type={self.field_type!r})"
        )


def normalize_field_type(value: FieldType | Representation) -> FieldType:
    if isinstance(value, Representation):
        return as_field_type(value)
    if isinstance(value, FieldType):
        return value
    raise TypeError("representation metadata must be a FieldType or Representation")


def unpack_representation_tensor(
    value: torch.Tensor | RepresentationTensor,
    expected: FieldType,
    argument: str,
) -> tuple[torch.Tensor, bool]:
    """Return the raw tensor and whether representation metadata was present."""
    if isinstance(value, RepresentationTensor):
        if value.field_type != expected:
            raise TypeError(
                f"{argument} representation mismatch: expected {expected!r}, "
                f"got {value.field_type!r}"
            )
        return value.tensor, True
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{argument} must be a torch.Tensor or RepresentationTensor")
    return value, False


def unpack_tensor_product_inputs(
    values: Iterable[tuple[str, torch.Tensor | RepresentationTensor, FieldType]],
    operation: str,
) -> tuple[list[torch.Tensor], bool]:
    """Validate tensor-product inputs and warn once for all missing metadata."""
    tensors = []
    missing = []
    all_typed = True
    for argument, value, expected in values:
        tensor, typed = unpack_representation_tensor(value, expected, argument)
        tensors.append(tensor)
        all_typed = all_typed and typed
        if not typed:
            missing.append(argument)
    if missing:
        warnings.warn(
            f"{operation} received raw tensor input(s) without representation "
            f"metadata: {', '.join(missing)}. Wrap them with "
            "RepresentationTensor to enable representation checks.",
            MissingRepresentationMetadataWarning,
            stacklevel=5,
        )
    return tensors, all_typed


def wrap_if_typed(
    tensor: torch.Tensor,
    field_type: FieldType,
    typed: bool,
) -> torch.Tensor | RepresentationTensor:
    return RepresentationTensor(tensor, field_type) if typed else tensor
