from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property

import torch

from ..gspaces import GSpace
from ..representations import DirectSumRepresentation, Representation, direct_sum


class FieldType(Sequence[Representation]):
    """A direct sum of feature fields transforming under one fiber group."""

    def __init__(self, gspace: GSpace, representations: Sequence[Representation]):
        if not isinstance(gspace, GSpace):
            raise TypeError("gspace must be a no-base-space GSpace")
        self.gspace = gspace
        self.fibergroup = gspace.fibergroup
        self.representations = tuple(representations)
        if not self.representations:
            raise ValueError("FieldType needs at least one representation")
        if any(rep.group is not self.fibergroup for rep in self.representations):
            raise ValueError("all representations must belong to the gspace fiber group")
        self.size = sum(rep.size for rep in self.representations)
        offsets = [0]
        for rep in self.representations:
            offsets.append(offsets[-1] + rep.size)
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return len(self.representations)

    def __getitem__(self, index):
        return self.representations[index]

    def __iter__(self) -> Iterator[Representation]:
        return iter(self.representations)

    @property
    def fields_start(self) -> tuple[int, ...]:
        return self._offsets[:-1]

    @property
    def fields_end(self) -> tuple[int, ...]:
        return self._offsets[1:]

    @cached_property
    def representation(self) -> Representation:
        return direct_sum(self.representations, name=f"FieldType[{len(self)}]")

    def transform_fibers(self, tensor: torch.Tensor, element) -> torch.Tensor:
        if tensor.shape[-1] != self.size:
            raise ValueError(f"last dimension must be {self.size}, got {tensor.shape[-1]}")
        matrix = self.representation(element).to(device=tensor.device, dtype=tensor.dtype)
        return tensor @ matrix.T

    def wrap(self, tensor: torch.Tensor):
        """Attach this field type to ``tensor`` for runtime representation checks."""
        from .representation_tensor import RepresentationTensor

        return RepresentationTensor(tensor, self)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FieldType)
            and self.gspace == other.gspace
            and self.representations == other.representations
        )

    def __repr__(self) -> str:
        reps = ", ".join(rep.name for rep in self.representations)
        return f"FieldType({self.gspace.name}, [{reps}])"


def as_field_type(representation: Representation) -> FieldType:
    """Expose direct-sum blocks to structured kernels without wrapping tensors."""
    representations = (
        representation.representations
        if isinstance(representation, DirectSumRepresentation)
        else (representation,)
    )
    return FieldType(GSpace(representation.group), representations)
