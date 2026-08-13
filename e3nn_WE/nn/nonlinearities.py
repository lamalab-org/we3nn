from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .field_type import FieldType, as_field_type
from ..representations import Representation


class PointwiseNonLinearity(nn.Module):
    """Apply one scalar function to permutation-representation coordinates."""

    def __init__(self, in_type: FieldType, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        unsupported = [rep.name for rep in in_type if not rep.is_permutation]
        if unsupported:
            raise ValueError(
                "Pointwise activation is not equivariant in the current basis of "
                f"representations {unsupported}. Use a permutation representation, "
                "norm activation, gated activation, or change to a supported basis."
            )
        self.in_type = in_type
        self.out_type = in_type
        self.function = function

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, torch.Tensor):
            raise TypeError("pointwise activations expect an ordinary torch.Tensor")
        if input.shape[-1] != self.in_type.size:
            raise ValueError(f"expected last dimension {self.in_type.size}, got {input.shape[-1]}")
        return self.function(input)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape


class ReLU(PointwiseNonLinearity):
    def __init__(self, in_type: FieldType, inplace: bool = False):
        self.inplace = inplace
        super().__init__(in_type, lambda tensor: F.relu(tensor, inplace=self.inplace))


class ELU(PointwiseNonLinearity):
    def __init__(self, in_type: FieldType, alpha: float = 1.0, inplace: bool = False):
        self.alpha = float(alpha)
        self.inplace = inplace
        super().__init__(in_type, lambda tensor: F.elu(tensor, alpha=self.alpha, inplace=self.inplace))


class PointwiseActivation(PointwiseNonLinearity):
    """Generic pointwise activation accepting a Representation and raw tensors."""

    def __init__(self, representation: FieldType | Representation, activation: Callable[[torch.Tensor], torch.Tensor]):
        field_type = (
            as_field_type(representation)
            if isinstance(representation, Representation)
            else representation
        )
        super().__init__(field_type, activation)
