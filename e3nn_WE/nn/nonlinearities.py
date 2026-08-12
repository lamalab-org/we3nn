from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .field_type import FieldType
from .geometric_tensor import GeometricTensor


class PointwiseNonLinearity(nn.Module):
    """Apply one scalar function to permutation-representation coordinates."""

    def __init__(self, in_type: FieldType, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        unsupported = [rep.name for rep in in_type if not rep.is_permutation]
        if unsupported:
            raise ValueError(
                "pointwise nonlinearities require permutation or one-dimensional representations; "
                f"unsupported: {unsupported}"
            )
        self.in_type = in_type
        self.out_type = in_type
        self.function = function

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        if not isinstance(input, GeometricTensor) or input.type != self.in_type:
            raise TypeError(f"expected a GeometricTensor of type {self.in_type!r}")
        return GeometricTensor(self.function(input.tensor), self.out_type)

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
