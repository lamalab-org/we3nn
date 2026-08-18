from __future__ import annotations

import torch
from torch import nn

from .representation_tensor import RepresentationTensor


class SequentialModule(nn.Sequential):
    """Representation-aware ``torch.nn.Sequential`` operating on tensors."""

    def __init__(self, *args):
        super().__init__(*args)
        modules = list(self._modules.values())
        if not modules:
            raise ValueError("SequentialModule needs at least one module")
        for left, right in zip(modules, modules[1:]):
            if not hasattr(left, "out_type") or not hasattr(right, "in_type"):
                raise TypeError("all children must expose in_type and out_type")
            if left.out_type != right.in_type:
                raise ValueError(f"type mismatch between {left!r} and {right!r}")
        self.in_type = modules[0].in_type
        self.out_type = modules[-1].out_type

    def forward(
        self,
        input: torch.Tensor | RepresentationTensor,
    ) -> torch.Tensor | RepresentationTensor:
        shape = input.shape if isinstance(input, (torch.Tensor, RepresentationTensor)) else None
        if shape is None or shape[-1] != self.in_type.size:
            raise TypeError(f"expected a tensor with final dimension {self.in_type.size}")
        return super().forward(input)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        shape = input_shape
        for module in self:
            shape = module.evaluate_output_shape(shape)
        return shape
