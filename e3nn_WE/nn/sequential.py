from __future__ import annotations

from collections import OrderedDict

from torch import nn

from .geometric_tensor import GeometricTensor


class SequentialModule(nn.Sequential):
    """Type-checking counterpart of ``torch.nn.Sequential`` for geometric tensors."""

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

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        if not isinstance(input, GeometricTensor) or input.type != self.in_type:
            raise TypeError(f"expected a GeometricTensor of type {self.in_type!r}")
        return super().forward(input)

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        shape = input_shape
        for module in self:
            shape = module.evaluate_output_shape(shape)
        return shape
