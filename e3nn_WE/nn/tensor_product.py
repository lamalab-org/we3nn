"""e3nn-style bilinear tensor products for finite groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn

from ..clebsch_gordan import clebsch_gordan, full_coupling_basis
from ..representations import Irrep, Representation
from ..intertwiner import intertwiner_basis, tensor_product_representation
from .field_type import FieldType
from .geometric_tensor import GeometricTensor
from ..gspaces import no_base_space


@dataclass(frozen=True)
class TensorProductInstruction:
    i_in1: int
    i_in2: int
    i_out: int
    path_shape: tuple[int, ...]
    path_weight: float = 1.0

    @property
    def path_numel(self) -> int:
        return math.prod(self.path_shape)


def _regular_indices(group) -> torch.Tensor:
    """indices[q, a] selects ``(rho_regular(q).T x)[a]`` from x."""
    index = {element.value: i for i, element in enumerate(group.elements)}
    return torch.tensor(
        [[index[group.combine(q, a).value] for a in group.elements] for q in group.elements],
        dtype=torch.long,
    )


class _TensorProductPath(nn.Module):
    def __init__(
        self,
        left: Representation,
        right: Representation,
        output: Representation,
        *,
        internal_weights: bool,
        path_weight: float,
    ):
        super().__init__()
        self.left = left
        self.right = right
        self.output = output
        self.group = output.group
        self.path_weight = float(path_weight)
        regular = self.group.regular_repr
        if output is regular:
            self.kind = "output_regular"
            self.weight_shape = (left.size, right.size)
        elif left is regular:
            self.kind = "left_regular"
            self.weight_shape = (output.size, right.size)
        elif right is regular:
            self.kind = "right_regular"
            self.weight_shape = (output.size, left.size)
        else:
            self.kind = "cg"
            self.weight_shape = (_generic_couplings(left, right, output).shape[0],)
        if math.prod(self.weight_shape) == 0:
            raise ValueError("the requested tensor-product instruction has no equivariant coupling")
        if internal_weights:
            self.weight = nn.Parameter(torch.empty(self.weight_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    @property
    def weight_numel(self) -> int:
        return math.prod(self.weight_shape)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            bound = math.sqrt(6.0 / (self.left.size + self.right.size + self.output.size))
            nn.init.uniform_(self.weight, -bound, bound)

    def _matrices(self, representation: Representation, reference: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [representation(element) for element in self.group.elements]
        ).to(device=reference.device, dtype=reference.dtype)

    def _inverse_transforms(self, value: torch.Tensor, representation: Representation) -> torch.Tensor:
        if representation is self.group.regular_repr:
            indices = _regular_indices(self.group).to(value.device)
            return value[..., indices]
        matrices = self._matrices(representation, value)
        return torch.einsum("...i,qia->...qa", value, matrices)

    def forward(self, left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        coefficients = self.weight if weight is None else weight
        if coefficients is None:
            raise RuntimeError("external tensor-product weights were not supplied")
        shared = coefficients.ndim == len(self.weight_shape)
        scale = self.path_weight
        if self.kind == "cg":
            basis = _generic_couplings(self.left, self.right, self.output).to(
                device=left.device, dtype=left.dtype
            )
            if shared:
                output = torch.einsum("p,poij,...i,...j->...o", coefficients, basis, left, right)
            else:
                output = torch.einsum("...p,poij,...i,...j->...o", coefficients, basis, left, right)
            return scale * output

        order_scale = math.sqrt(self.group.order())
        if self.kind == "output_regular":
            left_transformed = self._inverse_transforms(left, self.left)
            right_transformed = self._inverse_transforms(right, self.right)
            if shared:
                output = torch.einsum(
                    "ab,...qa,...qb->...q", coefficients, left_transformed, right_transformed
                )
            else:
                output = torch.einsum(
                    "...ab,...qa,...qb->...q", coefficients, left_transformed, right_transformed
                )
            return scale * output / order_scale

        output_matrices = self._matrices(self.output, left)
        if self.kind == "left_regular":
            right_transformed = self._inverse_transforms(right, self.right)
            if shared:
                intermediate = torch.einsum("ab,...qb,...q->...qa", coefficients, right_transformed, left)
            else:
                intermediate = torch.einsum("...ab,...qb,...q->...qa", coefficients, right_transformed, left)
        else:
            left_transformed = self._inverse_transforms(left, self.left)
            if shared:
                intermediate = torch.einsum("ab,...qb,...q->...qa", coefficients, left_transformed, right)
            else:
                intermediate = torch.einsum("...ab,...qb,...q->...qa", coefficients, left_transformed, right)
        return scale * torch.einsum("qoa,...qa->...o", output_matrices, intermediate) / order_scale


class TensorProduct(nn.Module):
    """A complete equivariant bilinear map between finite-group FieldTypes.

    By default, every compatible triple of input/output fields is included.
    ``instructions`` may instead contain ``(i_in1, i_in2, i_out)`` triples or
    :class:`TensorProductInstruction` objects. Each instruction spans every
    independent real equivariant coupling for that field triple.

    With ``internal_weights=False``, pass a final-axis weight tensor to
    :meth:`forward`. Shared weights have shape ``(weight_numel,)``; with
    ``shared_weights=False`` they have shape ``(..., weight_numel)`` matching
    the inputs' leading dimensions.
    """

    def __init__(
        self,
        in1_type: FieldType | Representation,
        in2_type: FieldType | Representation,
        out_type: FieldType | Representation,
        instructions: Iterable[TensorProductInstruction | tuple[int, int, int]] | None = None,
        *,
        internal_weights: bool = True,
        shared_weights: bool = True,
    ):
        super().__init__()
        self._raw_tensor_api = all(isinstance(value, Representation) for value in (in1_type, in2_type, out_type))
        if isinstance(in1_type, Representation):
            in1_type = FieldType(no_base_space(in1_type.group), [in1_type])
        if isinstance(in2_type, Representation):
            in2_type = FieldType(no_base_space(in2_type.group), [in2_type])
        if isinstance(out_type, Representation):
            out_type = FieldType(no_base_space(out_type.group), [out_type])
        if in1_type.fibergroup is not in2_type.fibergroup or in1_type.fibergroup is not out_type.fibergroup:
            raise ValueError("all FieldTypes must use the same group instance")
        if internal_weights and not shared_weights:
            raise ValueError("internal tensor-product weights must be shared")
        self.in1_type = in1_type
        self.in2_type = in2_type
        self.out_type = out_type
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)

        requested = []
        if instructions is None:
            for i_out, output in enumerate(out_type):
                for i_in1, left in enumerate(in1_type):
                    for i_in2, right in enumerate(in2_type):
                        dimension = _coupling_dimension(left, right, output)
                        if dimension:
                            requested.append((i_in1, i_in2, i_out, 1.0))
        else:
            for instruction in instructions:
                if isinstance(instruction, TensorProductInstruction):
                    requested.append(
                        (instruction.i_in1, instruction.i_in2, instruction.i_out, instruction.path_weight)
                    )
                else:
                    if len(instruction) != 3:
                        raise ValueError("instructions must be (i_in1, i_in2, i_out) triples")
                    requested.append((*map(int, instruction), 1.0))

        paths = []
        normalized_instructions = []
        for i_in1, i_in2, i_out, path_weight in requested:
            left, right, output = in1_type[i_in1], in2_type[i_in2], out_type[i_out]
            path = _TensorProductPath(
                left, right, output, internal_weights=internal_weights, path_weight=path_weight
            )
            paths.append(path)
            normalized_instructions.append(
                TensorProductInstruction(i_in1, i_in2, i_out, path.weight_shape, path_weight)
            )
        self.paths = nn.ModuleList(paths)
        self.instructions = tuple(normalized_instructions)
        self.weight_numel = sum(path.weight_numel for path in self.paths)

    def forward(
        self,
        input1: GeometricTensor | torch.Tensor,
        input2: GeometricTensor | torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> GeometricTensor:
        raw = isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor)
        if raw:
            if not self._raw_tensor_api:
                raise TypeError("raw tensors require TensorProduct construction from Representations")
            input1 = GeometricTensor(input1, self.in1_type)
            input2 = GeometricTensor(input2, self.in2_type)
        if not isinstance(input1, GeometricTensor) or input1.type != self.in1_type:
            raise TypeError(f"input1 must have type {self.in1_type!r}")
        if not isinstance(input2, GeometricTensor) or input2.type != self.in2_type:
            raise TypeError(f"input2 must have type {self.in2_type!r}")
        if input1.tensor.shape[:-1] != input2.tensor.shape[:-1]:
            raise ValueError("tensor-product inputs must have matching leading dimensions")
        if self.internal_weights:
            if weight is not None:
                raise ValueError("an internally weighted TensorProduct does not accept external weights")
        else:
            expected = (self.weight_numel,) if self.shared_weights else (*input1.tensor.shape[:-1], self.weight_numel)
            if weight is None or tuple(weight.shape) != expected:
                raise ValueError(f"external weight must have shape {expected}")

        output = input1.tensor.new_zeros(*input1.tensor.shape[:-1], self.out_type.size)
        weight_offset = 0
        for instruction, path in zip(self.instructions, self.paths):
            left = input1.tensor[..., self.in1_type.fields_start[instruction.i_in1]:self.in1_type.fields_end[instruction.i_in1]]
            right = input2.tensor[..., self.in2_type.fields_start[instruction.i_in2]:self.in2_type.fields_end[instruction.i_in2]]
            external = None
            if not self.internal_weights:
                external = weight[..., weight_offset:weight_offset + path.weight_numel].reshape(
                    *weight.shape[:-1], *path.weight_shape
                )
                weight_offset += path.weight_numel
            value = path(left, right, external)
            start, end = self.out_type.fields_start[instruction.i_out], self.out_type.fields_end[instruction.i_out]
            output[..., start:end] = output[..., start:end] + value
        result = GeometricTensor(output, self.out_type)
        return result.tensor if raw else result

    def evaluate_output_shape(self, input1_shape: tuple[int, ...], input2_shape: tuple[int, ...]) -> tuple[int, ...]:
        if input1_shape[:-1] != input2_shape[:-1]:
            raise ValueError("input leading dimensions must match")
        if input1_shape[-1] != self.in1_type.size or input2_shape[-1] != self.in2_type.size:
            raise ValueError("input feature dimensions do not match the FieldTypes")
        return (*input1_shape[:-1], self.out_type.size)

    @torch.no_grad()
    def check_equivariance(self, atol: float = 2e-5, rtol: float = 2e-5):
        reference = next(self.parameters(), None)
        if reference is None:
            reference = torch.empty(0)
        shape = (3,)
        left = GeometricTensor(
            torch.randn(*shape, self.in1_type.size, device=reference.device, dtype=reference.dtype), self.in1_type
        )
        right = GeometricTensor(
            torch.randn(*shape, self.in2_type.size, device=reference.device, dtype=reference.dtype), self.in2_type
        )
        external = None
        if not self.internal_weights:
            weight_shape = (self.weight_numel,) if self.shared_weights else (*shape, self.weight_numel)
            external = torch.randn(*weight_shape, device=reference.device, dtype=reference.dtype)
        output = self(left, right, external)
        errors = []
        for element in self.in1_type.fibergroup.elements:
            actual = self(left.transform_fibers(element), right.transform_fibers(element), external).tensor
            expected = output.transform_fibers(element).tensor
            error = float((actual - expected).abs().max())
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                raise AssertionError(f"tensor-product equivariance failed for {element}: {error:.3e}")
            errors.append((element, error))
        return errors


class FullyConnectedTensorProduct(TensorProduct):
    """Alias emphasizing that all compatible field triples are connected."""


class FullTensorProduct(nn.Module):
    """Unweighted direct tensor product in the product coordinate basis."""

    def __init__(self, in1_type: FieldType, in2_type: FieldType):
        super().__init__()
        if in1_type.fibergroup is not in2_type.fibergroup:
            raise ValueError("input types must use the same group")
        self.in1_type = in1_type
        self.in2_type = in2_type
        self.out_type = FieldType(
            in1_type.gspace,
            [tensor_product_representation(in1_type.representation, in2_type.representation)],
        )

    def forward(self, input1: GeometricTensor, input2: GeometricTensor) -> GeometricTensor:
        if input1.type != self.in1_type or input2.type != self.in2_type:
            raise TypeError("FullTensorProduct input types do not match")
        if input1.tensor.shape[:-1] != input2.tensor.shape[:-1]:
            raise ValueError("input leading dimensions must match")
        output = torch.einsum("...i,...j->...ij", input1.tensor, input2.tensor).flatten(-2)
        return GeometricTensor(output, self.out_type)


def _coupling_dimension(left: Representation, right: Representation, output: Representation) -> int:
    regular = output.group.regular_repr
    if output is regular:
        return left.size * right.size
    if left is regular:
        return output.size * right.size
    if right is regular:
        return output.size * left.size
    return int(_generic_couplings(left, right, output).shape[0])


def _generic_couplings(left: Representation, right: Representation, output: Representation) -> torch.Tensor:
    if isinstance(left, Irrep) and isinstance(right, Irrep) and isinstance(output, Irrep):
        return full_coupling_basis(left, right, output)
    basis = intertwiner_basis(tensor_product_representation(left, right), output)
    return basis.reshape(-1, output.size, left.size, right.size)
