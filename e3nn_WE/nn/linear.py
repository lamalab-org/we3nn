"""All finite-group equivariant linear maps between two FieldTypes."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..representations import Representation
from ..intertwiner import intertwiner_basis as generic_intertwiner_basis
from ..gspaces import no_base_space
from .field_type import FieldType
from .geometric_tensor import GeometricTensor


@lru_cache(maxsize=None)
def _intertwiner_basis(out_rep: Representation, in_rep: Representation) -> torch.Tensor:
    """Orthonormal basis of W satisfying rho_out(g) W = W rho_in(g)."""
    if out_rep.group is not in_rep.group:
        raise ValueError("representations belong to different groups")
    group = out_rep.group
    regular = group.regular_repr

    # Frobenius-orthonormal analytic basis for Hom(regular, rho). An
    # intertwiner is uniquely determined by the image of the identity delta.
    # This avoids an O(|G|^4) null-space construction for regular fields.
    if in_rep is regular:
        basis = torch.empty(out_rep.size, out_rep.size, group.order(), dtype=torch.float64)
        scale = math.sqrt(group.order())
        for column, element in enumerate(group.elements):
            # basis[a, :, g] = rho(g) e_a
            basis[:, :, column] = out_rep(element).T / scale
        return basis
    if out_rep is regular:
        return _intertwiner_basis(in_rep, regular).transpose(1, 2).contiguous()

    out_size, in_size = out_rep.size, in_rep.size
    candidates = []
    for flat_index in range(out_size * in_size):
        elementary = torch.zeros(out_size, in_size, dtype=torch.float64)
        elementary.reshape(-1)[flat_index] = 1.0
        projected = torch.zeros_like(elementary)
        for element in out_rep.group.elements:
            projected += out_rep(element) @ elementary @ in_rep(element).T
        candidates.append((projected / out_rep.group.order()).reshape(-1))
    span = torch.stack(candidates)
    _, singular_values, vh = torch.linalg.svd(span, full_matrices=False)
    # An exactly empty Reynolds projection still contains trigonometric
    # round-off. Use an absolute floor so it cannot become a spurious map.
    tolerance = max(1e-10, float(max(span.shape) * torch.finfo(span.dtype).eps * singular_values.max()))
    rank = int((singular_values > tolerance).sum())
    return vh[:rank].reshape(rank, out_size, in_size).contiguous()


def _intertwiner_dimension(out_rep: Representation, in_rep: Representation) -> int:
    if out_rep.group is not in_rep.group:
        return 0
    regular = out_rep.group.regular_repr
    if in_rep is regular:
        return out_rep.size
    if out_rep is regular:
        return in_rep.size
    return int(_intertwiner_basis(out_rep, in_rep).shape[0])


class _PairExpansion(nn.Module):
    def __init__(
        self,
        out_rep: Representation,
        in_rep: Representation,
        row_starts: list[int],
        column_starts: list[int],
        backend: str = "auto",
    ):
        super().__init__()
        if backend == "generic":
            generic_basis = generic_intertwiner_basis(in_rep, out_rep)
            dimension = generic_basis.shape[0]
        else:
            generic_basis = None
            dimension = _intertwiner_dimension(out_rep, in_rep)
        if dimension == 0:
            raise RuntimeError("attempted to construct an empty intertwiner block")
        # Occurrences are a Cartesian product of all fields with this pair of
        # representation identities. Store each field index once rather than
        # repeating a full row/column index grid for every block.
        unique_rows = tuple(dict.fromkeys(row_starts))
        unique_columns = tuple(dict.fromkeys(column_starts))
        if len(unique_rows) * len(unique_columns) != len(row_starts):
            raise RuntimeError("internal field-pair grouping is not Cartesian")
        self.out_rep = out_rep
        self.in_rep = in_rep
        self.backend = backend
        self._regular_to_regular = (
            backend != "generic"
            and out_rep is out_rep.group.regular_repr
            and in_rep is out_rep.group.regular_repr
        )
        self.coefficients = nn.Parameter(torch.empty(len(unique_rows), len(unique_columns), dimension))
        if self._regular_to_regular:
            index = {element.value: i for i, element in enumerate(out_rep.group.elements)}
            relative = torch.tensor(
                [
                    [index[out_rep.group.combine(column.inverse(), row).value] for column in out_rep.group.elements]
                    for row in out_rep.group.elements
                ],
                dtype=torch.long,
            )
            self.register_buffer("relative", relative, persistent=False)
            self.register_buffer("basis", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "basis",
                (generic_basis if generic_basis is not None else _intertwiner_basis(out_rep, in_rep)).to(torch.get_default_dtype()),
                persistent=False,
            )
            self.register_buffer("relative", torch.empty(0, dtype=torch.long), persistent=False)
        self._row_slice = _contiguous_field_slice(unique_rows, out_rep.size)
        self._column_slice = _contiguous_field_slice(unique_columns, in_rep.size)
        rows = torch.tensor(unique_rows)[:, None] + torch.arange(out_rep.size)[None, :]
        columns = torch.tensor(unique_columns)[:, None] + torch.arange(in_rep.size)[None, :]
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("columns", columns, persistent=False)
        self.out_size = out_rep.size
        self.in_size = in_rep.size

    def reset_parameters(self, fan_in: int, fan_out: int) -> None:
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.coefficients, -bound, bound)

    def write_into(self, weight: torch.Tensor) -> None:
        if self._regular_to_regular:
            blocks = self.coefficients[..., self.relative] / math.sqrt(self.out_rep.group.order())
            # coefficient axes (out field, in field) precede (out coord, in coord)
            if self._row_slice is not None and self._column_slice is not None:
                dense_block = blocks.permute(0, 2, 1, 3).reshape(
                    self.coefficients.shape[0] * self.out_size,
                    self.coefficients.shape[1] * self.in_size,
                )
                weight[self._row_slice, self._column_slice] = dense_block
            else:
                weight[self.rows[:, None, :, None], self.columns[None, :, None, :]] = blocks
            return
        blocks = torch.einsum("rcp,poi->rcoi", self.coefficients, self.basis)
        if self._row_slice is not None and self._column_slice is not None:
            dense_block = blocks.permute(0, 2, 1, 3).reshape(
                self.coefficients.shape[0] * self.out_size,
                self.coefficients.shape[1] * self.in_size,
            )
            weight[self._row_slice, self._column_slice] = dense_block
        else:
            weight[self.rows[:, None, :, None], self.columns[None, :, None, :]] = blocks

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if not self._regular_to_regular:
            # Rebuild from the cached float64 mathematical source. In
            # particular, ``module.double()`` must not merely widen a basis
            # which was rounded to float32 during construction.
            if self.backend == "generic":
                source = generic_intertwiner_basis(self.in_rep, self.out_rep)
            else:
                source = _intertwiner_basis(self.out_rep, self.in_rep)
            self.basis = source.to(
                device=self.coefficients.device, dtype=self.coefficients.dtype
            )
        return self


def _contiguous_field_slice(starts: tuple[int, ...], field_size: int) -> slice | None:
    if all(value == starts[0] + index * field_size for index, value in enumerate(starts)):
        return slice(starts[0], starts[0] + len(starts) * field_size)
    return None


class _BiasExpansion(nn.Module):
    def __init__(self, out_rep: Representation, row_starts: list[int]):
        super().__init__()
        self.out_rep = out_rep
        trivial = out_rep.group.trivial_representation
        basis = _intertwiner_basis(out_rep, trivial)[:, :, 0].to(torch.get_default_dtype())
        self.coefficients = nn.Parameter(torch.empty(len(row_starts), basis.shape[0]))
        self.register_buffer("basis", basis, persistent=False)
        self._row_slice = _contiguous_field_slice(tuple(row_starts), out_rep.size)
        rows = torch.tensor(row_starts)[:, None] + torch.arange(out_rep.size)[None, :]
        self.register_buffer("rows", rows, persistent=False)

    def reset_parameters(self, bound: float) -> None:
        nn.init.uniform_(self.coefficients, -bound, bound)

    def write_into(self, bias: torch.Tensor) -> None:
        values = torch.einsum("cp,po->co", self.coefficients, self.basis)
        if self._row_slice is not None:
            bias[self._row_slice] = values.reshape(-1)
        else:
            bias[self.rows] = values

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        trivial = self.out_rep.group.trivial_representation
        self.basis = _intertwiner_basis(self.out_rep, trivial)[:, :, 0].to(
            device=self.coefficients.device, dtype=self.coefficients.dtype
        )
        return self


class Linear(nn.Module):
    """A complete learnable equivariant map between finite-group fields.

    Parameters are stored in a minimal intertwiner basis. The dense kernel is
    materialized only for the matrix multiply, keeping persistent memory no
    larger than a conventional dense layer.
    """

    def __init__(
        self,
        in_type: FieldType | Representation,
        out_type: FieldType | Representation,
        bias: bool = True,
        initialize: bool = True,
        *,
        backend: str = "auto",
    ):
        super().__init__()
        if backend not in {"auto", "structured", "generic"}:
            raise ValueError("backend must be 'auto', 'structured', or 'generic'")
        self.backend = "structured" if backend == "auto" else backend
        self._raw_tensor_api = isinstance(in_type, Representation) and isinstance(out_type, Representation)
        if isinstance(in_type, Representation):
            in_type = FieldType(no_base_space(in_type.group), [in_type])
        if isinstance(out_type, Representation):
            out_type = FieldType(no_base_space(out_type.group), [out_type])
        if in_type.fibergroup is not out_type.fibergroup:
            raise ValueError("input and output FieldTypes must use the same group instance")
        self.in_type = in_type
        self.out_type = out_type
        self.space = in_type.gspace
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

        pair_occurrences: dict[tuple[Representation, Representation], tuple[list[int], list[int]]] = {}
        for out_rep, row in zip(out_type, out_type.fields_start):
            for in_rep, column in zip(in_type, in_type.fields_start):
                key = (out_rep, in_rep)
                dimension = (
                    generic_intertwiner_basis(in_rep, out_rep).shape[0]
                    if self.backend == "generic"
                    else _intertwiner_dimension(*key)
                )
                if dimension:
                    if key not in pair_occurrences:
                        pair_occurrences[key] = ([], [])
                    pair_occurrences[key][0].append(row)
                    pair_occurrences[key][1].append(column)
        self._pairs = nn.ModuleList(
            _PairExpansion(out_rep, in_rep, rows, columns, self.backend)
            for (out_rep, in_rep), (rows, columns) in pair_occurrences.items()
        )

        bias_occurrences: dict[Representation, list[int]] = defaultdict(list)
        if bias:
            for out_rep, row in zip(out_type, out_type.fields_start):
                if _intertwiner_dimension(out_rep, out_rep.group.trivial_representation):
                    bias_occurrences[out_rep].append(row)
        self._biases = nn.ModuleList(_BiasExpansion(rep, rows) for rep, rows in bias_occurrences.items())
        self.bias = bool(bias)
        if initialize:
            self.reset_parameters()

    @property
    def weights(self):
        return tuple(pair.coefficients for pair in self._pairs)

    @property
    def bias_parameters(self):
        return tuple(item.coefficients for item in self._biases)

    def reset_parameters(self) -> None:
        for pair in self._pairs:
            pair.reset_parameters(self.in_type.size, self.out_type.size)
        bound = 1.0 / math.sqrt(self.in_type.size)
        for bias in self._biases:
            bias.reset_parameters(bound)

    def expand_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        reference = self._anchor
        weight = reference.new_zeros(self.out_type.size, self.in_type.size)
        for pair in self._pairs:
            pair.write_into(weight)
        bias_tensor = reference.new_zeros(self.out_type.size) if self.bias else None
        if bias_tensor is not None:
            for bias in self._biases:
                bias.write_into(bias_tensor)
        return weight, bias_tensor

    def forward(self, input: GeometricTensor | torch.Tensor) -> GeometricTensor | torch.Tensor:
        raw = isinstance(input, torch.Tensor)
        if raw:
            if not self._raw_tensor_api:
                raise TypeError("raw tensor input is only accepted when Linear was built from Representations")
            input = GeometricTensor(input, self.in_type)
        if not isinstance(input, GeometricTensor) or input.type != self.in_type:
            raise TypeError(f"Linear expected a GeometricTensor of type {self.in_type!r}")
        weight, bias = self.expand_parameters()
        output = GeometricTensor(F.linear(input.tensor, weight, bias), self.out_type)
        return output.tensor if raw else output

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if input_shape[-1] != self.in_type.size:
            raise ValueError(f"expected last dimension {self.in_type.size}")
        return (*input_shape[:-1], self.out_type.size)

    @torch.no_grad()
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> list[tuple[object, float]]:
        x = torch.randn(4, self.in_type.size, device=self._anchor.device, dtype=self._anchor.dtype)
        input = GeometricTensor(x, self.in_type)
        output = self(input)
        errors = []
        for element in self.space.fibergroup.testing_elements:
            transformed = self(input.transform_fibers(element)).tensor
            expected = output.transform_fibers(element).tensor
            error = float((transformed - expected).abs().max())
            if not torch.allclose(transformed, expected, atol=atol, rtol=rtol):
                raise AssertionError(f"equivariance failed for {element}: max error {error:.3e}")
            errors.append((element, error))
        return errors

    def extra_repr(self) -> str:
        parameters = sum(p.numel() for p in self.parameters())
        return f"in={self.in_type.size}, out={self.out_type.size}, parameters={parameters}, bias={self.bias}"
