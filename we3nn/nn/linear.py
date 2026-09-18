"""Finite-group equivariant linear maps on ordinary PyTorch tensors."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..representations import Representation
from ..intertwiner import intertwiner_basis as generic_intertwiner_basis
from .field_type import FieldType, as_field_type
from .representation_tensor import (
    RepresentationTensor,
    unpack_representation_tensor,
    wrap_if_typed,
)


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
        trivial = out_rep.group.trivial_representation
        regular = out_rep.group.regular_repr
        if backend != "generic" and out_rep is trivial and in_rep is trivial:
            self.direct_kind = "trivial_trivial"
        elif backend != "generic" and out_rep is regular and in_rep is trivial:
            self.direct_kind = "trivial_regular"
        elif backend != "generic" and out_rep is trivial and in_rep is regular:
            self.direct_kind = "regular_trivial"
        elif self._regular_to_regular:
            self.direct_kind = "regular_regular"
        else:
            self.direct_kind = "generic"


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

    @property
    def is_contiguous(self) -> bool:
        return (
            self._row_slice is not None
            and self._column_slice is not None
        )

    def dense_block(self) -> torch.Tensor:
        """Expand a contiguous field-pair block without an indexed write."""
        if not self.is_contiguous:
            raise RuntimeError("field-pair block is not contiguous")
        if self._regular_to_regular:
            blocks = self.coefficients[..., self.relative] / math.sqrt(self.out_rep.group.order())
        else:
            blocks = torch.einsum("rcp,poi->rcoi", self.coefficients, self.basis)
        return blocks.permute(0, 2, 1, 3).reshape(
            self.coefficients.shape[0] * self.out_size,
            self.coefficients.shape[1] * self.in_size,
        )

    def pack_input(self, input: torch.Tensor) -> torch.Tensor:
        """Pack this pair's input occurrences as ``[..., U, I]``."""
        if self._column_slice is not None:
            value = input[..., self._column_slice]
            return value.reshape(
                *input.shape[:-1], self.coefficients.shape[1], self.in_size
            )
        return input[..., self.columns]

    def direct(self, input: torch.Tensor) -> torch.Tensor:
        """Apply this reduced block without constructing its dense operator."""
        value = self.pack_input(input)
        coefficients = self.coefficients

        if self.direct_kind == "trivial_trivial":
            mixed = F.linear(value[..., :, 0], coefficients[..., 0])
            return mixed.unsqueeze(-1) * self.basis[0, 0, 0]

        if self.direct_kind == "trivial_regular":
            mixed = F.linear(value[..., :, 0], coefficients[..., 0])
            return mixed.unsqueeze(-1) * self.basis[0, :, 0]

        if self.direct_kind == "regular_trivial":
            projected = torch.einsum("...ui,i->...u", value, self.basis[0, 0])
            return F.linear(projected, coefficients[..., 0]).unsqueeze(-1)

        if self.direct_kind == "regular_regular":
            # inverse_relative[p, o] is the input coordinate i satisfying
            # relative[o, i] == p.  Derive it from the existing quadratic
            # table instead of retaining a second equally large buffer.
            inverse_relative = torch.argsort(self.relative, dim=1).T
            # shifted[..., u, p, o] = input[..., u, i(p, o)].  The following
            # contraction is the regular-representation group convolution.
            shifted = value[..., :, inverse_relative]
            return torch.einsum("...upo,vup->...vo", shifted, coefficients) / math.sqrt(
                self.out_rep.group.order()
            )

        # Pick the smaller of the two natural differentiable contraction
        # orders.  This bounds the temporary by either [..., U, P, O] or
        # [..., V, P, I] and works for structured and generic bases alike.
        out_fields, in_fields, paths = coefficients.shape
        basis_first_size = in_fields * paths * self.out_size
        coefficients_first_size = out_fields * paths * self.in_size
        if basis_first_size <= coefficients_first_size:
            coupled = torch.einsum("poi,...ui->...upo", self.basis, value)
            return torch.einsum("vup,...upo->...vo", coefficients, coupled)
        mixed = torch.einsum("vup,...ui->...vpi", coefficients, value)
        return torch.einsum("poi,...vpi->...vo", self.basis, mixed)

    def dense(self, input: torch.Tensor) -> torch.Tensor:
        """Apply only this pair through a locally expanded dense block."""
        value = self.pack_input(input).flatten(-2)
        if self._regular_to_regular:
            blocks = self.coefficients[..., self.relative] / math.sqrt(
                self.out_rep.group.order()
            )
        else:
            blocks = torch.einsum("rcp,poi->rcoi", self.coefficients, self.basis)
        operator = blocks.permute(0, 2, 1, 3).reshape(
            self.coefficients.shape[0] * self.out_size,
            self.coefficients.shape[1] * self.in_size,
        )
        return F.linear(value, operator).reshape(
            *value.shape[:-1], self.coefficients.shape[0], self.out_size
        )

    def add_to_output(self, output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Accumulate a packed ``[..., V, O]`` result into flat output fields."""
        flat_value = value.reshape(*value.shape[:-2], -1)
        if self._row_slice is not None:
            output[..., self._row_slice] = output[..., self._row_slice] + flat_value
            return output
        return output.index_add(-1, self.rows.reshape(-1), flat_value)

    def auto_uses_direct(self) -> bool:
        """Conservative deterministic auto-selection for this pair."""
        if self.direct_kind in {
            "trivial_trivial",
            "trivial_regular",
            "regular_trivial",
        }:
            return True
        if self.direct_kind == "regular_regular":
            return False
        paths = self.coefficients.shape[-1]
        dense_coordinates = self.out_size * self.in_size
        return paths * (self.out_size + self.in_size) <= dense_coordinates

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

    def dense_block(self) -> torch.Tensor:
        if self._row_slice is None:
            raise RuntimeError("bias block is not contiguous")
        return torch.einsum("cp,po->co", self.coefficients, self.basis).reshape(-1)

    def add_to_output(self, output: torch.Tensor) -> torch.Tensor:
        """Apply invariant bias without expanding the global bias vector."""
        values = torch.einsum("cp,po->co", self.coefficients, self.basis)
        if self._row_slice is not None:
            output[..., self._row_slice] = output[..., self._row_slice] + values.reshape(-1)
            return output
        flat_values = values.reshape(-1).expand(*output.shape[:-1], values.numel())
        return output.index_add(-1, self.rows.reshape(-1), flat_values)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        trivial = self.out_rep.group.trivial_representation
        self.basis = _intertwiner_basis(self.out_rep, trivial)[:, :, 0].to(
            device=self.coefficients.device, dtype=self.coefficients.dtype
        )
        return self


class WELinear(nn.Module):
    """A complete learnable equivariant map between finite-group fields.

    Parameters are stored in a minimal intertwiner basis. ``execution="dense"``
    preserves the historical global dense expansion and its versioned
    no-gradient inference cache. ``execution="direct"`` contracts inputs with
    reduced coefficients and intertwiner structure without materializing the
    global dense weight. ``execution="auto"`` selects direct channel mixing
    and small structured contractions conservatively. It falls back to the
    global dense operation when any representation pair is better served by
    dense GEMM, and uses the versioned dense cache during inference.

    ``backend`` selects how intertwiner bases are constructed; it is
    independent of the execution strategy. :meth:`expand_parameters` always
    returns the same physical dense operator for every execution strategy.
    """

    def __init__(
        self,
        in_type: FieldType | Representation,
        out_type: FieldType | Representation,
        bias: bool = True,
        initialize: bool = True,
        *,
        backend: str = "auto",
        execution: str = "auto",
    ):
        super().__init__()
        if backend not in {"auto", "structured", "generic"}:
            raise ValueError("backend must be 'auto', 'structured', or 'generic'")
        if execution not in {"auto", "dense", "direct"}:
            raise ValueError("execution must be 'auto', 'dense', or 'direct'")
        self.backend = "structured" if backend == "auto" else backend
        self.execution = execution
        if isinstance(in_type, Representation):
            in_type = as_field_type(in_type)
        if isinstance(out_type, Representation):
            out_type = as_field_type(out_type)
        if in_type.fibergroup is not out_type.fibergroup:
            raise ValueError("input and output FieldTypes must use the same group instance")
        self.in_type = in_type
        self.out_type = out_type
        self.space = in_type.gspace
        self.register_buffer("_anchor", torch.empty(0), persistent=False)
        self.register_buffer("_inference_weight", torch.empty(0), persistent=False)
        self.register_buffer("_inference_bias", torch.empty(0), persistent=False)
        self._inference_versions = None

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
        tiled_pairs = sorted(self._pairs, key=lambda pair: pair._column_slice.start if pair._column_slice else -1)
        tile_end = 0
        complete_tiling = bool(tiled_pairs)
        for pair in tiled_pairs:
            complete_tiling = complete_tiling and (
                pair.is_contiguous
                and pair._row_slice.start == 0
                and pair._row_slice.stop == self.out_type.size
                and pair._column_slice.start == tile_end
            )
            if not complete_tiling:
                break
            tile_end = pair._column_slice.stop
        complete_tiling = complete_tiling and tile_end == self.in_type.size
        if complete_tiling:
            blocks = [pair.dense_block() for pair in tiled_pairs]
            weight = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)
        else:
            weight = reference.new_zeros(self.out_type.size, self.in_type.size)
            for pair in self._pairs:
                pair.write_into(weight)
        complete_bias = (
            self.bias
            and len(self._biases) == 1
            and self._biases[0]._row_slice is not None
            and self._biases[0]._row_slice.start == 0
            and self._biases[0]._row_slice.stop == self.out_type.size
        )
        bias_tensor = self._biases[0].dense_block() if complete_bias else (
            reference.new_zeros(self.out_type.size) if self.bias else None
        )
        if bias_tensor is not None:
            if not complete_bias:
                for bias in self._biases:
                    bias.write_into(bias_tensor)
        return weight, bias_tensor

    def forward(
        self,
        input: torch.Tensor | RepresentationTensor,
    ) -> torch.Tensor | RepresentationTensor:
        tensor, typed = unpack_representation_tensor(input, self.in_type, "input")
        if tensor.shape[-1] != self.in_type.size:
            raise ValueError(f"expected last dimension {self.in_type.size}, got {tensor.shape[-1]}")
        auto_dense = self.execution == "auto" and (
            not torch.is_grad_enabled()
            or not all(pair.auto_uses_direct() for pair in self._pairs)
        )
        if self.execution == "dense" or auto_dense:
            output = self._forward_dense(tensor)
        else:
            output = self._forward_structured(tensor)
        return wrap_if_typed(output, self.out_type, typed)

    def _forward_dense(self, tensor: torch.Tensor) -> torch.Tensor:
        """Historical global dense execution, including inference caching."""
        if torch.is_grad_enabled():
            weight, bias = self.expand_parameters()
        else:
            versions = tuple(parameter._version for parameter in self.parameters())
            if versions != self._inference_versions:
                weight, bias = self.expand_parameters()
                self._inference_weight = weight.detach()
                self._inference_bias = (
                    bias.detach() if bias is not None else self._anchor.new_empty(0)
                )
                self._inference_versions = versions
            weight = self._inference_weight
            bias = self._inference_bias if self.bias else None
        return F.linear(tensor, weight, bias)

    def _forward_structured(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute multiplicity-grouped representation-pair contractions."""
        output = tensor.new_zeros(*tensor.shape[:-1], self.out_type.size)
        for pair in self._pairs:
            use_direct = self.execution == "direct" or pair.auto_uses_direct()
            value = pair.direct(tensor) if use_direct else pair.dense(tensor)
            output = pair.add_to_output(output, value)
        if self.bias:
            for bias in self._biases:
                output = bias.add_to_output(output)
        return output

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._inference_weight = self._anchor.new_empty(0)
        self._inference_bias = self._anchor.new_empty(0)
        self._inference_versions = None
        return self

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if input_shape[-1] != self.in_type.size:
            raise ValueError(f"expected last dimension {self.in_type.size}")
        return (*input_shape[:-1], self.out_type.size)

    @torch.no_grad()
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> list[tuple[object, float]]:
        x = torch.randn(4, self.in_type.size, device=self._anchor.device, dtype=self._anchor.dtype)
        output = self(x)
        errors = []
        for element in self.space.fibergroup.testing_elements:
            transformed = self(self.in_type.transform_fibers(x, element))
            expected = self.out_type.transform_fibers(output, element)
            error = float((transformed - expected).abs().max())
            if not torch.allclose(transformed, expected, atol=atol, rtol=rtol):
                raise AssertionError(f"equivariance failed for {element}: max error {error:.3e}")
            errors.append((element, error))
        return errors

    def extra_repr(self) -> str:
        parameters = sum(p.numel() for p in self.parameters())
        return (
            f"in={self.in_type.size}, out={self.out_type.size}, "
            f"parameters={parameters}, bias={self.bias}, execution={self.execution!r}"
        )
