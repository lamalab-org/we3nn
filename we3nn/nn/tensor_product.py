"""e3nn-style bilinear tensor products for finite groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn

from ..clebsch_gordan import full_coupling_basis
from ..representations import Irrep, Representation
from ..intertwiner import intertwiner_basis, tensor_product_representation
from .field_type import FieldType, as_field_type
from .representation_tensor import (
    RepresentationTensor,
    unpack_tensor_product_inputs,
    wrap_if_typed,
)


@dataclass(frozen=True)
class TensorProductInstruction:
    """Select one equivariant field triple in :class:`TensorProduct`.

    ``i_in1``, ``i_in2``, and ``i_out`` index fields in the two input
    ``FieldType`` objects and the output ``FieldType``. ``coupling`` can select
    one real Hom-space direction; ``None`` keeps every direction. Only the
    e3nn-style ``"uvw"`` connection mode is currently defined. ``has_weight``
    controls whether the path consumes a learned or external reduced weight,
    and ``path_weight`` applies an additional fixed scalar normalization.

    ``path_shape`` is populated by the constructed module and reports the
    reduced-weight shape required by this instruction.
    """

    i_in1: int
    i_in2: int
    i_out: int
    coupling: int | None = None
    connection_mode: str = "uvw"
    has_weight: bool = True
    path_shape: tuple[int, ...] = ()
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


def _contiguous_field_slice(starts: tuple[int, ...], field_size: int) -> slice | None:
    if all(value == starts[0] + index * field_size for index, value in enumerate(starts)):
        return slice(starts[0], starts[0] + len(starts) * field_size)
    return None


def _representation_occurrences(
    field_type: FieldType,
) -> dict[Representation, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Group field indices and coordinate starts by representation identity."""
    grouped = defaultdict(lambda: ([], []))
    for field_index, (representation, start) in enumerate(
        zip(field_type, field_type.fields_start)
    ):
        grouped[representation][0].append(field_index)
        grouped[representation][1].append(start)
    return {
        representation: (tuple(indices), tuple(starts))
        for representation, (indices, starts) in grouped.items()
    }


def _legacy_weight_prefixes(
    in1_type: FieldType,
    in2_type: FieldType,
    out_type: FieldType,
    dimensions: dict[tuple[Representation, Representation, Representation], int],
) -> tuple[
    torch.Tensor,
    dict[Representation, torch.Tensor],
    dict[tuple[Representation, Representation], torch.Tensor],
]:
    """Build separable prefixes for output-left-right-coupling weight order."""
    right_counts: dict[Representation, int] = defaultdict(int)
    for right in in2_type:
        right_counts[right] += 1

    output_totals = {
        output: sum(
            right_counts[right] * dimensions.get((output, left, right), 0)
            for left in in1_type
            for right in right_counts
        )
        for output in set(out_type)
    }
    output_prefix = []
    offset = 0
    for output in out_type:
        output_prefix.append(offset)
        offset += output_totals[output]

    left_prefixes = {}
    right_prefixes = {}
    for output in set(out_type):
        prefix = []
        offset = 0
        for left in in1_type:
            prefix.append(offset)
            offset += sum(
                count * dimensions.get((output, left, right), 0)
                for right, count in right_counts.items()
            )
        left_prefixes[output] = torch.tensor(prefix, dtype=torch.long)

        for left in set(in1_type):
            prefix = []
            offset = 0
            for right in in2_type:
                prefix.append(offset)
                offset += dimensions.get((output, left, right), 0)
            right_prefixes[(output, left)] = torch.tensor(prefix, dtype=torch.long)

    return (
        torch.tensor(output_prefix, dtype=torch.long),
        left_prefixes,
        right_prefixes,
    )


class _PackedOccurrences:
    """Pack equal-representation fields with a slice or one indexed gather."""

    def __init__(self, starts: tuple[int, ...], representation: Representation):
        self.size = representation.size
        self.multiplicity = len(starts)
        self.contiguous_slice = _contiguous_field_slice(starts, representation.size)
        self.indices = torch.tensor(starts)[:, None] + torch.arange(representation.size)[None, :]

    def pack(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if self.contiguous_slice is not None:
            value = tensor[..., self.contiguous_slice]
            return value.reshape(*tensor.shape[:-1], self.multiplicity, self.size)
        return tensor[..., indices]


class _LazyFullyConnectedInstructions(Sequence[TensorProductInstruction]):
    """Expose legacy logical instructions without allocating them eagerly."""

    def __init__(
        self,
        in1_type: FieldType,
        in2_type: FieldType,
        out_type: FieldType,
        length: int,
    ):
        self.in1_type = in1_type
        self.in2_type = in2_type
        self.out_type = out_type
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[TensorProductInstruction]:
        for i_out, output in enumerate(self.out_type):
            for i_in1, left in enumerate(self.in1_type):
                for i_in2, right in enumerate(self.in2_type):
                    dimension = _coupling_dimension(left, right, output)
                    if dimension:
                        regular = output.group.regular_repr
                        if output is regular:
                            path_shape = (left.size, right.size)
                        elif left is regular:
                            path_shape = (output.size, right.size)
                        elif right is regular:
                            path_shape = (output.size, left.size)
                        else:
                            path_shape = (dimension,)
                        yield TensorProductInstruction(
                            i_in1, i_in2, i_out, path_shape=path_shape
                        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for current, instruction in enumerate(self):
            if current == index:
                return instruction
        raise IndexError(index)


class _MultiplicityTensorProductBlock(nn.Module):
    """One representation-theoretic coupling with explicit multiplicity axes."""

    def __init__(
        self,
        left: Representation,
        right: Representation,
        output: Representation,
        left_starts: tuple[int, ...],
        right_starts: tuple[int, ...],
        output_starts: tuple[int, ...],
        *,
        internal_weights: bool,
        legacy_weight_indices: torch.Tensor,
    ):
        super().__init__()
        self.left = left
        self.right = right
        self.output = output
        self.group = output.group
        self.left_pack = _PackedOccurrences(left_starts, left)
        self.right_pack = _PackedOccurrences(right_starts, right)
        self.output_pack = _PackedOccurrences(output_starts, output)
        self.register_buffer("left_indices", self.left_pack.indices, persistent=False)
        self.register_buffer("right_indices", self.right_pack.indices, persistent=False)
        self.register_buffer("output_indices", self.output_pack.indices, persistent=False)

        regular = self.group.regular_repr
        if output is regular:
            self.kind = "output_regular"
            coupling_shape = (left.size, right.size)
        elif left is regular:
            self.kind = "left_regular"
            coupling_shape = (output.size, right.size)
        elif right is regular:
            self.kind = "right_regular"
            coupling_shape = (output.size, left.size)
        else:
            self.kind = "cg"
            coupling_basis = _generic_couplings(left, right, output)
            coupling_shape = (coupling_basis.shape[0],)

        self.multiplicity_shape = (
            self.output_pack.multiplicity,
            self.left_pack.multiplicity,
            self.right_pack.multiplicity,
        )
        self.coupling_shape = tuple(coupling_shape)
        self.weight_shape = (*self.multiplicity_shape, *self.coupling_shape)
        self.weight_numel = math.prod(self.weight_shape)
        if internal_weights:
            self.weight = nn.Parameter(torch.empty(self.weight_shape))
        else:
            self.register_parameter("weight", None)

        if self.kind == "cg":
            self.register_buffer("coupling_basis", coupling_basis, persistent=True)
        else:
            self.register_buffer("coupling_basis", torch.empty(0, dtype=torch.float64), persistent=False)
        need_left = self.kind in {"output_regular", "right_regular"} and left is not regular
        need_right = self.kind in {"output_regular", "left_regular"} and right is not regular
        need_output = self.kind in {"left_regular", "right_regular"}
        stack = lambda rep, needed: (
            torch.stack([rep(element) for element in self.group.elements])
            if needed else torch.empty(0, dtype=torch.float64)
        )
        self.register_buffer("left_matrices", stack(left, need_left), persistent=False)
        self.register_buffer("right_matrices", stack(right, need_right), persistent=False)
        self.register_buffer("output_matrices", stack(output, need_output), persistent=False)
        need_indices = self.kind != "cg" and (
            (self.kind == "output_regular" and (left is regular or right is regular))
            or (self.kind == "left_regular" and right is regular)
            or (self.kind == "right_regular" and left is regular)
        )
        self.register_buffer(
            "regular_indices",
            _regular_indices(self.group) if need_indices else torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        flat_indices = legacy_weight_indices.reshape(-1)
        if flat_indices.numel() and torch.equal(
            flat_indices,
            torch.arange(flat_indices[0], flat_indices[0] + flat_indices.numel()),
        ):
            self.legacy_weight_slice = slice(int(flat_indices[0]), int(flat_indices[-1]) + 1)
            self.register_buffer("legacy_weight_indices", torch.empty(0, dtype=torch.long), persistent=False)
        else:
            self.legacy_weight_slice = None
            self.register_buffer("legacy_weight_indices", flat_indices, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            bound = math.sqrt(6.0 / (self.left.size + self.right.size + self.output.size))
            nn.init.uniform_(self.weight, -bound, bound)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        reference = self.weight if self.weight is not None else self.coupling_basis
        device, dtype = reference.device, reference.dtype
        for name, rep in (
            ("left_matrices", self.left),
            ("right_matrices", self.right),
            ("output_matrices", self.output),
        ):
            if getattr(self, name).numel():
                setattr(
                    self,
                    name,
                    torch.stack([rep(g) for g in self.group.elements]).to(device=device, dtype=dtype),
                )
        if self.kind == "cg":
            self.coupling_basis = _generic_couplings(self.left, self.right, self.output).to(
                device=device, dtype=dtype
            )
        return self

    def pack_left(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.left_pack.pack(tensor, self.left_indices)

    def pack_right(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.right_pack.pack(tensor, self.right_indices)

    def external_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.legacy_weight_slice is not None:
            selected = weight[..., self.legacy_weight_slice]
        else:
            selected = weight[..., self.legacy_weight_indices]
        return selected.reshape(*weight.shape[:-1], *self.weight_shape)

    def _matrices(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        return getattr(self, name).to(device=reference.device, dtype=reference.dtype)

    def _inverse_transforms(
        self, value: torch.Tensor, representation: Representation, matrices_name: str
    ) -> torch.Tensor:
        if representation is self.group.regular_repr:
            return value[..., self.regular_indices]
        matrices = self._matrices(matrices_name, value)
        return torch.einsum("...ui,qia->...uqa", value, matrices)

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        left = self.pack_left(left)
        right = self.pack_right(right)
        coefficients = self.weight if weight is None else weight
        if coefficients is None:
            raise RuntimeError("external tensor-product weights were not supplied")
        shared = coefficients.ndim == len(self.weight_shape)

        if self.kind == "cg":
            basis = self.coupling_basis.to(device=left.device, dtype=left.dtype)
            coupled = torch.einsum("poij,...ui,...vj->...uvpo", basis, left, right)
            equation = "muvp,...uvpo->...mo" if shared else "...muvp,...uvpo->...mo"
            return torch.einsum(equation, coefficients, coupled)

        order_scale = math.sqrt(self.group.order())
        if self.kind == "output_regular":
            left_transformed = self._inverse_transforms(left, self.left, "left_matrices")
            right_transformed = self._inverse_transforms(right, self.right, "right_matrices")
            equation = (
                "muvab,...uqa,...vqb->...mq"
                if shared else "...muvab,...uqa,...vqb->...mq"
            )
            return torch.einsum(equation, coefficients, left_transformed, right_transformed) / order_scale

        output_matrices = self._matrices("output_matrices", left)
        if self.kind == "left_regular":
            right_transformed = self._inverse_transforms(right, self.right, "right_matrices")
            equation = (
                "muvab,...vqb,...uq->...mqa"
                if shared else "...muvab,...vqb,...uq->...mqa"
            )
            intermediate = torch.einsum(equation, coefficients, right_transformed, left)
        else:
            left_transformed = self._inverse_transforms(left, self.left, "left_matrices")
            equation = (
                "muvab,...uqb,...vq->...mqa"
                if shared else "...muvab,...uqb,...vq->...mqa"
            )
            intermediate = torch.einsum(equation, coefficients, left_transformed, right)
        return torch.einsum("qoa,...mqa->...mo", output_matrices, intermediate) / order_scale

    def add_to_output(self, output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        flat_indices = self.output_indices.reshape(-1)
        flat_value = value.reshape(*value.shape[:-2], -1)
        return output.index_add(-1, flat_indices, flat_value)

    def kernel_basis_entries(
        self, filter_features: torch.Tensor, input_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return vectorized sparse entries for all weights in this block."""
        right = self.pack_right(filter_features)
        leading_shape = right.shape[:-2]
        output_multiplicity, left_multiplicity, right_multiplicity = self.multiplicity_shape
        coupling_numel = math.prod(self.coupling_shape)

        if self.kind == "cg":
            basis = self.coupling_basis.to(device=right.device, dtype=right.dtype)
            base = torch.einsum("poij,...vj->...vpoi", basis, right)
        elif self.kind == "output_regular":
            right_transformed = self._inverse_transforms(right, self.right, "right_matrices")
            if self.left is self.group.regular_repr:
                left_transform = torch.nn.functional.one_hot(
                    self.regular_indices,
                    num_classes=self.left.size,
                ).to(device=right.device, dtype=right.dtype)
            else:
                left_transform = self._matrices("left_matrices", right).transpose(-1, -2)
            base = torch.einsum(
                "qai,...vqb->...vabqi", left_transform, right_transformed
            ) / math.sqrt(self.group.order())
        elif self.kind == "left_regular":
            right_transformed = self._inverse_transforms(right, self.right, "right_matrices")
            output_matrices = self._matrices("output_matrices", right)
            base = torch.einsum(
                "qoa,...vqb->...vaboq", output_matrices, right_transformed
            ) / math.sqrt(self.group.order())
        else:
            output_matrices = self._matrices("output_matrices", right)
            left_matrices = self._matrices("left_matrices", right).transpose(-1, -2)
            base = torch.einsum(
                "qoa,qbi,...vq->...vaboi", output_matrices, left_matrices, right
            ) / math.sqrt(self.group.order())

        base = base.reshape(
            *leading_shape,
            right_multiplicity,
            coupling_numel,
            self.output.size,
            self.left.size,
        )
        values = base.unsqueeze(-5).unsqueeze(-5).expand(
            *leading_shape,
            output_multiplicity,
            left_multiplicity,
            right_multiplicity,
            coupling_numel,
            self.output.size,
            self.left.size,
        )
        values = values.reshape(
            *leading_shape,
            self.weight_numel,
            self.output.size * self.left.size,
        )

        if self.legacy_weight_slice is not None:
            weight_indices = torch.arange(
                self.legacy_weight_slice.start,
                self.legacy_weight_slice.stop,
                device=right.device,
            )
        else:
            weight_indices = self.legacy_weight_indices.to(device=right.device)

        output_coordinates = self.output_indices.to(device=right.device)
        input_coordinates = self.left_indices.to(device=right.device)
        matrix_indices = (
            output_coordinates[:, None, None, None, :, None] * input_size
            + input_coordinates[None, :, None, None, None, :]
        ).expand(
            output_multiplicity,
            left_multiplicity,
            right_multiplicity,
            coupling_numel,
            self.output.size,
            self.left.size,
        )
        matrix_indices = matrix_indices.reshape(
            self.weight_numel, self.output.size * self.left.size
        )
        return weight_indices, matrix_indices, values


class _TensorProductPath(nn.Module):
    def __init__(
        self,
        left: Representation,
        right: Representation,
        output: Representation,
        *,
        internal_weights: bool,
        path_weight: float,
        coupling: int | None = None,
        has_weight: bool = True,
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
            coupling_basis = _generic_couplings(left, right, output)
            self.weight_shape = (coupling_basis.shape[0],)
        if self.kind == "cg":
            self.register_buffer("coupling_basis", coupling_basis, persistent=True)
        else:
            self.register_buffer("coupling_basis", torch.empty(0, dtype=torch.float64), persistent=False)
        need_left = self.kind in {"output_regular", "right_regular"} and left is not regular
        need_right = self.kind in {"output_regular", "left_regular"} and right is not regular
        need_output = self.kind in {"left_regular", "right_regular"}
        stack = lambda rep, needed: (
            torch.stack([rep(element) for element in self.group.elements])
            if needed
            else torch.empty(0, dtype=torch.float64)
        )
        self.register_buffer("left_matrices", stack(left, need_left), persistent=False)
        self.register_buffer("right_matrices", stack(right, need_right), persistent=False)
        self.register_buffer("output_matrices", stack(output, need_output), persistent=False)
        need_indices = self.kind != "cg" and (
            (self.kind == "output_regular" and (left is regular or right is regular))
            or (self.kind == "left_regular" and right is regular)
            or (self.kind == "right_regular" and left is regular)
        )
        self.register_buffer(
            "regular_indices",
            _regular_indices(self.group) if need_indices else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.full_weight_shape = self.weight_shape
        self.coupling = coupling
        self.has_weight = bool(has_weight)
        full_numel = math.prod(self.full_weight_shape)
        if full_numel == 0:
            raise ValueError("the requested tensor-product instruction has no equivariant coupling")
        if coupling is not None:
            if not 0 <= int(coupling) < full_numel:
                raise ValueError(f"coupling index {coupling} is outside [0, {full_numel})")
            self.coupling = int(coupling)
            self.weight_shape = (1,)
        if not self.has_weight and self.coupling is None and full_numel != 1:
            raise ValueError("an unweighted instruction must select one coupling")
        if internal_weights and self.has_weight:
            self.weight = nn.Parameter(torch.empty(self.weight_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    @property
    def weight_numel(self) -> int:
        return math.prod(self.weight_shape) if self.has_weight else 0

    def reset_parameters(self) -> None:
        if self.weight is not None:
            bound = math.sqrt(6.0 / (self.left.size + self.right.size + self.output.size))
            nn.init.uniform_(self.weight, -bound, bound)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        reference = self.weight if self.weight is not None else self.coupling_basis
        device, dtype = reference.device, reference.dtype
        for name, rep in (("left_matrices", self.left), ("right_matrices", self.right), ("output_matrices", self.output)):
            current = getattr(self, name)
            if current.numel():
                setattr(self, name, torch.stack([rep(g) for g in self.group.elements]).to(device=device, dtype=dtype))
        if self.kind == "cg":
            self.coupling_basis = _generic_couplings(self.left, self.right, self.output).to(device=device, dtype=dtype)
        return self

    def _matrices(self, representation: Representation, reference: torch.Tensor) -> torch.Tensor:
        if representation is self.left:
            matrices = self.left_matrices
        elif representation is self.right:
            matrices = self.right_matrices
        elif representation is self.output:
            matrices = self.output_matrices
        else:
            raise RuntimeError("unknown tensor-product representation")
        return matrices.to(device=reference.device, dtype=reference.dtype)

    def _inverse_transforms(self, value: torch.Tensor, representation: Representation) -> torch.Tensor:
        if representation is self.group.regular_repr:
            indices = self.regular_indices
            return value[..., indices]
        matrices = self._matrices(representation, value)
        return torch.einsum("...i,qia->...qa", value, matrices)

    def forward(self, left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        coefficients = self.weight if weight is None else weight
        if not self.has_weight:
            coefficients = left.new_ones((1,))
        elif coefficients is None:
            raise RuntimeError("external tensor-product weights were not supplied")
        shared = coefficients.ndim == len(self.weight_shape)
        if self.coupling is not None:
            expanded_shape = (*coefficients.shape[:-1], *self.full_weight_shape)
            expanded = coefficients.new_zeros(expanded_shape).flatten(-len(self.full_weight_shape))
            expanded[..., self.coupling] = coefficients[..., 0]
            coefficients = expanded.reshape(expanded_shape)
        scale = self.path_weight
        if self.kind == "cg":
            basis = self.coupling_basis.to(
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
    r"""General learnable finite-group equivariant tensor product.

    For representations :math:`V_1`, :math:`V_2`, and :math:`V_o`, each path
    uses a basis tensor

    .. math::

        C_p \in \operatorname{Hom}_G(V_1 \otimes V_2, V_o)

    and evaluates

    .. math::

        z_o = \sum_p w_p (C_p)_{oij} x_i y_j.

    By default, every compatible triple of input/output fields is included.
    ``instructions`` may instead contain ``(i_in1, i_in2, i_out)`` triples or
    :class:`TensorProductInstruction` objects. Each instruction spans every
    independent real equivariant coupling for that field triple.

    The fully connected default groups repeated copies of each unique
    representation triple into one execution block. A block stores weights as
    ``(out_multiplicity, left_multiplicity, right_multiplicity, *coupling_shape)``
    and performs one or a few batched contractions. This changes neither the
    number nor the ordering of logical reduced weights: every ordered field
    triple remains independently learnable, and flattened external weights
    retain the legacy ``output, left, right, coupling`` order. Explicit
    instructions continue to use the sparse per-path executor.

    For a grouped default, :attr:`instructions` is a lazy sequence: ``len()``
    reports the logical field-path count without allocating that many Python
    objects. :attr:`blocks` contains the compact execution modules, while
    :attr:`paths` is empty. Explicit products expose their legacy modules in
    :attr:`paths` and have no grouped blocks.

    With ``internal_weights=False``, pass a final-axis weight tensor to
    :meth:`forward`. Shared weights have shape ``(weight_numel,)``; with
    ``shared_weights=False`` they have shape ``(..., weight_numel)`` matching
    the inputs' leading dimensions.

    Args:
        in1_type: Representation carried by the first input's final axis.
        in2_type: Representation carried by the second input's final axis.
        out_type: Requested direct sum of output representations.
        instructions: Optional subset and configuration of coupling paths.
        internal_weights: Store reduced weights as module parameters.
        shared_weights: Share one external weight vector over leading axes.

    Input safety:
        Raw ``torch.Tensor`` inputs remain supported but emit
        :class:`MissingRepresentationMetadataWarning`. Wrap both inputs with
        :class:`RepresentationTensor` to validate their representations and
        receive a typed output. Incorrect metadata raises ``TypeError`` even
        when tensor dimensions happen to match.

    Choose this class when path selection or internal/external weight control
    is required. Use :class:`FullyConnectedTensorProduct` to emphasize the
    all-path default or :class:`FullTensorProduct` for an unprojected Kronecker
    product.

    Checkpoints written by the former per-path fully connected implementation
    are migrated automatically when loaded into a matching grouped module.
    New grouped checkpoints use ``blocks.*.weight`` keys; old software which
    only understands ``paths.*.weight`` cannot load them without upgrading.
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
        if isinstance(in1_type, Representation):
            in1_type = as_field_type(in1_type)
        if isinstance(in2_type, Representation):
            in2_type = as_field_type(in2_type)
        if isinstance(out_type, Representation):
            out_type = as_field_type(out_type)
        if in1_type.fibergroup is not in2_type.fibergroup or in1_type.fibergroup is not out_type.fibergroup:
            raise ValueError("all FieldTypes must use the same group instance")
        if internal_weights and not shared_weights:
            raise ValueError("internal tensor-product weights must be shared")
        self.in1_type = in1_type
        self.in2_type = in2_type
        self.out_type = out_type
        self.internal_weights = bool(internal_weights)
        self.shared_weights = bool(shared_weights)

        if instructions is None:
            left_occurrences = _representation_occurrences(in1_type)
            right_occurrences = _representation_occurrences(in2_type)
            output_occurrences = _representation_occurrences(out_type)
            coupling_dimensions = {}
            block_specs = []
            for output, (output_fields, output_starts) in output_occurrences.items():
                for left, (left_fields, left_starts) in left_occurrences.items():
                    for right, (right_fields, right_starts) in right_occurrences.items():
                        dimension = _coupling_dimension(left, right, output)
                        coupling_dimensions[(output, left, right)] = dimension
                        if not dimension:
                            continue
                        block_specs.append(
                            (
                                left,
                                right,
                                output,
                                left_fields,
                                left_starts,
                                right_fields,
                                right_starts,
                                output_fields,
                                output_starts,
                                dimension,
                            )
                        )

            output_prefix, left_prefixes, right_prefixes = _legacy_weight_prefixes(
                in1_type, in2_type, out_type, coupling_dimensions
            )
            blocks = []
            weight_numel = 0
            logical_paths = 0
            for (
                left,
                right,
                output,
                left_fields,
                left_starts,
                right_fields,
                right_starts,
                output_fields,
                output_starts,
                dimension,
            ) in block_specs:
                output_offsets = output_prefix[torch.tensor(output_fields)]
                left_offsets = left_prefixes[output][torch.tensor(left_fields)]
                right_offsets = right_prefixes[(output, left)][torch.tensor(right_fields)]
                legacy_indices = (
                    output_offsets[:, None, None, None]
                    + left_offsets[None, :, None, None]
                    + right_offsets[None, None, :, None]
                    + torch.arange(dimension)[None, None, None, :]
                )
                blocks.append(
                    _MultiplicityTensorProductBlock(
                        left,
                        right,
                        output,
                        left_starts,
                        right_starts,
                        output_starts,
                        internal_weights=internal_weights,
                        legacy_weight_indices=legacy_indices,
                    )
                )
                multiplicity_paths = (
                    len(output_fields) * len(left_fields) * len(right_fields)
                )
                logical_paths += multiplicity_paths
                weight_numel += multiplicity_paths * dimension
            self.blocks = nn.ModuleList(blocks)
            self.paths = nn.ModuleList()
            self.instructions = _LazyFullyConnectedInstructions(
                in1_type, in2_type, out_type, logical_paths
            )
            self.weight_numel = weight_numel
            self._grouped = True
            return

        requested = []
        for instruction in instructions:
            if isinstance(instruction, TensorProductInstruction):
                if instruction.connection_mode != "uvw":
                    raise ValueError("only connection_mode='uvw' is currently supported")
                requested.append(
                    (
                        instruction.i_in1,
                        instruction.i_in2,
                        instruction.i_out,
                        instruction.coupling,
                        instruction.connection_mode,
                        instruction.has_weight,
                        instruction.path_weight,
                    )
                )
            else:
                if len(instruction) != 3:
                    raise ValueError("instructions must be (i_in1, i_in2, i_out) triples")
                requested.append((*map(int, instruction), None, "uvw", True, 1.0))

        paths = []
        normalized_instructions = []
        for i_in1, i_in2, i_out, coupling, connection_mode, has_weight, path_weight in requested:
            left, right, output = in1_type[i_in1], in2_type[i_in2], out_type[i_out]
            path = _TensorProductPath(
                left,
                right,
                output,
                internal_weights=internal_weights,
                path_weight=path_weight,
                coupling=coupling,
                has_weight=has_weight,
            )
            paths.append(path)
            normalized_instructions.append(
                TensorProductInstruction(
                    i_in1,
                    i_in2,
                    i_out,
                    coupling=coupling,
                    connection_mode=connection_mode,
                    has_weight=has_weight,
                    path_shape=path.weight_shape,
                    path_weight=path_weight,
                )
            )
        self.paths = nn.ModuleList(paths)
        self.blocks = nn.ModuleList()
        self.instructions = tuple(normalized_instructions)
        self.weight_numel = sum(path.weight_numel for path in self.paths)
        self._grouped = False

    def forward(
        self,
        input1: torch.Tensor | RepresentationTensor,
        input2: torch.Tensor | RepresentationTensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor | RepresentationTensor:
        (input1, input2), typed = unpack_tensor_product_inputs(
            (
                ("input1", input1, self.in1_type),
                ("input2", input2, self.in2_type),
            ),
            type(self).__name__,
        )
        if input1.shape[-1] != self.in1_type.size or input2.shape[-1] != self.in2_type.size:
            raise ValueError("tensor-product input feature dimensions do not match the representations")
        if input1.shape[:-1] != input2.shape[:-1]:
            raise ValueError("tensor-product inputs must have matching leading dimensions")
        if self.internal_weights:
            if weight is not None:
                raise ValueError("an internally weighted TensorProduct does not accept external weights")
        else:
            expected = (self.weight_numel,) if self.shared_weights else (*input1.shape[:-1], self.weight_numel)
            if self.weight_numel == 0 and weight is None:
                pass
            elif weight is None or tuple(weight.shape) != expected:
                raise ValueError(f"external weight must have shape {expected}")

        output = input1.new_zeros(*input1.shape[:-1], self.out_type.size)
        if self._grouped:
            for block in self.blocks:
                external = None
                if not self.internal_weights:
                    external = block.external_weight(weight)
                output = block.add_to_output(output, block(input1, input2, external))
            return wrap_if_typed(output, self.out_type, typed)

        weight_offset = 0
        for instruction, path in zip(self.instructions, self.paths):
            left = input1[..., self.in1_type.fields_start[instruction.i_in1]:self.in1_type.fields_end[instruction.i_in1]]
            right = input2[..., self.in2_type.fields_start[instruction.i_in2]:self.in2_type.fields_end[instruction.i_in2]]
            external = None
            if not self.internal_weights and path.has_weight:
                external = weight[..., weight_offset:weight_offset + path.weight_numel].reshape(
                    *weight.shape[:-1], *path.weight_shape
                )
                weight_offset += path.weight_numel
            value = path(left, right, external)
            start, end = self.out_type.fields_start[instruction.i_out], self.out_type.fields_end[instruction.i_out]
            output[..., start:end] = output[..., start:end] + value
        return wrap_if_typed(output, self.out_type, typed)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate legacy fully-connected per-path state dictionaries."""
        if self._grouped:
            legacy_prefix = f"{prefix}paths."
            legacy_keys = [key for key in state_dict if key.startswith(legacy_prefix)]
            if legacy_keys:
                legacy_weights = []
                path_index = 0
                while True:
                    key = f"{legacy_prefix}{path_index}.weight"
                    if key not in state_dict:
                        break
                    legacy_weights.append(state_dict[key].reshape(-1))
                    path_index += 1
                flat_weights = torch.cat(legacy_weights) if legacy_weights else None
                for block_index, block in enumerate(self.blocks):
                    if block.weight is not None and flat_weights is not None:
                        if block.legacy_weight_slice is not None:
                            selected = flat_weights[block.legacy_weight_slice]
                        else:
                            selected = flat_weights[block.legacy_weight_indices]
                        state_dict[f"{prefix}blocks.{block_index}.weight"] = selected.reshape(
                            block.weight_shape
                        )
                    if block.coupling_basis.numel():
                        state_dict[f"{prefix}blocks.{block_index}.coupling_basis"] = (
                            block.coupling_basis
                        )
                for key in legacy_keys:
                    del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        left = torch.randn(*shape, self.in1_type.size, device=reference.device, dtype=reference.dtype)
        right = torch.randn(*shape, self.in2_type.size, device=reference.device, dtype=reference.dtype)
        external = None
        if not self.internal_weights:
            weight_shape = (self.weight_numel,) if self.shared_weights else (*shape, self.weight_numel)
            external = torch.randn(*weight_shape, device=reference.device, dtype=reference.dtype)
        output = self(
            RepresentationTensor(left, self.in1_type),
            RepresentationTensor(right, self.in2_type),
            external,
        ).tensor
        errors = []
        for element in self.in1_type.fibergroup.elements:
            actual = self(
                RepresentationTensor(
                    self.in1_type.transform_fibers(left, element), self.in1_type
                ),
                RepresentationTensor(
                    self.in2_type.transform_fibers(right, element), self.in2_type
                ),
                external,
            ).tensor
            expected = self.out_type.transform_fibers(output, element)
            error = float((actual - expected).abs().max())
            if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
                raise AssertionError(f"tensor-product equivariance failed for {element}: {error:.3e}")
            errors.append((element, error))
        return errors


class FullyConnectedTensorProduct(TensorProduct):
    """Connect every compatible input-field pair to every requested output.

    This is the fully connected specialization of :class:`TensorProduct`.
    Its constructor and forward arguments are identical; omitting
    ``instructions`` enumerates every nonzero real equivariant Hom-space path.
    Reduced weights can be internal, shared external, or per-sample external.
    Repeated representation copies are executed as multiplicity axes, so the
    module count scales with unique representation triples rather than the
    Cartesian product of field multiplicities.

    Raw inputs emit :class:`MissingRepresentationMetadataWarning`. Two
    correctly typed :class:`RepresentationTensor` inputs produce a typed
    output; mismatched metadata raises ``TypeError``.
    """


class FullTensorProduct(nn.Module):
    r"""Return the unweighted direct product in the product coordinate basis.

    This module computes

    .. math::

        (x \otimes y)_{ij} = x_i y_j

    and flattens ``(i, j)`` into the final axis. It performs no irrep
    decomposition, no projection onto a requested output, and has no learned
    parameters. ``out_type`` is the tensor-product representation
    :math:`\rho_1(g) \otimes \rho_2(g)` in that same flattened basis.

    Use this class when downstream code needs the complete product feature
    space. Use :class:`TensorProduct` when a decomposed/output-selected and
    weighted equivariant map is desired.

    Raw inputs emit :class:`MissingRepresentationMetadataWarning`; two typed
    inputs produce a typed product and representation mismatches are errors.
    """

    def __init__(self, in1_type: FieldType | Representation, in2_type: FieldType | Representation):
        super().__init__()
        if isinstance(in1_type, Representation):
            in1_type = as_field_type(in1_type)
        if isinstance(in2_type, Representation):
            in2_type = as_field_type(in2_type)
        if in1_type.fibergroup is not in2_type.fibergroup:
            raise ValueError("input types must use the same group")
        self.in1_type = in1_type
        self.in2_type = in2_type
        self.out_type = FieldType(
            in1_type.gspace,
            [tensor_product_representation(in1_type.representation, in2_type.representation)],
        )

    def forward(
        self,
        input1: torch.Tensor | RepresentationTensor,
        input2: torch.Tensor | RepresentationTensor,
    ) -> torch.Tensor | RepresentationTensor:
        (input1, input2), typed = unpack_tensor_product_inputs(
            (
                ("input1", input1, self.in1_type),
                ("input2", input2, self.in2_type),
            ),
            type(self).__name__,
        )
        if input1.shape[-1] != self.in1_type.size or input2.shape[-1] != self.in2_type.size:
            raise ValueError("FullTensorProduct feature dimensions do not match")
        if input1.shape[:-1] != input2.shape[:-1]:
            raise ValueError("input leading dimensions must match")
        output = torch.einsum("...i,...j->...ij", input1, input2).flatten(-2)
        return wrap_if_typed(output, self.out_type, typed)


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
