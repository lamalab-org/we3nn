"""Representation containers inspired by :mod:`e3nn.o3` irreps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .groups import FiniteGroup, GroupElement


class Representation:
    def __init__(
        self,
        group: "FiniteGroup",
        name: str,
        size: int,
        matrices: Callable[["GroupElement"], torch.Tensor],
        *,
        supported_nonlinearities: frozenset[str] = frozenset({"norm", "gated"}),
        is_permutation: bool = False,
        irreps: tuple[tuple[int, tuple[int, ...]], ...] | None = None,
        change_of_basis: torch.Tensor | None = None,
        is_irreducible: bool = False,
        is_orthogonal: bool = True,
        basis_kind: str | None = None,
    ):
        self.group = group
        self.name = name
        self.size = int(size)
        self._matrices = matrices
        self.supported_nonlinearities = supported_nonlinearities
        self.is_permutation = is_permutation
        self.irreps = irreps
        self.change_of_basis = (
            torch.eye(self.size, dtype=torch.float64) if change_of_basis is None else change_of_basis.to(torch.float64)
        )
        self.change_of_basis_inv = torch.linalg.inv(self.change_of_basis)
        self.is_irreducible = bool(is_irreducible)
        self.is_orthogonal = bool(is_orthogonal)
        self.basis_kind = basis_kind or ("permutation" if is_permutation else "arbitrary")
        self.pointwise_action = "permutation" if is_permutation else None

    @property
    def dim(self) -> int:
        return self.size

    def __call__(self, element: "GroupElement" | None = None) -> torch.Tensor | "Representation":
        if element is None:
            return self
        matrix = self._matrices(element)
        if matrix.shape != (self.size, self.size):
            raise RuntimeError(f"representation {self.name} produced matrix with shape {matrix.shape}")
        return matrix

    def matrix(self, element: "GroupElement", *, dtype=None, device=None) -> torch.Tensor:
        return self(element).to(dtype=dtype, device=device)

    def character(self, element: "GroupElement") -> float:
        return float(torch.trace(self(element)))

    def __repr__(self) -> str:
        return f"{self.group.name}:{self.name}"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def check_representation(self, exhaustive: bool = True, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
        elements = self.group.elements if exhaustive else self.group.generators
        identity = torch.eye(self.size, dtype=torch.float64)
        torch.testing.assert_close(self(self.group.identity), identity, atol=atol, rtol=rtol)
        for first in elements:
            matrix = self(first)
            torch.testing.assert_close(self(first.inverse()) @ matrix, identity, atol=atol, rtol=rtol)
            if self.is_orthogonal:
                torch.testing.assert_close(matrix.T @ matrix, identity, atol=atol, rtol=rtol)
            for second in elements:
                torch.testing.assert_close(
                    self(self.group.combine(first, second)), matrix @ self(second), atol=atol, rtol=rtol
                )
        return True

    def __add__(self, other: "Representation") -> "DirectSumRepresentation":
        return direct_sum((self, other))

    def __radd__(self, other):
        return self if other == 0 else direct_sum((other, self))

    def __mul__(self, multiplicity: int) -> "DirectSumRepresentation":
        if not isinstance(multiplicity, int) or multiplicity < 1:
            raise ValueError("representation multiplicity must be a positive integer")
        return direct_sum((self,) * multiplicity)

    def __rmul__(self, multiplicity: int) -> "DirectSumRepresentation":
        return self * multiplicity


class Irrep(Representation):
    def __init__(
        self,
        group: "FiniteGroup",
        irrep_id: tuple[int, ...],
        size: int,
        matrices: Callable[["GroupElement"], torch.Tensor],
        *,
        complex_type: bool = False,
    ):
        self.id = tuple(irrep_id)
        self.sum_of_squares_constituents = 2 if complex_type else 1
        self.irrep_type = "complex" if complex_type else "real"
        is_trivial = self.id == group.trivial_id
        nonlinearities = frozenset({"pointwise", "norm", "gated"}) if is_trivial else frozenset({"norm", "gated"})
        Representation.__init__(
            self,
            group,
            f"irrep{self.id}",
            size,
            matrices,
            supported_nonlinearities=nonlinearities,
            is_permutation=is_trivial,
            irreps=((1, self.id),),
            is_irreducible=True,
            basis_kind="irrep",
        )

    @property
    def is_trivial(self) -> bool:
        return self.id == self.group.trivial_id


@dataclass(frozen=True)
class RepBlock:
    multiplicity: int
    rep: Representation


class DirectSumRepresentation(Representation):
    def __init__(self, representations: Sequence[Representation], name: str | None = None):
        if not representations:
            raise ValueError("a direct sum needs at least one representation")
        group = representations[0].group
        if any(rep.group is not group for rep in representations):
            raise ValueError("all representations in a direct sum must belong to the same group")
        self.representations = tuple(representations)
        blocks = []
        for rep in self.representations:
            if blocks and blocks[-1].rep is rep:
                blocks[-1] = RepBlock(blocks[-1].multiplicity + 1, rep)
            else:
                blocks.append(RepBlock(1, rep))
        self.blocks = tuple(blocks)

        def matrices(element: "GroupElement") -> torch.Tensor:
            return torch.block_diag(*(rep(element) for rep in self.representations))

        irreps = []
        for rep in self.representations:
            if rep.irreps is None:
                irreps = None
                break
            irreps.extend(rep.irreps)
        super().__init__(
            group,
            name or "+".join(rep.name for rep in self.representations),
            sum(rep.size for rep in self.representations),
            matrices,
            supported_nonlinearities=frozenset.intersection(
                *(rep.supported_nonlinearities for rep in self.representations)
            ),
            is_permutation=all(rep.is_permutation for rep in self.representations),
            irreps=None if irreps is None else tuple(irreps),
            change_of_basis=torch.block_diag(*(rep.change_of_basis for rep in self.representations)),
            is_orthogonal=all(rep.is_orthogonal for rep in self.representations),
            basis_kind="permutation" if all(rep.is_permutation for rep in self.representations) else "direct_sum",
        )

    def __add__(self, other: Representation) -> "DirectSumRepresentation":
        right = other.representations if isinstance(other, DirectSumRepresentation) else (other,)
        return DirectSumRepresentation((*self.representations, *right))

    def __mul__(self, multiplicity: int) -> "DirectSumRepresentation":
        if not isinstance(multiplicity, int) or multiplicity < 1:
            raise ValueError("representation multiplicity must be a positive integer")
        return DirectSumRepresentation(self.representations * multiplicity)


def direct_sum(representations: Sequence[Representation], name: str | None = None) -> DirectSumRepresentation:
    if not representations:
        raise ValueError("a direct sum needs at least one representation")
    return DirectSumRepresentation(representations, name)


DirectSum = DirectSumRepresentation


@dataclass(frozen=True)
class MulIrrep:
    mul: int
    irrep: Irrep

    @property
    def dim(self) -> int:
        return self.mul * self.irrep.size


class Irreps(Sequence[MulIrrep]):
    """A typed direct sum, analogous to ``e3nn.o3.Irreps``.

    Entries can be ``(multiplicity, irrep)``, or ``(multiplicity, irrep_id)``
    when a group is supplied.
    """

    def __init__(self, group: "FiniteGroup", entries: Iterable[MulIrrep | tuple[int, Irrep | tuple[int, ...]]]):
        self.group = group
        parsed = []
        for entry in entries:
            if isinstance(entry, MulIrrep):
                mul, irrep = entry.mul, entry.irrep
            else:
                mul, value = entry
                irrep = value if isinstance(value, Irrep) else group.irrep(*value)
            if int(mul) < 0 or irrep.group is not group:
                raise ValueError("invalid multiplicity or irrep from another group")
            if int(mul):
                parsed.append(MulIrrep(int(mul), irrep))
        self._entries = tuple(parsed)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    def __iter__(self):
        return iter(self._entries)

    @property
    def dim(self) -> int:
        return sum(entry.dim for entry in self)

    def simplify(self) -> "Irreps":
        merged: list[tuple[int, Irrep]] = []
        for entry in self:
            if merged and merged[-1][1] is entry.irrep:
                merged[-1] = (merged[-1][0] + entry.mul, entry.irrep)
            else:
                merged.append((entry.mul, entry.irrep))
        return Irreps(self.group, merged)

    def regroup(self) -> "Irreps":
        counts = {irrep.id: 0 for irrep in self.group.irreps()}
        for entry in self:
            counts[entry.irrep.id] += entry.mul
        return Irreps(self.group, [(counts[i], i) for i in self.group.irrep_ids() if counts[i]])

    @cached_property
    def representation(self) -> Representation:
        expanded = [entry.irrep for entry in self for _ in range(entry.mul)]
        if not expanded:
            raise ValueError("empty Irreps has no non-empty representation matrix")
        return direct_sum(expanded)

    def __repr__(self) -> str:
        return " + ".join(f"{e.mul}x{e.irrep.id}" for e in self) or "Irreps()"
