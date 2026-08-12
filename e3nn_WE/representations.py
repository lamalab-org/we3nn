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
    ):
        self.group = group
        self.name = name
        self.size = int(size)
        self._matrices = matrices
        self.supported_nonlinearities = supported_nonlinearities
        self.is_permutation = is_permutation

    def __call__(self, element: "GroupElement") -> torch.Tensor:
        matrix = self._matrices(element)
        if matrix.shape != (self.size, self.size):
            raise RuntimeError(f"representation {self.name} produced matrix with shape {matrix.shape}")
        return matrix

    def character(self, element: "GroupElement") -> float:
        return float(torch.trace(self(element)))

    def __repr__(self) -> str:
        return f"{self.group.name}:{self.name}"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


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
        )

    @property
    def is_trivial(self) -> bool:
        return self.id == self.group.trivial_id


def direct_sum(representations: Sequence[Representation], name: str | None = None) -> Representation:
    if not representations:
        raise ValueError("a direct sum needs at least one representation")
    group = representations[0].group
    if any(rep.group is not group for rep in representations):
        raise ValueError("all representations in a direct sum must belong to the same group")
    size = sum(rep.size for rep in representations)

    def matrices(element: "GroupElement") -> torch.Tensor:
        return torch.block_diag(*(rep(element) for rep in representations))

    return Representation(group, name or "+".join(rep.name for rep in representations), size, matrices)


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
