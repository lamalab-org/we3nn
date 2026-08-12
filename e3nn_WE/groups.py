"""Finite cyclic and dihedral groups and their real representations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import math
from typing import Iterable, Iterator, Sequence

import torch


@dataclass(frozen=True, slots=True)
class GroupElement:
    """An immutable element belonging to a finite group."""

    group: "FiniteGroup"
    value: int | tuple[int, int]

    def __matmul__(self, other: "GroupElement") -> "GroupElement":
        return self.group.combine(self, other)

    def __mul__(self, other: "GroupElement") -> "GroupElement":
        return self.group.combine(self, other)

    def inverse(self) -> "GroupElement":
        return self.group.inverse(self)


class FiniteGroup:
    """Common API for finite groups, close to escnn's fiber-group API."""

    name: str
    _elements: tuple[GroupElement, ...]

    @property
    def elements(self) -> tuple[GroupElement, ...]:
        return self._elements

    @property
    def testing_elements(self) -> tuple[GroupElement, ...]:
        return self._elements

    def order(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[GroupElement]:
        return iter(self._elements)

    def sample(self) -> GroupElement:
        return self._elements[torch.randint(self.order(), ()).item()]

    def _check(self, element: GroupElement) -> None:
        if not isinstance(element, GroupElement) or element.group is not self:
            raise ValueError(f"{element!r} is not an element of {self.name}")

    @cached_property
    def regular_repr(self):
        from .representations import Representation

        index = {g.value: i for i, g in enumerate(self.elements)}

        def matrices(element: GroupElement) -> torch.Tensor:
            self._check(element)
            matrix = torch.zeros(self.order(), self.order(), dtype=torch.float64)
            for column, h in enumerate(self.elements):
                row = index[self.combine(element, h).value]
                matrix[row, column] = 1.0
            return matrix

        return Representation(
            self,
            "regular",
            self.order(),
            matrices,
            supported_nonlinearities=frozenset({"pointwise", "norm", "gated"}),
            is_permutation=True,
        )

    @property
    def regular_representation(self):
        return self.regular_repr

    @cached_property
    def trivial_representation(self):
        return self.irrep(*self.trivial_id)

    @property
    def trivial_repr(self):
        return self.trivial_representation

    def tensor_product(self, left, right) -> list[tuple[int, tuple[int, ...]]]:
        """Decompose a tensor product using real character orthogonality."""
        products = []
        for irrep_id in self.irrep_ids():
            candidate = self.irrep(*irrep_id)
            inner = sum(
                left.character(g) * right.character(g) * candidate.character(g)
                for g in self.elements
            ) / self.order()
            # Endomorphism dimension is 2 for real forms of complex-type C_n irreps.
            endomorphism_dim = candidate.sum_of_squares_constituents
            multiplicity = int(round(inner / endomorphism_dim))
            if multiplicity:
                products.append((multiplicity, irrep_id))
        return products


class CyclicGroup(FiniteGroup):
    """The rotation group C_n with real irreducible representations."""

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.name = f"C{n}"
        self._elements = tuple(GroupElement(self, k) for k in range(n))
        self.identity = self._elements[0]
        self.trivial_id = (0,)

    def __repr__(self) -> str:
        return f"CyclicGroup({self.n})"

    def element(self, value: int | GroupElement) -> GroupElement:
        if isinstance(value, GroupElement):
            self._check(value)
            return value
        return self._elements[int(value) % self.n]

    def combine(self, left: GroupElement, right: GroupElement) -> GroupElement:
        self._check(left)
        self._check(right)
        return self.element(int(left.value) + int(right.value))

    def inverse(self, element: GroupElement) -> GroupElement:
        self._check(element)
        return self.element(-int(element.value))

    def irrep_ids(self) -> tuple[tuple[int], ...]:
        return tuple((k,) for k in range(self.n // 2 + 1))

    def irreps(self):
        return tuple(self.irrep(*irrep_id) for irrep_id in self.irrep_ids())

    @lru_cache(maxsize=None)
    def irrep(self, frequency: int):
        from .representations import Irrep

        frequency = int(frequency)
        if not 0 <= frequency <= self.n // 2:
            raise ValueError(f"C{self.n} real irrep frequency must be in [0, {self.n // 2}]")
        one_dimensional = frequency == 0 or (self.n % 2 == 0 and frequency == self.n // 2)
        size = 1 if one_dimensional else 2

        def matrices(element: GroupElement) -> torch.Tensor:
            self._check(element)
            angle = 2.0 * math.pi * frequency * int(element.value) / self.n
            if size == 1:
                return torch.tensor([[math.cos(angle)]], dtype=torch.float64)
            c, s = math.cos(angle), math.sin(angle)
            return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

        return Irrep(self, (frequency,), size, matrices, complex_type=size == 2)

    @cached_property
    def standard_representation(self):
        if self.n > 2:
            return self.irrep(1)
        from .representations import Representation

        def matrices(element: GroupElement) -> torch.Tensor:
            angle = 2.0 * math.pi * int(element.value) / self.n
            c, s = math.cos(angle), math.sin(angle)
            return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

        return Representation(self, "standard", 2, matrices)


class DihedralGroup(FiniteGroup):
    """The order-2n symmetry group D_n of a regular n-gon.

    Elements are ``(flip, rotation)`` and represented as ``r^rotation s^flip``.
    This is the tuple convention used by escnn's dihedral groups.
    """

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.name = f"D{n}"
        self._elements = tuple(GroupElement(self, (flip, k)) for flip in range(2) for k in range(n))
        self.identity = self._elements[0]
        self.trivial_id = (0, 0)

    def __repr__(self) -> str:
        return f"DihedralGroup({self.n})"

    def element(self, value: Sequence[int] | GroupElement) -> GroupElement:
        if isinstance(value, GroupElement):
            self._check(value)
            return value
        if len(value) != 2:
            raise ValueError("a dihedral element is a (flip, rotation) pair")
        flip, rotation = int(value[0]) % 2, int(value[1]) % self.n
        return self._elements[flip * self.n + rotation]

    def combine(self, left: GroupElement, right: GroupElement) -> GroupElement:
        self._check(left)
        self._check(right)
        flip_l, rotation_l = left.value
        flip_r, rotation_r = right.value
        return self.element((flip_l ^ flip_r, rotation_l + (-1) ** flip_l * rotation_r))

    def inverse(self, element: GroupElement) -> GroupElement:
        self._check(element)
        flip, rotation = element.value
        return self.element((flip, -((-1) ** flip) * rotation))

    def irrep_ids(self) -> tuple[tuple[int, int], ...]:
        ids: list[tuple[int, int]] = [(0, 0), (1, 0)]
        ids.extend((1, k) for k in range(1, (self.n - 1) // 2 + 1))
        if self.n % 2 == 0:
            ids.extend(((0, self.n // 2), (1, self.n // 2)))
        return tuple(ids)

    def irreps(self):
        return tuple(self.irrep(*irrep_id) for irrep_id in self.irrep_ids())

    @lru_cache(maxsize=None)
    def irrep(self, flip_frequency: int, frequency: int):
        from .representations import Irrep

        flip_frequency, frequency = int(flip_frequency), int(frequency)
        irrep_id = (flip_frequency, frequency)
        if irrep_id not in self.irrep_ids():
            raise ValueError(f"{irrep_id} is not an irrep id of D{self.n}; choose from {self.irrep_ids()}")
        one_dimensional = frequency == 0 or (self.n % 2 == 0 and frequency == self.n // 2)
        size = 1 if one_dimensional else 2

        def matrices(element: GroupElement) -> torch.Tensor:
            self._check(element)
            flip, rotation = element.value
            angle = 2.0 * math.pi * frequency * rotation / self.n
            if size == 1:
                value = math.cos(angle) * ((-1.0) ** (flip_frequency * flip))
                return torch.tensor([[value]], dtype=torch.float64)
            c, s = math.cos(angle), math.sin(angle)
            rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
            reflection = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
            return rotation_matrix @ (reflection if flip else torch.eye(2, dtype=torch.float64))

        return Irrep(self, irrep_id, size, matrices)

    @cached_property
    def standard_representation(self):
        if self.n > 2:
            return self.irrep(1, 1)
        from .representations import Representation

        def matrices(element: GroupElement) -> torch.Tensor:
            flip, rotation = element.value
            angle = 2.0 * math.pi * rotation / self.n
            c, s = math.cos(angle), math.sin(angle)
            rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
            reflection = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
            return rotation_matrix @ (reflection if flip else torch.eye(2, dtype=torch.float64))

        return Representation(self, "standard", 2, matrices)


@lru_cache(maxsize=None)
def cyclic_group(n: int) -> CyclicGroup:
    return CyclicGroup(n)


@lru_cache(maxsize=None)
def dihedral_group(n: int) -> DihedralGroup:
    return DihedralGroup(n)
