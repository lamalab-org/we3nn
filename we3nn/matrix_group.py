"""Arbitrary finite groups specified by multiplication tables."""

from __future__ import annotations

from functools import cached_property
from typing import Hashable, Sequence

from .groups import FiniteGroup, GroupElement


class MatrixFiniteGroup(FiniteGroup):
    """A finite group with deterministic elements and explicit operation tables.

    Despite the historical name from the implementation plan, group elements
    need not themselves be matrices. User representations are attached with
    :meth:`FiniteGroup.representation`.
    """

    def __init__(
        self,
        elements: Sequence[Hashable],
        multiplication_table: Sequence[Sequence[int]],
        inverse_table: Sequence[int],
        identity: int,
        *,
        name: str = "FiniteGroup",
        generators: Sequence[int] | None = None,
    ):
        values = tuple(elements)
        if not values or len(set(values)) != len(values):
            raise ValueError("elements must be a nonempty deterministic sequence of unique values")
        order = len(values)
        table = tuple(tuple(map(int, row)) for row in multiplication_table)
        inverses = tuple(map(int, inverse_table))
        if len(table) != order or any(len(row) != order for row in table):
            raise ValueError("multiplication_table must be square with one row per element")
        if len(inverses) != order or not 0 <= identity < order:
            raise ValueError("invalid inverse table or identity index")
        if any(not 0 <= index < order for row in table for index in row) or any(not 0 <= index < order for index in inverses):
            raise ValueError("group tables contain an out-of-range element index")
        self.name = str(name)
        self._elements = tuple(GroupElement(self, value) for value in values)
        self._index = {value: index for index, value in enumerate(values)}
        self._table = table
        self._inverse_table = inverses
        self.identity = self._elements[identity]
        self._generator_indices = tuple(range(order) if generators is None else map(int, generators))
        self._validate_group_axioms()

    def __repr__(self) -> str:
        return f"MatrixFiniteGroup(name={self.name!r}, order={self.order()})"

    @property
    def generators(self):
        return tuple(self._elements[index] for index in self._generator_indices)

    def element(self, value) -> GroupElement:
        if isinstance(value, GroupElement):
            self._check(value)
            return value
        return self._elements[self._index[value]]

    def combine(self, left: GroupElement, right: GroupElement) -> GroupElement:
        self._check(left)
        self._check(right)
        return self._elements[self._table[self._index[left.value]][self._index[right.value]]]

    def inverse(self, element: GroupElement) -> GroupElement:
        self._check(element)
        return self._elements[self._inverse_table[self._index[element.value]]]

    def _validate_group_axioms(self) -> None:
        for first in self._elements:
            if self.combine(first, self.identity) != first or self.combine(self.identity, first) != first:
                raise ValueError("identity table entries are invalid")
            if self.combine(first, self.inverse(first)) != self.identity:
                raise ValueError("inverse table entries are invalid")
            for second in self._elements:
                for third in self._elements:
                    if self.combine(self.combine(first, second), third) != self.combine(first, self.combine(second, third)):
                        raise ValueError("multiplication table is not associative")

    @cached_property
    def trivial_representation(self):
        import torch
        from .representations import Representation

        return Representation(
            self,
            "trivial",
            1,
            lambda element: torch.ones(1, 1, dtype=torch.float64),
            supported_nonlinearities=frozenset({"pointwise", "norm", "gated"}),
            is_permutation=True,
        )

    @property
    def trivial_repr(self):
        return self.trivial_representation
