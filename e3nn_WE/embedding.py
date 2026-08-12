"""Embeddings into O(3), restriction, and finite-irrep decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from e3nn import o3

from .groups import CyclicGroup, DihedralGroup, FiniteGroup, GroupElement
from .intertwiner import intertwiner_basis
from .representations import Irrep, Representation, direct_sum


class O3Embedding:
    def __init__(self, group: FiniteGroup, matrix_fn, *, name: str = "O3 embedding"):
        self.group = group
        self._matrix_fn = matrix_fn
        self.name = name

    def matrix(self, element: GroupElement, *, dtype=None, device=None) -> torch.Tensor:
        self.group._check(element)
        matrix = torch.as_tensor(self._matrix_fn(element), dtype=torch.float64)
        if matrix.shape != (3, 3):
            raise ValueError("an O3 embedding must return 3x3 matrices")
        return matrix.to(dtype=dtype, device=device)

    def check_embedding(self, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
        identity = torch.eye(3, dtype=torch.float64)
        for first in self.group.elements:
            matrix = self.matrix(first)
            torch.testing.assert_close(matrix.T @ matrix, identity, atol=atol, rtol=rtol)
            for second in self.group.elements:
                torch.testing.assert_close(
                    self.matrix(self.group.combine(first, second)),
                    matrix @ self.matrix(second),
                    atol=atol,
                    rtol=rtol,
                )
        return True


def planar_o3(group: CyclicGroup | DihedralGroup) -> O3Embedding:
    def matrices(element):
        matrix = torch.eye(3, dtype=torch.float64)
        matrix[:2, :2] = group.standard_representation(element)
        return matrix

    return O3Embedding(group, matrices, name=f"planar {group.name} in O3")


@dataclass(frozen=True)
class IrrepDecomposition:
    irreps: tuple[Irrep, ...]
    change_of_basis: torch.Tensor

    @property
    def representation(self):
        return direct_sum(self.irreps)

    def reconstruct(self, element) -> torch.Tensor:
        block = self.representation(element)
        return self.change_of_basis @ block @ torch.linalg.inv(self.change_of_basis)


class RestrictedO3Representation(Representation):
    def __init__(self, o3_irrep: o3.Irrep, embedding: O3Embedding):
        self.o3_irrep = o3_irrep
        self.embedding = embedding
        super().__init__(
            embedding.group,
            f"restrict({o3_irrep})",
            o3_irrep.dim,
            lambda element: o3_irrep.D_from_matrix(embedding.matrix(element)).to(torch.float64),
            is_orthogonal=True,
            basis_kind="restricted_o3",
        )

    def decompose(self) -> IrrepDecomposition:
        if not hasattr(self.group, "irreps"):
            raise NotImplementedError("decomposition requires a supplied irrep catalog")
        copies = []
        columns = []
        for irrep in self.group.irreps():
            character_inner = sum(
                self.character(element) * irrep.character(element) for element in self.group.elements
            ) / self.group.order()
            multiplicity = int(round(character_inner / irrep.sum_of_squares_constituents))
            if not multiplicity:
                continue
            if irrep.sum_of_squares_constituents != 1:
                raise NotImplementedError("automatic decomposition of complex-type real irreps is not yet canonical")
            maps = intertwiner_basis(irrep, self, method="nullspace")
            if maps.shape[0] != multiplicity:
                raise RuntimeError("intertwiner dimension disagrees with character multiplicity")
            for mapping in maps:
                columns.append(math.sqrt(irrep.size) * mapping)
                copies.append(irrep)
        change = torch.cat(columns, dim=1) if columns else torch.empty(self.size, 0, dtype=torch.float64)
        if change.shape != (self.size, self.size):
            raise RuntimeError("restricted representation decomposition is incomplete")
        decomposition = IrrepDecomposition(tuple(copies), change)
        for element in self.group.elements:
            torch.testing.assert_close(
                decomposition.reconstruct(element), self(element), atol=3e-6, rtol=3e-6
            )
        return decomposition


@lru_cache(maxsize=None)
def restrict_o3(o3_irrep: o3.Irrep | str, embedding: O3Embedding) -> RestrictedO3Representation:
    return RestrictedO3Representation(o3.Irrep(o3_irrep), embedding)


def restricted_o3_couplings(
    left: o3.Irrep | str,
    right: o3.Irrep | str,
    output: o3.Irrep | str,
) -> torch.Tensor:
    """Inherited O(3) Wigner coupling, distinct from the full finite CG space."""
    left, right, output = o3.Irrep(left), o3.Irrep(right), o3.Irrep(output)
    if output not in left * right:
        return torch.empty(0, output.dim, left.dim, right.dim, dtype=torch.float64)
    # e3nn orders wigner_3j axes as (left, right, output).
    return o3.wigner_3j(left.l, right.l, output.l).permute(2, 0, 1).unsqueeze(0).to(torch.float64)
