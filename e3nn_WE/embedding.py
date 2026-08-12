"""Embeddings into O(3), restriction, and finite-irrep decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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
        self._matrix_cache = {}

    def matrix(self, element: GroupElement, *, dtype=None, device=None) -> torch.Tensor:
        self.group._check(element)
        matrix = self._matrix_cache.get(element.value)
        if matrix is None:
            matrix = torch.as_tensor(self._matrix_fn(element), dtype=torch.float64).detach().cpu().contiguous()
            self._matrix_cache[element.value] = matrix
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
        self._exact_decomposition = None
        # e3nn 0.6's matrix-to-angle path leaves a few 1e-6 of group-law
        # residual for reflections. C_n/D_n have complete real irrep catalogs, so
        # snap the numerical restriction to the equivalent exact finite-group
        # representation once, at construction time.
        if isinstance(embedding.group, (CyclicGroup, DihedralGroup)):
            decomposition = self._compute_decomposition()
            change = decomposition.change_of_basis
            change_inv = torch.linalg.inv(change)
            block_rep = decomposition.representation
            self._matrices = lambda element: change @ block_rep(element) @ change_inv
            self._matrix_cache.clear()
            self._exact_decomposition = decomposition

    @lru_cache(maxsize=None)
    def decompose(self) -> IrrepDecomposition:
        if self._exact_decomposition is not None:
            return self._exact_decomposition
        return self._compute_decomposition()

    def _compute_decomposition(self) -> IrrepDecomposition:
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
            maps = intertwiner_basis(irrep, self, method="nullspace")
            expected_maps = multiplicity * irrep.sum_of_squares_constituents
            if maps.shape[0] != expected_maps:
                raise RuntimeError("intertwiner dimension disagrees with character multiplicity")
            for mapping in maps:
                candidate = mapping.clone()
                if columns:
                    occupied = torch.cat(columns, dim=1)
                    candidate -= occupied @ (occupied.T @ candidate)
                u, singular_values, vh = torch.linalg.svd(candidate, full_matrices=False)
                if int((singular_values > 1e-7).sum()) != irrep.size:
                    continue
                columns.append(u @ vh)
                copies.append(irrep)
                if sum(copy is irrep for copy in copies) == multiplicity:
                    break
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
