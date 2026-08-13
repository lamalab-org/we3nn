"""Clebsch--Gordan/intertwiner coefficients for finite real representations."""

from __future__ import annotations

from functools import lru_cache
import math

import torch

from .representations import Irrep, Representation


def clebsch_gordan(
    left: Representation,
    right: Representation,
    output: Representation,
    *,
    method: str = "auto",
) -> torch.Tensor:
    """Return an orthonormal basis of equivariant bilinear couplings.

    The result has shape ``(multiplicity, output.size, left.size, right.size)`` and
    satisfies, for every path ``C`` and group element ``g``,

    ``C(rho_l(g)x, rho_r(g)y) = rho_o(g) C(x, y)``.

    The leading dimension counts representation copies, matching escnn's CG
    convention. For complex-type real C_n irreps, the full real Hom space also
    contains postcomposition by the two-dimensional endomorphism algebra;
    tensor-product modules expand those extra real weight directions
    internally while this public CG API reports coupling multiplicity.
    """
    if left.group is not right.group or left.group is not output.group:
        raise ValueError("all irreps must belong to the same group instance")
    if not all(isinstance(rep, Irrep) for rep in (left, right, output)):
        from .intertwiner import intertwiner_basis, tensor_product_representation

        basis = intertwiner_basis(
            tensor_product_representation(left, right), output, method=method
        )
        return basis.reshape(-1, output.size, left.size, right.size)
    if method == "auto":
        full = _invariant_tensor_basis(output, left, right)
    elif method in {"nullspace", "reynolds"}:
        from .intertwiner import intertwiner_basis, tensor_product_representation

        basis = intertwiner_basis(tensor_product_representation(left, right), output, method=method)
        full = basis.reshape(-1, output.size, left.size, right.size)
    else:
        raise ValueError("method must be 'auto', 'nullspace', or 'reynolds'")
    full = _canonical_subspace_basis(full)
    multiplicity = _multiplicity_from_hom_dimension(full.shape[0], output)
    return _copy_coupling_basis(full, output, multiplicity)


def _multiplicity_from_hom_dimension(hom_dimension: int, output: Irrep) -> int:
    endomorphism_dimension = output.sum_of_squares_constituents
    multiplicity, remainder = divmod(hom_dimension, endomorphism_dimension)
    if remainder:
        raise RuntimeError(
            "coupling-space dimension is not divisible by the output irrep's "
            "endomorphism dimension"
        )
    return multiplicity


def _copy_coupling_basis(
    full: torch.Tensor,
    output: Irrep,
    multiplicity: int,
) -> torch.Tensor:
    """Extract one normalized projection for each output-irrep copy.

    ``Hom_G(left x right, output)`` contains an entire orbit under
    ``End_G(output)`` for every representation copy. Consequently, truncating
    an arbitrary Hom basis can select several endomorphism directions for the
    same copy. We instead work with the transposed embedding maps, remove the
    images of copies already selected, and take an isometric polar factor.
    """
    if multiplicity == 0:
        return full[:0]

    domain_dimension = math.prod(full.shape[2:])
    embeddings: list[torch.Tensor] = []
    couplings: list[torch.Tensor] = []
    rank_tolerance = 128.0 * torch.finfo(full.dtype).eps * max(full.shape)

    for mapping in full.reshape(full.shape[0], output.size, domain_dimension):
        embedding = mapping.T.clone()
        if embeddings:
            occupied = torch.cat(embeddings, dim=1)
            embedding -= occupied @ (occupied.T @ embedding)

        u, singular_values, vh = torch.linalg.svd(embedding, full_matrices=False)
        if singular_values.numel() != output.size:
            continue
        tolerance = rank_tolerance * max(float(singular_values.max()), 1.0)
        if int((singular_values > tolerance).sum()) != output.size:
            continue

        isometric_embedding = u @ vh
        embeddings.append(isometric_embedding)
        # Preserve the historical unit-Frobenius normalization of CG tensors.
        coupling = isometric_embedding.T / math.sqrt(output.size)
        couplings.append(coupling.reshape(output.size, *full.shape[2:]))
        if len(couplings) == multiplicity:
            break

    if len(couplings) != multiplicity:
        raise RuntimeError("failed to extract all representation-copy couplings")
    return torch.stack(couplings).contiguous()


def _canonical_subspace_basis(basis: torch.Tensor) -> torch.Tensor:
    """Choose a coordinate-pivot basis depending only on the subspace."""
    if basis.shape[0] == 0:
        return basis
    flat = basis.flatten(1)
    projector = flat.T @ flat
    selected = []
    tolerance = 1e-10
    for coordinate in range(projector.shape[0]):
        vector = projector[:, coordinate].clone()
        for previous in selected:
            vector -= torch.dot(previous, vector) * previous
        norm = torch.linalg.vector_norm(vector)
        if float(norm) > tolerance:
            vector /= norm
            pivot = int(vector.abs().argmax())
            if vector[pivot] < 0:
                vector = -vector
            selected.append(vector)
        if len(selected) == flat.shape[0]:
            break
    return torch.stack(selected).reshape_as(basis)


@lru_cache(maxsize=None)
def _invariant_tensor_basis(
    output: Representation,
    left: Representation,
    right: Representation,
) -> torch.Tensor:
    """Orthonormal invariant tensors with axes ``(output, left, right)``.

    This generic routine is only used for irreps (dimensions at most two).
    Tensor-product modules use analytic paths whenever a regular
    representation is present, avoiding a large Reynolds projector.
    """
    if output.group is not left.group or output.group is not right.group:
        raise ValueError("all representations must belong to the same group")
    shape = (output.size, left.size, right.size)
    candidates = []
    for flat_index in range(output.size * left.size * right.size):
        elementary = torch.zeros(shape, dtype=torch.float64)
        elementary.reshape(-1)[flat_index] = 1.0
        projected = torch.zeros_like(elementary)
        for element in output.group.elements:
            projected += torch.einsum(
                "oa,ib,jc,abc->oij",
                output(element),
                left(element),
                right(element),
                elementary,
            )
        candidates.append((projected / output.group.order()).reshape(-1))
    span = torch.stack(candidates)
    _, singular_values, vh = torch.linalg.svd(span, full_matrices=False)
    tolerance = max(
        1e-10,
        float(max(span.shape) * torch.finfo(span.dtype).eps * singular_values.max()),
    )
    rank = int((singular_values > tolerance).sum())
    return vh[:rank].reshape(rank, *shape).contiguous()


def tensor_product_multiplicity(
    left: Representation,
    right: Representation,
    output: Irrep,
    *,
    method: str = "auto",
) -> int:
    """Return the number of representation copies of ``output`` in ``left x right``."""
    if not isinstance(output, Irrep):
        raise TypeError("tensor-product multiplicity requires an irreducible output")
    full = finite_group_couplings(left, right, output, method=method)
    return _multiplicity_from_hom_dimension(full.shape[0], output)


def coupling_dimension(left: Representation, right: Representation, output: Representation) -> int:
    """Backward-compatible alias for the number of CG copy representatives."""
    return int(clebsch_gordan(left, right, output).shape[0])


def finite_group_couplings(
    left: Representation,
    right: Representation,
    output: Representation,
    *,
    method: str = "auto",
) -> torch.Tensor:
    """Explicitly named full finite-group coupling API."""
    if all(isinstance(rep, Irrep) for rep in (left, right, output)):
        return full_coupling_basis(left, right, output)
    from .intertwiner import intertwiner_basis, tensor_product_representation

    basis = intertwiner_basis(tensor_product_representation(left, right), output, method=method)
    return basis.reshape(-1, output.dim, left.dim, right.dim)


def full_coupling_basis(left: Irrep, right: Irrep, output: Irrep) -> torch.Tensor:
    """All independent real Hom-space paths, including irrep endomorphisms."""
    return _invariant_tensor_basis(output, left, right)
