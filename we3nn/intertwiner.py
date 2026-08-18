"""Generic numerical representation intertwiners and subspace diagnostics."""

from __future__ import annotations

from functools import lru_cache
import math

import torch

from .representations import Representation


def _canonicalize(basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[0] == 0:
        return basis
    flat = basis.flatten(1)
    pivots = flat.abs().argmax(dim=1)
    signs = torch.sign(flat[torch.arange(flat.shape[0]), pivots])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    flat = flat * signs[:, None]
    order = sorted(range(flat.shape[0]), key=lambda i: (int(pivots[i]), tuple((-flat[i].abs()).tolist())))
    return flat[order].reshape_as(basis).contiguous()


def _nullspace(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] == 0:
        return torch.eye(matrix.shape[1], dtype=matrix.dtype)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    sigma_max = float(singular_values.max()) if singular_values.numel() else 0.0
    tolerance = 32.0 * max(matrix.shape) * torch.finfo(matrix.dtype).eps * max(sigma_max, 1.0)
    rank = int((singular_values > tolerance).sum())
    return vh[rank:]


@lru_cache(maxsize=None)
def _intertwiner_cpu(rep_in: Representation, rep_out: Representation, method: str, elements: str) -> torch.Tensor:
    if rep_in.group is not rep_out.group:
        raise ValueError("representations must belong to the same group")
    if method not in {"nullspace", "reynolds"}:
        raise ValueError("invalid intertwiner method")
    group_elements = rep_in.group.elements if elements == "all" else rep_in.group.generators
    shape = (rep_out.size, rep_in.size)
    elementary = torch.eye(math.prod(shape), dtype=torch.float64).reshape(-1, *shape)
    if method == "reynolds":
        projected = []
        for candidate in elementary:
            # The Reynolds action on Hom(V_in, V_out) uses the inverse. The
            # transpose is only an optimization for orthogonal inputs.
            average = sum(
                rep_out(element)
                @ candidate
                @ (
                    rep_in(element).T
                    if rep_in.is_orthogonal
                    else torch.linalg.inv(rep_in(element))
                )
                for element in rep_in.group.elements
            ) / rep_in.group.order()
            projected.append(average.reshape(-1))
        span = torch.stack(projected)
        _, singular_values, vh = torch.linalg.svd(span, full_matrices=False)
        tolerance = 32.0 * max(span.shape) * torch.finfo(span.dtype).eps * max(float(singular_values.max()), 1.0)
        basis = vh[: int((singular_values > tolerance).sum())]
    else:
        constraints = []
        for element in group_elements:
            residuals = [
                (rep_out(element) @ candidate - candidate @ rep_in(element)).reshape(-1)
                for candidate in elementary
            ]
            constraints.append(torch.stack(residuals, dim=1))
        constraint = torch.cat(constraints, dim=0)
        # Restricted O(3) matrices inherit the numerical error of e3nn's
        # matrix-to-angle conversion. Estimate that defect from the supplied
        # representation itself instead of imposing a global absolute cutoff.
        defect = max(_representation_defect(rep_in), _representation_defect(rep_out))
        if defect:
            _, singular_values, vh = torch.linalg.svd(constraint, full_matrices=True)
            scale = max(float(torch.linalg.matrix_norm(constraint, ord=2)), 1.0)
            tolerance = (
                32.0 * max(constraint.shape) * torch.finfo(constraint.dtype).eps * scale
                + 8.0 * math.sqrt(len(group_elements)) * defect
            )
            rank = int((singular_values > tolerance).sum())
            basis = vh[rank:]
        else:
            basis = _nullspace(constraint)
    return _canonicalize(basis.reshape(-1, *shape))


def _representation_defect(rep: Representation) -> float:
    """Maximum declared representation-law residual in float64."""
    identity = torch.eye(rep.size, dtype=torch.float64)
    error = 0.0
    for generator in rep.group.generators:
        matrix = rep(generator)
        if rep.is_orthogonal:
            error = max(error, float((matrix.T @ matrix - identity).abs().max()))
        for other in rep.group.generators:
            expected = rep(rep.group.combine(generator, other))
            error = max(error, float((expected - matrix @ rep(other)).abs().max()))
    return error


def intertwiner_basis(
    rep_in: Representation,
    rep_out: Representation,
    *,
    method: str = "auto",
    dtype: torch.dtype = torch.float64,
    device=None,
    elements: str = "all",
) -> torch.Tensor:
    """Orthonormal basis of ``Hom_G(rep_in, rep_out)``."""
    selected = "nullspace" if method == "auto" else method
    if elements not in {"all", "generators"}:
        raise ValueError("elements must be 'all' or 'generators'")
    return _intertwiner_cpu(rep_in, rep_out, selected, elements).to(dtype=dtype, device=device)


def invariant_basis(rep: Representation, **kwargs) -> torch.Tensor:
    return intertwiner_basis(rep.group.trivial_representation, rep, **kwargs)[..., 0]


class TensorProductRepresentation(Representation):
    def __init__(self, left: Representation, right: Representation):
        if left.group is not right.group:
            raise ValueError("tensor factors must belong to the same group")
        self.left = left
        self.right = right
        super().__init__(
            left.group,
            f"{left.name} x {right.name}",
            left.size * right.size,
            lambda element: torch.kron(left(element), right(element)),
            is_orthogonal=left.is_orthogonal and right.is_orthogonal,
        )


def tensor_product_representation(left: Representation, right: Representation) -> TensorProductRepresentation:
    return TensorProductRepresentation(left, right)


def subspace_diagnostics(first: torch.Tensor, second: torch.Tensor) -> dict:
    first_q = torch.linalg.qr(first.flatten(1).T, mode="reduced").Q
    second_q = torch.linalg.qr(second.flatten(1).T, mode="reduced").Q
    singular_values = torch.linalg.svdvals(first_q.T @ second_q).clamp(0, 1)
    projector_error = torch.linalg.matrix_norm(first_q @ first_q.T - second_q @ second_q.T).item()
    largest_angle = float(torch.acos(singular_values.min())) if singular_values.numel() else 0.0
    return {
        "dimension_first": first_q.shape[1],
        "dimension_second": second_q.shape[1],
        "singular_values": singular_values.tolist(),
        "smallest_singular_value": float(singular_values.min()) if singular_values.numel() else None,
        "largest_singular_value": float(singular_values.max()) if singular_values.numel() else None,
        "condition_number": (
            float(singular_values.max() / singular_values.min())
            if singular_values.numel() and float(singular_values.min()) > 0
            else None
        ),
        "largest_principal_angle": largest_angle,
        "projector_error": projector_error,
    }


def subspace_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(subspace_diagnostics(first, second)["projector_error"])


def find_representation_intertwiner(
    source_rep: Representation,
    target_rep: Representation,
    *,
    atol: float = 1e-8,
) -> torch.Tensor:
    """Return one orthogonal global basis transform between equivalent reps."""
    if source_rep.dim != target_rep.dim:
        raise ValueError("equivalent representations must have equal dimensions")
    basis = intertwiner_basis(source_rep, target_rep)
    if not basis.shape[0]:
        raise ValueError("representations are not equivalent")
    coefficients = torch.arange(1, basis.shape[0] + 1, dtype=torch.float64)
    candidate = torch.einsum("q,qoi->oi", coefficients, basis)
    if torch.linalg.matrix_rank(candidate) != source_rep.dim:
        for index in range(basis.shape[0]):
            candidate = basis[index]
            if torch.linalg.matrix_rank(candidate) == source_rep.dim:
                break
        else:
            raise ValueError("intertwiner space contains no invertible equivalence")
    u, _, vh = torch.linalg.svd(candidate)
    transform = u @ vh
    for element in source_rep.group.elements:
        residual = target_rep(element) @ transform - transform @ source_rep(element)
        if float(residual.abs().max()) > atol:
            raise ValueError("representations are not orthogonally equivalent within tolerance")
    return transform
