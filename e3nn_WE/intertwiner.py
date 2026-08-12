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
    # Restricted e3nn Wigner-D matrices currently carry angle-extraction
    # residuals around 1e-6 even in float64. The absolute floor separates
    # those numerical null directions from the O(1) constraint spectrum.
    tolerance = max(1e-5, 32.0 * max(matrix.shape) * torch.finfo(matrix.dtype).eps * max(sigma_max, 1.0))
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
            average = sum(
                rep_out(element) @ candidate @ rep_in(element).T for element in rep_in.group.elements
            ) / rep_in.group.order()
            projected.append(average.reshape(-1))
        span = torch.stack(projected)
        _, singular_values, vh = torch.linalg.svd(span, full_matrices=False)
        tolerance = max(1e-5, 32.0 * max(span.shape) * torch.finfo(span.dtype).eps * max(float(singular_values.max()), 1.0))
        basis = vh[: int((singular_values > tolerance).sum())]
    else:
        constraints = []
        for element in group_elements:
            residuals = [
                (rep_out(element) @ candidate - candidate @ rep_in(element)).reshape(-1)
                for candidate in elementary
            ]
            constraints.append(torch.stack(residuals, dim=1))
        basis = _nullspace(torch.cat(constraints, dim=0))
    return _canonicalize(basis.reshape(-1, *shape))


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
        "largest_principal_angle": largest_angle,
        "projector_error": projector_error,
    }


def subspace_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(subspace_diagnostics(first, second)["projector_error"])
