"""Small dependency-free utilities used by the message-passing example."""

from __future__ import annotations

import torch


def scatter_sum(
    source: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Differentiable sum-scatter implemented with core PyTorch."""
    if dim < 0:
        dim += source.ndim
    if not 0 <= dim < source.ndim:
        raise ValueError(f"invalid scatter dimension {dim}")
    if index.ndim != 1 or index.shape[0] != source.shape[dim]:
        raise ValueError("index must be one-dimensional and match source along dim")
    if index.dtype != torch.long:
        raise TypeError("scatter indices must have dtype torch.long")
    size = int(index.max()) + 1 if dim_size is None and index.numel() else 0
    size = size if dim_size is None else int(dim_size)
    output_shape = list(source.shape)
    output_shape[dim] = size
    output = source.new_zeros(output_shape)
    return output.index_add(dim, index, source)


def scatter(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """The subset of :mod:`torch_scatter.scatter` needed by the example."""
    if reduce not in {"sum", "add"}:
        raise NotImplementedError("the built-in fallback currently supports sum reduction")
    return scatter_sum(source, index, dim=dim, dim_size=dim_size)


def edge_radial_basis(
    displacement: torch.Tensor,
    *,
    basis_size: int,
    max_radius: float,
) -> torch.Tensor:
    """Smooth Gaussian radial basis with a cosine cutoff."""
    if displacement.shape[-1] < 1:
        raise ValueError("displacement must have a nonempty coordinate axis")
    if basis_size < 1 or max_radius <= 0:
        raise ValueError("basis_size and max_radius must be positive")
    radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    centers = torch.linspace(0.0, max_radius, basis_size, device=radius.device, dtype=radius.dtype)
    width = basis_size / max_radius
    gaussian = torch.exp(-0.5 * (width * (radius - centers)) ** 2)
    cutoff = 0.5 * (torch.cos(torch.pi * radius / max_radius) + 1.0)
    cutoff = torch.where(radius < max_radius, cutoff, torch.zeros_like(cutoff))
    return gaussian * cutoff


def edge_wa_joint_features(edge_xy: torch.Tensor, anchor_xy: torch.Tensor) -> torch.Tensor:
    """Two D_n-invariant joint edge/anchor scalar features."""
    if edge_xy.shape != anchor_xy.shape or edge_xy.shape[-1] != 2:
        raise ValueError("edge and anchor vectors must have matching (..., 2) shapes")
    dot = (edge_xy * anchor_xy).sum(dim=-1, keepdim=True)
    anchor_norm_squared = anchor_xy.square().sum(dim=-1, keepdim=True)
    return torch.cat((dot, anchor_norm_squared), dim=-1)
