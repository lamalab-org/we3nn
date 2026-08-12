"""Circular harmonics for finite subgroups of O(2)."""

from __future__ import annotations

import math

import torch
from torch import nn
from e3nn import o3

from .groups import CyclicGroup, DihedralGroup, FiniteGroup
from .gspaces import GSpace, no_base_space
from .nn.field_type import FieldType
from .nn.geometric_tensor import GeometricTensor
from .representations import Representation
from .embedding import O3Embedding, planar_o3, restrict_o3


class CircularHarmonics(nn.Module):
    """Real angular harmonics transforming in C_n or D_n irreps.

    Frequencies from zero through ``max_frequency`` are returned in ascending
    order. To avoid aliased representations, the maximum is limited to
    ``floor(n / 2)``. For even C_n, cosine and sine at the Nyquist frequency
    are two copies of the same one-dimensional rotation irrep. For even D_n,
    they are the reflection-even and reflection-odd Nyquist irreps.

    ``normalization`` follows e3nn terminology:

    * ``"norm"``: each cosine/sine pair has unit norm;
    * ``"component"``: each nonconstant component has unit angular variance;
    * ``"integral"``: orthonormal under integration over ``[0, 2*pi)``.
    """

    def __init__(
        self,
        space_or_group: GSpace | FiniteGroup,
        max_frequency: int | None = None,
        normalization: str = "component",
    ):
        super().__init__()
        self.space = space_or_group if isinstance(space_or_group, GSpace) else no_base_space(space_or_group)
        self.group = self.space.fibergroup
        if not isinstance(self.group, (CyclicGroup, DihedralGroup)):
            raise TypeError("circular harmonics require a cyclic or dihedral group")
        maximum = self.group.n // 2
        self.max_frequency = maximum if max_frequency is None else int(max_frequency)
        if not 0 <= self.max_frequency <= maximum:
            raise ValueError(f"max_frequency must be between 0 and {maximum} for {self.group.name}")
        if normalization not in {"norm", "component", "integral"}:
            raise ValueError("normalization must be 'norm', 'component', or 'integral'")
        self.normalization = normalization

        representations = []
        layout: list[tuple[int, str]] = [(0, "cos")]
        representations.append(self.group.trivial_representation)
        for frequency in range(1, self.max_frequency + 1):
            nyquist = self.group.n % 2 == 0 and frequency == self.group.n // 2
            if isinstance(self.group, CyclicGroup):
                if nyquist:
                    representations.extend((self.group.irrep(frequency), self.group.irrep(frequency)))
                    layout.extend(((frequency, "cos"), (frequency, "sin")))
                else:
                    representations.append(self.group.irrep(frequency))
                    layout.append((frequency, "pair"))
            elif nyquist:
                representations.extend(
                    (self.group.irrep(0, frequency), self.group.irrep(1, frequency))
                )
                layout.extend(((frequency, "cos"), (frequency, "sin")))
            else:
                representations.append(self.group.irrep(1, frequency))
                layout.append((frequency, "pair"))
        self.out_type = FieldType(self.space, representations)
        self.rep_out = self.out_type.representation
        self._layout = tuple(layout)

    def _scales(self, frequency: int, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalization == "norm":
            pair_scale = 1.0
            scalar_scale = 1.0
        elif self.normalization == "component":
            pair_scale = math.sqrt(2.0)
            scalar_scale = 1.0 if frequency == 0 else math.sqrt(2.0)
        else:
            pair_scale = 1.0 / math.sqrt(math.pi)
            scalar_scale = 1.0 / math.sqrt(2.0 * math.pi) if frequency == 0 else 1.0 / math.sqrt(math.pi)
        return (
            torch.as_tensor(pair_scale, device=device, dtype=dtype),
            torch.as_tensor(scalar_scale, device=device, dtype=dtype),
        )

    def forward(self, angles: torch.Tensor) -> GeometricTensor:
        if not isinstance(angles, torch.Tensor) or not angles.is_floating_point():
            raise TypeError("angles must be a floating-point torch.Tensor")
        values = []
        for frequency, mode in self._layout:
            pair_scale, scalar_scale = self._scales(frequency, device=angles.device, dtype=angles.dtype)
            phase = frequency * angles
            if mode == "pair":
                values.append(pair_scale * torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1))
            elif mode == "cos":
                values.append((scalar_scale * torch.cos(phase)).unsqueeze(-1))
            else:
                values.append((scalar_scale * torch.sin(phase)).unsqueeze(-1))
        return GeometricTensor(torch.cat(values, dim=-1), self.out_type)

    def from_vectors(self, vectors: torch.Tensor) -> GeometricTensor:
        if vectors.shape[-1] != 2:
            raise ValueError("vectors must have final dimension 2")
        return self(torch.atan2(vectors[..., 1], vectors[..., 0]))


def circular_harmonics(
    group: FiniteGroup,
    angles: torch.Tensor,
    max_frequency: int | None = None,
    normalization: str = "component",
) -> torch.Tensor:
    """Functional circular harmonics returning an untyped tensor."""
    return CircularHarmonics(group, max_frequency, normalization)(angles).tensor


class RestrictedSphericalHarmonics(nn.Module):
    """Ordinary e3nn 3D spherical harmonics restricted to C_n or D_n.

    C_n and D_n are embedded in O(3) by acting on ``(x, y)`` and leaving
    ``z`` fixed. The numerical harmonics are computed by
    :func:`e3nn.o3.spherical_harmonics`; this wrapper supplies their exact
    restricted finite-group representation as a :class:`FieldType`.
    """

    def __init__(
        self,
        space_or_group: GSpace | FiniteGroup | None = None,
        degrees: int | list[int] | tuple[int, ...] | None = None,
        *,
        group: FiniteGroup | None = None,
        ls: int | list[int] | tuple[int, ...] | None = None,
        normalize: bool = True,
        normalization: str = "component",
        embedding: O3Embedding | None = None,
        basis: str = "o3",
    ):
        super().__init__()
        if space_or_group is None:
            space_or_group = group
        if degrees is None:
            degrees = ls
        if space_or_group is None or degrees is None:
            raise ValueError("supply a group/space and degrees/ls")
        self.space = space_or_group if isinstance(space_or_group, GSpace) else no_base_space(space_or_group)
        self.group = self.space.fibergroup
        if not isinstance(self.group, (CyclicGroup, DihedralGroup)):
            raise TypeError("restricted spherical harmonics require C_n or D_n")
        self.degrees = (int(degrees),) if isinstance(degrees, int) else tuple(map(int, degrees))
        if not self.degrees or any(degree < 0 for degree in self.degrees):
            raise ValueError("degrees must contain nonnegative integers")
        if normalization not in {"integral", "component", "norm"}:
            raise ValueError("unsupported spherical-harmonic normalization")
        self.normalize = bool(normalize)
        self.normalization = normalization
        self.embedding = embedding or planar_o3(self.group)
        representations = [restrict_o3(o3.Irrep(degree, (-1) ** degree), self.embedding) for degree in self.degrees]
        if basis not in {"o3", "finite_irreps"}:
            raise ValueError("basis must be 'o3' or 'finite_irreps'")
        self.basis = basis
        if basis == "finite_irreps":
            decompositions = [rep.decompose() for rep in representations]
            finite_reps = [irrep for decomposition in decompositions for irrep in decomposition.irreps]
            change_inv = torch.block_diag(
                *(torch.linalg.inv(decomposition.change_of_basis) for decomposition in decompositions)
            )
            self.out_type = FieldType(self.space, finite_reps)
        else:
            change_inv = torch.eye(sum(rep.dim for rep in representations), dtype=torch.float64)
            self.out_type = FieldType(self.space, representations)
        self.register_buffer("change_to_output_basis", change_inv, persistent=True)
        self.rep_out = self.out_type.representation

    def forward(self, vectors: torch.Tensor) -> GeometricTensor:
        if vectors.shape[-1] != 3 or not vectors.is_floating_point():
            raise ValueError("vectors must be a floating-point tensor with final dimension 3")
        values = [
            o3.spherical_harmonics(
                degree,
                vectors,
                normalize=self.normalize,
                normalization=self.normalization,
            )
            for degree in self.degrees
        ]
        values = torch.cat(values, dim=-1)
        change = self.change_to_output_basis.to(device=values.device, dtype=values.dtype)
        return GeometricTensor(values @ change.T, self.out_type)


def spherical_harmonics(
    group: FiniteGroup,
    degrees: int | list[int] | tuple[int, ...],
    vectors: torch.Tensor,
    *,
    normalize: bool = True,
    normalization: str = "component",
) -> torch.Tensor:
    """Functional restricted e3nn spherical harmonics."""
    return RestrictedSphericalHarmonics(
        group, degrees, normalize=normalize, normalization=normalization
    )(vectors).tensor
