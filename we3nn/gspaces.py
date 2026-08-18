"""No-base-space compatibility helpers matching the escnn example surface."""

from __future__ import annotations

from dataclasses import dataclass

from .groups import FiniteGroup, cyclic_group, dihedral_group


@dataclass(frozen=True)
class GSpace:
    fibergroup: FiniteGroup
    dimensionality: int = 0

    @property
    def fiber_group(self) -> FiniteGroup:
        return self.fibergroup

    def irrep(self, *irrep_id: int):
        return self.fibergroup.irrep(*irrep_id)

    @property
    def regular_repr(self):
        return self.fibergroup.regular_repr

    @property
    def trivial_repr(self):
        return self.fibergroup.trivial_representation

    @property
    def name(self) -> str:
        return f"{self.fibergroup.name} on R^0"


def no_base_space(group: FiniteGroup) -> GSpace:
    return GSpace(group)


def rot2dOnR2(N: int) -> GSpace:
    """Compatibility constructor; only its fiber action is modeled."""
    return GSpace(cyclic_group(N))


def flipRot2dOnR2(N: int) -> GSpace:
    """Compatibility constructor for the D_N fiber group."""
    return GSpace(dihedral_group(N))
