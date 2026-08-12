"""Finite cyclic and dihedral equivariance with an e3nn-like API."""

from . import gspaces, nn
from .groups import CyclicGroup, DihedralGroup, FiniteGroup, GroupElement, cyclic_group, dihedral_group
from .representations import Irrep, Irreps, Representation, direct_sum

__all__ = [
    "CyclicGroup",
    "DihedralGroup",
    "FiniteGroup",
    "GroupElement",
    "Irrep",
    "Irreps",
    "Representation",
    "cyclic_group",
    "dihedral_group",
    "direct_sum",
    "gspaces",
    "nn",
]

__version__ = "0.1.0"
