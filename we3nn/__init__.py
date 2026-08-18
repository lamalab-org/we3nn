"""Finite cyclic and dihedral equivariance with an e3nn-like API."""

from . import embeddings, gspaces, nn
from .clebsch_gordan import (
    clebsch_gordan,
    coupling_dimension,
    finite_group_couplings,
    tensor_product_multiplicity,
)
from .groups import CyclicGroup, DihedralGroup, FiniteGroup, Group, GroupElement, cyclic_group, dihedral_group
from .harmonics import CircularHarmonics, RestrictedSphericalHarmonics, circular_harmonics, spherical_harmonics
from .matrix_group import MatrixFiniteGroup
from .intertwiner import (
    TensorProductRepresentation,
    invariant_basis,
    find_representation_intertwiner,
    intertwiner_basis,
    subspace_diagnostics,
    subspace_distance,
    tensor_product_representation,
)
from .embedding import (
    IrrepDecomposition,
    O3Embedding,
    RestrictedO3Representation,
    planar_o3,
    restrict_o3,
    restrict_o3_irrep,
    restricted_o3_couplings,
)
from .representations import DirectSum, DirectSumRepresentation, Irrep, Irreps, RepBlock, Representation, direct_sum
from .nn import PointActiv, RestrictedWETensorProduct, WELinear, WETensorProduct

__all__ = [
    "CyclicGroup",
    "CircularHarmonics",
    "DihedralGroup",
    "DirectSumRepresentation",
    "DirectSum",
    "FiniteGroup",
    "GroupElement",
    "Group",
    "Irrep",
    "Irreps",
    "MatrixFiniteGroup",
    "O3Embedding",
    "PointActiv",
    "RepBlock",
    "Representation",
    "RestrictedSphericalHarmonics",
    "RestrictedWETensorProduct",
    "RestrictedO3Representation",
    "IrrepDecomposition",
    "TensorProductRepresentation",
    "WELinear",
    "WETensorProduct",
    "cyclic_group",
    "circular_harmonics",
    "clebsch_gordan",
    "coupling_dimension",
    "finite_group_couplings",
    "tensor_product_multiplicity",
    "dihedral_group",
    "direct_sum",
    "embeddings",
    "intertwiner_basis",
    "invariant_basis",
    "find_representation_intertwiner",
    "gspaces",
    "nn",
    "planar_o3",
    "restrict_o3",
    "restrict_o3_irrep",
    "restricted_o3_couplings",
    "subspace_diagnostics",
    "subspace_distance",
    "spherical_harmonics",
    "tensor_product_representation",
]

__version__ = "0.4.0"
