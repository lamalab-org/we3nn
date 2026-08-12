import pytest
import torch

from e3nn_WE import (
    cyclic_group,
    dihedral_group,
    invariant_basis,
    intertwiner_basis,
    subspace_distance,
)


@pytest.mark.parametrize("group", [cyclic_group(5), cyclic_group(6), dihedral_group(5), dihedral_group(6)])
def test_nullspace_reynolds_and_generators_agree(group):
    representations = [*group.irreps(), group.regular_repr]
    for rep_in in representations:
        for rep_out in representations:
            if rep_in.size * rep_out.size > 150:
                continue
            nullspace = intertwiner_basis(rep_in, rep_out, method="nullspace")
            reynolds = intertwiner_basis(rep_in, rep_out, method="reynolds")
            generators = intertwiner_basis(rep_in, rep_out, method="nullspace", elements="generators")
            assert nullspace.shape == reynolds.shape == generators.shape
            assert subspace_distance(nullspace, reynolds) < 2e-10
            assert subspace_distance(nullspace, generators) < 2e-10
            for element in group.elements:
                residual = rep_out(element) @ nullspace - nullspace @ rep_in(element)
                assert float(residual.abs().max()) < 2e-10 if residual.numel() else True


def test_invariant_basis_matches_trivial_bias_space():
    group = dihedral_group(6)
    basis = invariant_basis(group.regular_repr)
    assert basis.shape == (1, 12)
    for element in group.elements:
        torch.testing.assert_close(
            basis @ group.regular_repr(element).T, basis, atol=1e-12, rtol=1e-12
        )


def test_cg_nullspace_and_reynolds_projectors_agree():
    group = cyclic_group(7)
    for left in group.irreps():
        for right in group.irreps():
            for output in group.irreps():
                from e3nn_WE import clebsch_gordan

                nullspace = clebsch_gordan(left, right, output, method="nullspace")
                reynolds = clebsch_gordan(left, right, output, method="reynolds")
                assert subspace_distance(nullspace, reynolds) < 2e-10
