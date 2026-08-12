import itertools

import pytest
import torch

from e3nn_WE import CyclicGroup, DihedralGroup, Irreps, cyclic_group, dihedral_group


@pytest.mark.parametrize("factory,n", [(cyclic_group, 1), (cyclic_group, 2), (cyclic_group, 7), (dihedral_group, 1), (dihedral_group, 2), (dihedral_group, 6)])
def test_group_axioms(factory, n):
    group = factory(n)
    assert group.order() == n * (2 if isinstance(group, DihedralGroup) else 1)
    for a, b in itertools.product(group.elements, repeat=2):
        assert a * group.identity == a
        assert group.identity * a == a
        assert a * a.inverse() == group.identity
        assert (a * b).group is group
    for a, b, c in itertools.product(group.elements, repeat=3):
        assert (a * b) * c == a * (b * c)


@pytest.mark.parametrize("group", [cyclic_group(3), cyclic_group(4), cyclic_group(7), dihedral_group(3), dihedral_group(4), dihedral_group(6)])
def test_irreps_are_orthogonal_homomorphisms_and_complete(group):
    assert sum(irrep.size**2 / irrep.sum_of_squares_constituents for irrep in group.irreps()) == group.order()
    for irrep in group.irreps():
        for a, b in itertools.product(group.elements, repeat=2):
            matrix = irrep(a)
            assert torch.allclose(matrix.T @ matrix, torch.eye(irrep.size, dtype=torch.float64), atol=1e-12)
            assert torch.allclose(irrep(a * b), irrep(a) @ irrep(b), atol=1e-12)


@pytest.mark.parametrize("group", [cyclic_group(7), dihedral_group(6)])
def test_regular_is_a_permutation_representation(group):
    regular = group.regular_repr
    for element in group.elements:
        matrix = regular(element)
        assert torch.equal(matrix.sum(0), torch.ones(group.order(), dtype=torch.float64))
        assert torch.equal(matrix.sum(1), torch.ones(group.order(), dtype=torch.float64))
        assert torch.equal(matrix @ matrix.T, torch.eye(group.order(), dtype=torch.float64))


def test_standard_representations_have_expected_rotation_and_reflection():
    c6 = cyclic_group(6)
    d6 = dihedral_group(6)
    angle = torch.tensor(torch.pi / 3, dtype=torch.float64)
    expected_rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float64,
    )
    assert torch.allclose(c6.standard_representation(c6.element(1)), expected_rotation)
    assert torch.allclose(d6.standard_representation(d6.element((0, 1))), expected_rotation)
    assert torch.equal(
        d6.standard_representation(d6.element((1, 0))),
        torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
    )


@pytest.mark.parametrize("group", [cyclic_group(1), cyclic_group(2), dihedral_group(1), dihedral_group(2)])
def test_small_order_planar_standard_representation(group):
    standard = group.standard_representation
    assert standard.size == 2
    for a, b in itertools.product(group.elements, repeat=2):
        assert torch.allclose(standard(a * b), standard(a) @ standard(b), atol=1e-12)


def test_tensor_product_and_irreps_container():
    group = cyclic_group(5)
    decomposition = group.tensor_product(group.irrep(1), group.irrep(1))
    assert decomposition == [(2, (0,)), (1, (2,))]
    irreps = Irreps(group, [(2, (0,)), (1, (1,)), (2, group.irrep(1))])
    assert irreps.dim == 8
    assert irreps.simplify().dim == irreps.dim
    assert irreps.regroup().dim == irreps.dim


def test_cached_factories_preserve_group_identity_required_by_types():
    assert cyclic_group(8) is cyclic_group(8)
    assert dihedral_group(8) is dihedral_group(8)
    assert CyclicGroup(3) is not CyclicGroup(3)
