import math

import pytest
import torch

from we3nn import CircularHarmonics, RestrictedSphericalHarmonics, cyclic_group, dihedral_group, planar_o3


def _transform_angles(group, angles, element):
    if isinstance(element.value, int):
        return angles + 2.0 * math.pi * element.value / group.n
    flip, rotation = element.value
    alpha = 2.0 * math.pi * rotation / group.n
    return alpha - angles if flip else alpha + angles


@pytest.mark.parametrize("group", [cyclic_group(3), cyclic_group(4), cyclic_group(7), dihedral_group(3), dihedral_group(4), dihedral_group(6)])
@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
def test_circular_harmonics_are_equivariant(group, normalization):
    module = CircularHarmonics(group, normalization=normalization)
    angles = torch.linspace(-2.7, 2.8, 23, dtype=torch.float64)
    output = module(angles)
    for element in group.elements:
        actual = module(_transform_angles(group, angles, element))
        expected = module.out_type.transform_fibers(output, element)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_harmonics_from_vectors_and_gradients():
    module = CircularHarmonics(dihedral_group(7), max_frequency=3)
    vectors = torch.randn(17, 2, dtype=torch.float64, requires_grad=True)
    from_vectors = module.from_vectors(vectors)
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])
    torch.testing.assert_close(from_vectors, module(angles))
    from_vectors.square().sum().backward()
    assert vectors.grad is not None
    assert torch.isfinite(vectors.grad).all()


def test_even_group_nyquist_layouts():
    cyclic = CircularHarmonics(cyclic_group(6))
    dihedral = CircularHarmonics(dihedral_group(6))
    assert [rep.id for rep in cyclic.out_type][-2:] == [(3,), (3,)]
    assert [rep.id for rep in dihedral.out_type][-2:] == [(0, 3), (1, 3)]


def test_invalid_aliased_frequency_is_rejected():
    with pytest.raises(ValueError, match="between"):
        CircularHarmonics(dihedral_group(6), max_frequency=4)


@pytest.mark.parametrize("group", [cyclic_group(5), dihedral_group(6)])
def test_restricted_e3nn_spherical_harmonics_are_equivariant(group):
    module = RestrictedSphericalHarmonics(group, [0, 1, 2, 3])
    vectors = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    output = module(vectors)
    for element in group.elements:
        planar = group.standard_representation(element)
        matrix = torch.eye(3, dtype=torch.float64)
        matrix[:2, :2] = planar
        actual = module(vectors.detach() @ matrix.T)
        expected = module.out_type.transform_fibers(output, element).detach()
        # e3nn's Wigner-D construction uses angle extraction with numerical
        # tolerances; its float64 restricted action is accurate to a few e-6.
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    output.square().mean().backward()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()


def test_restricted_spherical_harmonics_finite_irrep_basis():
    group = dihedral_group(6)
    module = RestrictedSphericalHarmonics(group, [0, 1, 2, 3], basis="finite_irreps")
    vectors = torch.randn(9, 3, dtype=torch.float64)
    output = module(vectors)
    assert all(rep.is_irreducible for rep in module.out_type)
    embedding = planar_o3(group)
    for element in group.elements:
        actual = module(vectors @ embedding.matrix(element).T)
        expected = module.out_type.transform_fibers(output, element)
        torch.testing.assert_close(actual, expected, atol=8e-6, rtol=8e-6)
