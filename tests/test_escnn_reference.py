"""Numerical convention checks, enabled by the optional ``reference`` extra."""

import numpy as np
import pytest

try:
    from escnn import gspaces as escnn_gspaces, nn as escnn_nn
except ImportError as error:
    pytest.skip(f"complete escnn reference installation unavailable: {error}", allow_module_level=True)

from e3nn_WE import cyclic_group, dihedral_group, gspaces, nn


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_cyclic_representation_matrices_match_escnn(n):
    reference = escnn_gspaces.rot2dOnR2(N=n).fibergroup
    ours = cyclic_group(n)
    for frequency in range(n // 2 + 1):
        for reference_element, element in zip(reference.elements, ours.elements):
            np.testing.assert_allclose(
                ours.irrep(frequency)(element).numpy(),
                reference.irrep(frequency)(reference_element),
                atol=1e-14,
            )
    for reference_element, element in zip(reference.elements, ours.elements):
        np.testing.assert_allclose(
            ours.regular_repr(element).numpy(),
            reference.regular_representation(reference_element),
            atol=1e-14,
        )


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_dihedral_representation_matrices_match_escnn(n):
    reference = escnn_gspaces.flipRot2dOnR2(N=n).fibergroup
    ours = dihedral_group(n)
    for irrep_id in ours.irrep_ids():
        for reference_element, element in zip(reference.elements, ours.elements):
            np.testing.assert_allclose(
                ours.irrep(*irrep_id)(element).numpy(),
                reference.irrep(*irrep_id)(reference_element),
                atol=1e-14,
            )
    for reference_element, element in zip(reference.elements, ours.elements):
        np.testing.assert_allclose(
            ours.regular_repr(element).numpy(),
            reference.regular_representation(reference_element),
            atol=1e-14,
        )


def test_d6_linear_has_same_number_of_trainable_degrees_of_freedom_as_escnn():
    esc_space = escnn_gspaces.no_base_space(escnn_gspaces.flipRot2dOnR2(N=6).fibergroup)
    our_space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    esc_in = escnn_nn.FieldType(esc_space, 10 * [esc_space.irrep(0, 0)] + 2 * [esc_space.irrep(1, 1)])
    our_in = nn.FieldType(our_space, 10 * [our_space.irrep(0, 0)] + 2 * [our_space.irrep(1, 1)])
    esc_out = escnn_nn.FieldType(esc_space, 4 * [esc_space.regular_repr])
    our_out = nn.FieldType(our_space, 4 * [our_space.regular_repr])
    esc_layer = escnn_nn.Linear(esc_in, esc_out, bias=True)
    our_layer = nn.Linear(our_in, our_out, bias=True)
    assert sum(parameter.numel() for parameter in our_layer.parameters()) == sum(
        parameter.numel() for parameter in esc_layer.parameters()
    )
