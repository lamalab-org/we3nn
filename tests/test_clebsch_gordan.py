import itertools

import pytest
import torch

from we3nn import (
    clebsch_gordan,
    cyclic_group,
    dihedral_group,
    finite_group_couplings,
    tensor_product_multiplicity,
)


@pytest.mark.parametrize(
    "group",
    [cyclic_group(3), cyclic_group(4), cyclic_group(7), dihedral_group(3), dihedral_group(4), dihedral_group(6)],
)
def test_cg_coefficients_are_orthonormal_complete_and_equivariant(group):
    for left, right, output in itertools.product(group.irreps(), repeat=3):
        coefficients = clebsch_gordan(left, right, output)
        gram = coefficients.flatten(1)
        torch.testing.assert_close(
            gram @ gram.T,
            torch.eye(coefficients.shape[0], dtype=torch.float64),
            atol=2e-12,
            rtol=2e-12,
        )
        decomposition = {irrep_id: multiplicity for multiplicity, irrep_id in group.tensor_product(left, right)}
        expected_paths = decomposition.get(output.id, 0)
        assert coefficients.shape[0] == expected_paths
        flattened = coefficients.flatten(2)
        copy_gram = torch.einsum("poi,qji->pqoj", flattened, flattened)
        expected_copy_gram = torch.eye(
            coefficients.shape[0], dtype=torch.float64
        )[:, :, None, None] * torch.eye(output.size, dtype=torch.float64)[None, None] / output.size
        torch.testing.assert_close(copy_gram, expected_copy_gram, atol=2e-12, rtol=2e-12)
        for element in group.elements:
            transformed = torch.einsum(
                "oa,pabc,ib,jc->poij",
                output(element),
                coefficients,
                left(element),
                right(element),
            )
            torch.testing.assert_close(transformed, coefficients, atol=2e-12, rtol=2e-12)


def test_cyclic_complex_type_reports_one_copy_but_tensor_product_has_two_real_weights():
    group = cyclic_group(7)
    vector = group.irrep(1)
    scalar = group.irrep(0)
    assert clebsch_gordan(vector, scalar, vector).shape == (1, 2, 2, 1)
    from we3nn.clebsch_gordan import full_coupling_basis

    assert full_coupling_basis(vector, scalar, vector).shape == (2, 2, 2, 1)
    assert tensor_product_multiplicity(vector, scalar, vector) == 1
    assert finite_group_couplings(vector, scalar, vector).shape[0] == 2


def test_cg_rejects_groups_with_equal_names_but_distinct_identity():
    left_group = cyclic_group(5)
    from we3nn import CyclicGroup

    right_group = CyclicGroup(5)
    with pytest.raises(ValueError, match="same group"):
        clebsch_gordan(left_group.irrep(1), right_group.irrep(1), left_group.irrep(0))
