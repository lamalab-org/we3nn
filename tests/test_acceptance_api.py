import torch
import importlib
import pytest

from we3nn import DihedralGroup, nn


def test_plan_script_raw_tensor_acceptance_api():
    group = DihedralGroup(6)
    assert group.order == 12 and group.order() == 12
    scalar = group.trivial_irrep
    vector = group.standard_representation()
    regular = group.regular_representation()
    rep_in = 3 * scalar + vector + vector
    rep_hidden = 2 * regular
    linear = nn.Linear(rep_in, rep_hidden)
    activation = nn.PointwiseActivation(rep_hidden, torch.relu)
    x = torch.randn(7, rep_in.dim)
    y = activation(linear(x))
    assert y.shape == (7, rep_hidden.dim)
    for element in group.elements:
        torch.testing.assert_close(
            activation(linear(x @ rep_in.matrix(element, dtype=x.dtype).T)),
            y @ rep_hidden.matrix(element, dtype=y.dtype).T,
            atol=2e-5,
            rtol=2e-5,
        )


def test_raw_tensor_tensor_product_api():
    group = DihedralGroup(5)
    vector = group.standard_representation()
    scalar = group.trivial_irrep
    product = nn.TensorProduct(vector, vector, scalar)
    x, y = torch.randn(4, 2), torch.randn(4, 2)
    assert product(x, y).shape == (4, 1)


def test_e3nn_group_namespace_and_upstream_o3_coexist():
    from e3nn import o3
    from we3nn import group

    finite = group.DihedralGroup(6)
    assert finite.order == 12
    assert o3.Irrep("1o").dim == 3
    value = group.nn.Linear(finite.trivial_irrep, finite.regular_representation())(
        torch.randn(3, 1)
    )
    assert value.shape == (3, 12)


def test_full_tensor_product_raw_api_and_instruction_metadata():
    group = DihedralGroup(4)
    vector = group.standard_representation()
    full = nn.FullTensorProduct(vector, vector)
    x, y = torch.randn(5, 2), torch.randn(5, 2)
    torch.testing.assert_close(full(x, y), torch.einsum("...i,...j->...ij", x, y).flatten(-2))
    product = nn.TensorProduct(vector, vector, group.trivial_irrep)
    instruction = product.instructions[0]
    assert instruction.connection_mode == "uvw"
    assert instruction.has_weight and instruction.coupling is None


def test_explicit_full_finite_coupling_api():
    from we3nn import group as finite

    group = finite.CyclicGroup(5)
    public = finite.clebsch_gordan(group.irrep(1), group.irrep(1), group.trivial_irrep)
    full = finite.finite_group_couplings(group.irrep(1), group.irrep(1), group.trivial_irrep)
    multiplicity = finite.tensor_product_multiplicity(
        group.irrep(1), group.irrep(1), group.trivial_irrep
    )
    assert multiplicity == public.shape[0]
    assert full.shape[0] >= public.shape[0]


def test_tensor_wrapper_is_not_part_of_library():
    from we3nn import group

    assert not hasattr(group.nn, "GeometricTensor")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("we3nn.nn.geometric_tensor")
