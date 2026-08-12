import torch

from e3nn_WE import DihedralGroup, nn


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
