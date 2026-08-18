import torch

from we3nn import DihedralGroup, nn


def test_d6_message_mlp_pattern_is_drop_in_and_equivariant():
    in_channels, hidden_channels, radial_basis_size = 8, 12, 16
    group = DihedralGroup(6)
    scalar_inputs = 2 * in_channels + 1 + radial_basis_size + 1
    c_in = scalar_inputs * group.trivial_irrep + 2 * group.standard_representation()
    c_out = (hidden_channels // 2) * group.regular_representation()
    scalar_out = hidden_channels * group.trivial_irrep
    vector_out = group.standard_representation()
    message_mlp = torch.nn.Sequential(
        nn.WELinear(c_in, c_out), nn.PointActiv(c_out, torch.relu),
        nn.WELinear(c_out, c_out), nn.PointActiv(c_out, torch.relu)
    )
    scalar_head = nn.WELinear(c_out, scalar_out)
    vector_head = torch.nn.Sequential(
        nn.WELinear(c_out, c_out), nn.PointActiv(c_out, torch.nn.functional.elu), nn.WELinear(c_out, vector_out)
    )
    x = torch.randn(23, c_in.size)
    message = message_mlp(x)
    assert scalar_head(message).shape == (23, hidden_channels)
    assert vector_head(message).shape == (23, 2)
    for element in group.elements:
        transformed_message = message_mlp(x @ c_in.matrix(element, dtype=x.dtype).T)
        torch.testing.assert_close(
            vector_head(transformed_message),
            vector_head(message) @ vector_out.matrix(element, dtype=x.dtype).T,
            atol=2e-5,
            rtol=2e-5,
        )
