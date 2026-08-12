import torch

from e3nn_WE import gspaces, nn


def test_d6_message_mlp_pattern_is_drop_in_and_equivariant():
    in_channels, hidden_channels, radial_basis_size = 8, 12, 16
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    scalar_inputs = 2 * in_channels + 1 + radial_basis_size + 1
    c_in = nn.FieldType(space, scalar_inputs * [space.irrep(0, 0)] + 2 * [space.irrep(1, 1)])
    c_out = nn.FieldType(space, (hidden_channels // 2) * [space.regular_repr])
    scalar_out = nn.FieldType(space, hidden_channels * [space.irrep(0, 0)])
    vector_out = nn.FieldType(space, [space.irrep(1, 1)])
    message_mlp = nn.SequentialModule(
        nn.Linear(c_in, c_out), nn.ReLU(c_out), nn.Linear(c_out, c_out), nn.ReLU(c_out)
    )
    scalar_head = nn.Linear(c_out, scalar_out)
    vector_head = nn.SequentialModule(nn.Linear(c_out, c_out), nn.ELU(c_out), nn.Linear(c_out, vector_out))
    x = nn.GeometricTensor(torch.randn(23, c_in.size), c_in)
    message = message_mlp(x)
    assert scalar_head(message).tensor.shape == (23, hidden_channels)
    assert vector_head(message).tensor.shape == (23, 2)
    for element in space.fibergroup.elements:
        transformed_message = message_mlp(x.transform_fibers(element))
        torch.testing.assert_close(
            vector_head(transformed_message).tensor,
            vector_head(message).transform_fibers(element).tensor,
            atol=2e-5,
            rtol=2e-5,
        )
