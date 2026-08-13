import torch

from mp_example import EquivariantMPLayer
from e3nn_WE.utils import scatter


def test_actual_message_passing_layer_forward_backward_and_d6_equivariance():
    torch.manual_seed(7)
    nodes, in_channels, hidden_channels = 9, 5, 8
    layer = EquivariantMPLayer(
        in_channels,
        hidden_channels,
        torch.nn.ReLU(),
        joint_edge_wa_features=True,
    ).double()
    h = torch.randn(nodes, in_channels, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(nodes, 3, dtype=torch.float64, requires_grad=True)
    anchor = torch.randn(nodes, 2, dtype=torch.float64, requires_grad=True)
    row = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8], dtype=torch.long)
    col = torch.tensor([1, 2, 2, 0, 4, 5, 3, 6, 7, 8, 6, 1], dtype=torch.long)
    edge_index = torch.stack((row, col))

    scalar_output, forces = layer(h, pos, anchor, edge_index)
    assert scalar_output.shape == (nodes, hidden_channels)
    assert forces.shape == (nodes, 3)
    (scalar_output.square().mean() + forces.square().mean()).backward()
    assert h.grad is not None and pos.grad is not None and anchor.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())

    with torch.no_grad():
        for element in layer.group.elements:
            matrix = layer.group.standard_representation()(element).to(dtype=torch.float64)
            transformed_pos = pos.detach().clone()
            transformed_pos[:, :2] = pos.detach()[:, :2] @ matrix.T
            transformed_anchor = anchor.detach() @ matrix.T
            transformed_scalars, transformed_forces = layer(
                h.detach(), transformed_pos, transformed_anchor, edge_index
            )
            expected_forces = forces.detach().clone()
            expected_forces[:, :2] = forces.detach()[:, :2] @ matrix.T
            torch.testing.assert_close(
                transformed_scalars, scalar_output.detach(), atol=2e-11, rtol=2e-11
            )
            torch.testing.assert_close(
                transformed_forces, expected_forces, atol=2e-11, rtol=2e-11
            )


def test_message_layer_preserves_supplied_row_force_col_feature_scatter_convention():
    torch.manual_seed(12)
    layer = EquivariantMPLayer(2, 4, torch.nn.ReLU()).double()
    h = torch.randn(5, 2, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)
    anchor = torch.randn(5, 2, dtype=torch.float64)
    edge_index = torch.tensor([[0, 0, 2, 3, 4], [1, 2, 4, 1, 0]])
    row, col = edge_index
    r_ij, rsq, radial = layer.compute_distances(pos, edge_index)
    message = layer.node_message_function(
        h[row], h[col], rsq, radial, anchor[row], r_ij[:, :2], r_ij[:, 2:3]
    )
    edge_h = layer.enn_scalar_out(message)
    edge_force = torch.cat(
        (layer.enn_vector_out(message), layer.enn_z_out(message)), dim=-1
    )
    actual_h, actual_force = layer(h, pos, anchor, edge_index)
    torch.testing.assert_close(actual_force, scatter(edge_force, row, dim=0, reduce="sum"))
    torch.testing.assert_close(
        actual_h,
        scatter(edge_h, col, dim=0, reduce="sum") + layer.residual_proj(h),
    )
