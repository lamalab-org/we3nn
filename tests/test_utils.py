import torch

from we3nn.utils import edge_radial_basis, edge_wa_joint_features, scatter_sum


def test_scatter_sum_matches_manual_and_has_gradients():
    source = torch.randn(7, 3, requires_grad=True)
    index = torch.tensor([2, 0, 2, 1, 0, 2, 1])
    output = scatter_sum(source, index, dim_size=4)
    manual = torch.stack(
        [source[index == i].sum(0) if (index == i).any() else torch.zeros(3) for i in range(4)]
    )
    torch.testing.assert_close(output, manual)
    output.square().sum().backward()
    assert source.grad is not None


def test_radial_and_joint_features_are_o2_invariant():
    edge = torch.randn(13, 2, dtype=torch.float64)
    anchor = torch.randn(13, 2, dtype=torch.float64)
    angle = torch.tensor(0.73, dtype=torch.float64)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float64,
    )
    reflection = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    for matrix in (rotation, reflection):
        transformed_edge = edge @ matrix.T
        transformed_anchor = anchor @ matrix.T
        torch.testing.assert_close(
            edge_radial_basis(transformed_edge, basis_size=8, max_radius=6.0),
            edge_radial_basis(edge, basis_size=8, max_radius=6.0),
        )
        torch.testing.assert_close(
            edge_wa_joint_features(transformed_edge, transformed_anchor),
            edge_wa_joint_features(edge, anchor),
        )
