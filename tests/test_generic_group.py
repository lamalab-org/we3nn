import pytest
import torch

from e3nn_WE import DirectSumRepresentation, MatrixFiniteGroup


def klein_four_group():
    values = ("e", "a", "b", "c")
    table = [[i ^ j for j in range(4)] for i in range(4)]
    return MatrixFiniteGroup(values, table, [0, 1, 2, 3], 0, name="V4", generators=[1, 2])


def test_table_group_regular_user_representation_and_validation():
    group = klein_four_group()
    assert group.order() == 4
    assert tuple(element.value for element in group.generators) == ("a", "b")
    regular = group.regular_repr
    assert regular.check_representation()
    matrices = {
        "e": torch.eye(2),
        "a": torch.diag(torch.tensor([-1.0, 1.0])),
        "b": torch.diag(torch.tensor([1.0, -1.0])),
        "c": -torch.eye(2),
    }
    supplied = group.representation(matrices, name="two signs")
    assert supplied.dim == 2 and supplied.check_representation()


def test_direct_sum_arithmetic_and_metadata():
    group = klein_four_group()
    trivial = group.trivial_representation
    regular = group.regular_repr
    summed = 3 * trivial + 2 * regular
    assert isinstance(summed, DirectSumRepresentation)
    assert summed.dim == 11
    assert [block.multiplicity for block in summed.blocks] == [3, 2]
    assert summed.check_representation()
    assert summed.is_permutation and summed.pointwise_action == "permutation"


def test_invalid_table_is_rejected():
    with pytest.raises(ValueError):
        MatrixFiniteGroup((0, 1), [[0, 1], [1, 1]], [0, 1], 0)


def test_arbitrary_supplied_representations_support_linear_cg_and_tensor_product():
    from e3nn_WE import clebsch_gordan, nn

    group = klein_four_group()
    matrices = {
        "e": torch.eye(2),
        "a": torch.diag(torch.tensor([-1.0, 1.0])),
        "b": torch.diag(torch.tensor([1.0, -1.0])),
        "c": -torch.eye(2),
    }
    supplied = group.representation(matrices, name="two signs")
    regular = group.regular_repr
    linear = nn.Linear(supplied, regular)
    x = torch.randn(5, supplied.dim)
    y = linear(x)
    product = nn.TensorProduct(supplied, supplied, group.trivial_representation)
    scalar = product(x, x)
    assert y.shape == (5, 4) and scalar.shape == (5, 1)
    assert clebsch_gordan(supplied, supplied, group.trivial_representation).shape[1:] == (1, 2, 2)
    for element in group.elements:
        torch.testing.assert_close(
            linear(x @ supplied.matrix(element, dtype=x.dtype).T),
            y @ regular.matrix(element, dtype=y.dtype).T,
            atol=2e-5,
            rtol=2e-5,
        )
