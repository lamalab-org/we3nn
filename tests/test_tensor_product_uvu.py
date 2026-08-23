import pytest
import torch

from we3nn import gspaces, nn
from we3nn.clebsch_gordan import full_coupling_basis


def _space(kind: str, n: int):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    return gspaces.no_base_space(base.fibergroup)


def _e(space, frequency: int):
    group = space.fibergroup
    return group.irrep(frequency) if group.name.startswith("C") else group.irrep(1, frequency)


def _explicit_reference(grouped: nn.TensorProduct) -> nn.TensorProduct:
    paths = [(path.i_in1, path.i_in2, path.i_out) for path in grouped.instructions]
    return nn.TensorProduct(
        grouped.in1_type,
        grouped.in2_type,
        grouped.out_type,
        instructions=paths,
        internal_weights=False,
        shared_weights=grouped.shared_weights,
    ).double()


def _compare_external_with_reference(grouped, left, right, weights):
    reference = _explicit_reference(grouped)
    left_ref = left.detach().clone().requires_grad_()
    right_ref = right.detach().clone().requires_grad_()
    weight_ref = weights.detach().clone().requires_grad_()

    actual = grouped(left, right, weights)
    expected = reference(left_ref, right_ref, weight_ref)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for new, old in ((left, left_ref), (right, right_ref), (weights, weight_ref)):
        torch.testing.assert_close(new.grad, old.grad, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize(
    "kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)]
)
@pytest.mark.parametrize(
    "triple",
    [
        ("scalar", "scalar", "scalar"),
        ("scalar", "e1", "e1"),
        ("e1", "e1", "scalar"),
        ("e1", "e1", "e2"),
    ],
)
def test_uvu_cg_matches_explicit_field_paths_and_gradients(kind, n, triple):
    torch.manual_seed(101)
    space = _space(kind, n)
    reps = {"scalar": space.trivial_repr, "e1": _e(space, 1), "e2": _e(space, 2)}
    left_rep, right_rep, output_rep = (reps[name] for name in triple)
    left_type = nn.FieldType(space, [left_rep] * 3)
    right_type = nn.FieldType(space, [right_rep] * 2)
    output_type = nn.FieldType(space, [output_rep] * 3)
    grouped = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=False,
    ).double()
    coupling = full_coupling_basis(left_rep, right_rep, output_rep)
    assert grouped.weight_numel == 3 * 2 * coupling.shape[0]

    left = torch.randn(4, left_type.size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, right_type.size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(4, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


@pytest.mark.parametrize("shared_weights", [True, False])
def test_uvu_shared_and_per_sample_external_weights(shared_weights):
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    left_type = nn.FieldType(space, [e1] * 3)
    right_type = nn.FieldType(space, [e1] * 2)
    output_type = nn.FieldType(space, [e2] * 3)
    grouped = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=shared_weights,
    ).double()
    left = torch.randn(5, left_type.size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(5, right_type.size, dtype=torch.float64, requires_grad=True)
    shape = (grouped.weight_numel,) if shared_weights else (5, grouped.weight_numel)
    weights = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


def test_uvu_internal_weights_match_explicit_reference_and_gradients():
    torch.manual_seed(103)
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    types = (
        nn.FieldType(space, [e1] * 3),
        nn.FieldType(space, [e1] * 2),
        nn.FieldType(space, [e2] * 3),
    )
    grouped = nn.TensorProduct(*types, connection_mode="uvu").double()
    reference = nn.TensorProduct(
        *types,
        instructions=[(p.i_in1, p.i_in2, p.i_out) for p in grouped.instructions],
    ).double()
    flattened = torch.cat([block.weight.detach().reshape(-1) for block in grouped.blocks])
    offset = 0
    with torch.no_grad():
        for path in reference.paths:
            count = path.weight_numel
            path.weight.copy_(flattened[offset : offset + count].reshape(path.weight_shape))
            offset += count
    assert offset == grouped.weight_numel

    left = torch.randn(4, types[0].size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, types[1].size, dtype=torch.float64, requires_grad=True)
    left_ref = left.detach().clone().requires_grad_()
    right_ref = right.detach().clone().requires_grad_()
    actual, expected = grouped(left, right), reference(left_ref, right_ref)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(left.grad, left_ref.grad, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(right.grad, right_ref.grad, atol=3e-11, rtol=3e-11)
    grouped_grads = torch.cat([block.weight.grad.reshape(-1) for block in grouped.blocks])
    reference_grads = torch.cat([path.weight.grad.reshape(-1) for path in reference.paths])
    torch.testing.assert_close(grouped_grads, reference_grads, atol=3e-11, rtol=3e-11)


def test_uvu_weight_count_and_single_grouped_block():
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    left = nn.FieldType(space, [e1] * 8)
    right = nn.FieldType(space, [e1] * 3)
    output = nn.FieldType(space, [e2] * 8)
    p = full_coupling_basis(e1, e1, e2).shape[0]
    uvu = nn.TensorProduct(left, right, output, connection_mode="uvu")
    uvw = nn.TensorProduct(left, right, output)
    assert uvu.weight_numel == 8 * 3 * p
    assert uvw.weight_numel == 8 * 3 * 8 * p
    assert len(uvu.blocks) == 1
    assert len(uvu.paths) == 0
    assert len(uvu.instructions) == 8 * 3


def test_uvu_api_validation_and_fully_connected_remains_uvw():
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    left = nn.FieldType(space, [e1] * 3)
    right = nn.FieldType(space, [e1] * 2)
    output = nn.FieldType(space, [e2] * 2)
    with pytest.raises(ValueError, match="matching left/output multiplicities"):
        nn.TensorProduct(left, right, output, connection_mode="uvu")
    with pytest.raises(ValueError, match="must be 'uvw' or 'uvu'"):
        nn.TensorProduct(left, right, output, connection_mode="bad")
    with pytest.raises(ValueError, match="explicit instructions"):
        nn.TensorProduct(left, right, output, [(0, 0, 0)], connection_mode="uvu")
    with pytest.raises(TypeError):
        nn.FullyConnectedTensorProduct(left, right, output, connection_mode="uvu")
    assert nn.FullyConnectedTensorProduct(left, right, output).connection_mode == "uvw"


def test_uvu_noncontiguous_occurrences_pair_in_group_encounter_order():
    space = _space("dihedral", 6)
    scalar, e1, e2 = space.trivial_repr, _e(space, 1), _e(space, 2)
    left_type = nn.FieldType(space, [e1, scalar, e1, e1])
    right_type = nn.FieldType(space, [e1, e1])
    output_type = nn.FieldType(space, [e2, e2, e2])
    grouped = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=False,
    ).double()
    logical = [(path.i_in1, path.i_in2, path.i_out) for path in grouped.instructions]
    assert logical == [
        (0, 0, 0),
        (0, 1, 0),
        (2, 0, 1),
        (2, 1, 1),
        (3, 0, 2),
        (3, 1, 2),
    ]
    left = torch.randn(2, left_type.size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, right_type.size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


def test_uvu_memory_bounded_cg_path_matches_reference(monkeypatch):
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    types = (
        nn.FieldType(space, [e1] * 4),
        nn.FieldType(space, [e1] * 3),
        nn.FieldType(space, [e2] * 4),
    )
    grouped = nn.TensorProduct(
        *types,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=False,
    ).double()
    tensor_product_module = __import__(
        "we3nn.nn.tensor_product", fromlist=["_CG_MAX_INTERMEDIATE_BYTES"]
    )
    monkeypatch.setattr(tensor_product_module, "_CG_MAX_INTERMEDIATE_BYTES", 8)
    left = torch.randn(3, types[0].size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


@pytest.mark.parametrize(
    "kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)]
)
def test_uvu_cg_equivariance(kind, n):
    space = _space(kind, n)
    e1, e2 = _e(space, 1), _e(space, 2)
    left_type = nn.FieldType(space, [e1] * 3)
    right_type = nn.FieldType(space, [e1] * 2)
    output_type = nn.FieldType(space, [e2] * 3)
    module = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        connection_mode="uvu",
        internal_weights=False,
    ).double()
    left = torch.randn(2, left_type.size, dtype=torch.float64)
    right = torch.randn(2, right_type.size, dtype=torch.float64)
    weights = torch.randn(module.weight_numel, dtype=torch.float64)
    output = module(left, right, weights)
    for element in space.fibergroup.elements:
        actual = module(
            left_type.transform_fibers(left, element),
            right_type.transform_fibers(right, element),
            weights,
        )
        expected = output_type.transform_fibers(output, element)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
