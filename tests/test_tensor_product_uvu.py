import pytest
import torch

from we3nn import RestrictedSphericalHarmonics, RepresentationTensor, gspaces, nn
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


def _configured_explicit_reference(grouped: nn.TensorProduct) -> nn.TensorProduct:
    instructions = [
        nn.TensorProductInstruction(
            path.i_in1,
            path.i_in2,
            path.i_out,
            coupling=path.coupling,
            has_weight=path.has_weight,
            path_weight=path.path_weight,
        )
        for path in grouped.instructions
    ]
    return nn.TensorProduct(
        grouped.in1_type,
        grouped.in2_type,
        grouped.out_type,
        instructions=instructions,
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


def test_multiplicity_chunks_are_stable_and_include_noncontiguous_occurrences():
    space = _space("dihedral", 6)
    scalar, e1, e2 = space.trivial_repr, _e(space, 1), _e(space, 2)
    left_type = nn.FieldType(space, [e1, scalar, e1, e2, e1])
    right_type = nn.FieldType(space, [scalar])
    output_type = nn.FieldType(space, [e1, e1, scalar])

    product = nn.TensorProduct(left_type, right_type, output_type)

    assert [chunk.representation for chunk in product.in1_chunks] == [e1, scalar, e2]
    assert product.in1_chunks[0].field_indices == (0, 2, 4)
    assert product.in1_chunks[0].coordinate_starts == (0, 3, 7)
    assert product.in1_chunks[0].multiplicity == 3
    assert product.in2_chunks[0].field_indices == (0,)
    assert [chunk.multiplicity for chunk in product.out_chunks] == [2, 1]


def test_block_instruction_and_weight_layout_are_public_value_types():
    instruction = nn.TensorProductBlockInstruction(1, 2, 3, connection_mode="uvu")
    layout = nn.TensorProductWeightLayout(0, "uvu", (4, 2, 1), slice(3, 11))

    assert instruction.i_in1 == 1
    assert instruction.connection_mode == "uvu"
    assert layout.shape == (4, 2, 1)
    assert layout.numel == 8
    with pytest.raises(AttributeError):
        instruction.connection_mode = "uvw"


def _mixed_types(space):
    scalar, e1 = space.trivial_repr, _e(space, 1)
    return (
        nn.FieldType(space, [e1, scalar, e1, scalar, e1]),
        nn.FieldType(space, [scalar, e1, scalar]),
        nn.FieldType(space, [e1, scalar, e1, e1]),
    )


def _mixed_block_instructions():
    return (
        nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(1, 1, 0, connection_mode="uvw"),
        nn.TensorProductBlockInstruction(0, 1, 1, connection_mode="uvw"),
    )


def test_mixed_block_instructions_match_explicit_paths_and_weight_layout():
    torch.manual_seed(113)
    space = _space("dihedral", 6)
    types = _mixed_types(space)
    grouped = nn.TensorProduct(
        *types,
        block_instructions=_mixed_block_instructions(),
        internal_weights=False,
        shared_weights=False,
    ).double()

    assert grouped.connection_mode is None
    assert len(grouped.paths) == 0
    assert len(grouped.blocks) == 3
    assert [block.connection_mode for block in grouped.blocks] == ["uvu", "uvw", "uvw"]
    assert [layout.weight_slice.start for layout in grouped.weight_layout] == [
        0,
        grouped.weight_layout[0].weight_slice.stop,
        grouped.weight_layout[1].weight_slice.stop,
    ]
    assert grouped.weight_layout[-1].weight_slice.stop == grouped.weight_numel
    assert [layout.shape for layout in grouped.weight_layout] == [
        block.weight_shape for block in grouped.blocks
    ]

    left = torch.randn(2, 3, types[0].size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(
        2, 3, grouped.weight_numel, dtype=torch.float64, requires_grad=True
    )
    _compare_external_with_reference(grouped, left, right, weights)


def test_duplicate_block_instructions_remain_independent_blocks():
    space = _space("dihedral", 6)
    types = _mixed_types(space)
    instruction = nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu")
    product = nn.TensorProduct(
        *types,
        block_instructions=[instruction, instruction],
        internal_weights=False,
    )

    assert len(product.blocks) == 2
    assert len(product.paths) == 0
    assert product.weight_layout[0].weight_slice.stop == product.weight_layout[1].weight_slice.start
    assert product.weight_numel == 2 * product.blocks[0].weight_numel


@pytest.mark.parametrize(
    "block_instructions,connection_mode,error",
    [
        ([nn.TensorProductBlockInstruction(0, 0, 0, "bad")], "uvw", ValueError),
        ([nn.TensorProductBlockInstruction(-1, 0, 0)], "uvw", IndexError),
        ([nn.TensorProductBlockInstruction(0, 4, 0)], "uvw", IndexError),
        ([nn.TensorProductBlockInstruction(0, 0, 7)], "uvw", IndexError),
        ([nn.TensorProductBlockInstruction(0, 0, 0)], "uvu", ValueError),
    ],
)
def test_block_instruction_validation(block_instructions, connection_mode, error):
    space = _space("dihedral", 6)
    with pytest.raises(error):
        nn.TensorProduct(
            *_mixed_types(space),
            block_instructions=block_instructions,
            connection_mode=connection_mode,
        )


def test_legacy_and_block_instructions_are_mutually_exclusive():
    space = _space("dihedral", 6)
    with pytest.raises(ValueError, match="mutually exclusive"):
        nn.TensorProduct(
            *_mixed_types(space),
            instructions=[(0, 0, 0)],
            block_instructions=[nn.TensorProductBlockInstruction(0, 0, 0)],
        )


def test_uvu_validation_is_local_to_selected_block():
    space = _space("dihedral", 6)
    types = _mixed_types(space)
    product = nn.TensorProduct(
        *types,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu")
        ],
    )
    assert len(product.blocks) == 1

    with pytest.raises(ValueError, match="matching output and left multiplicities"):
        nn.TensorProduct(
            *types,
            block_instructions=[
                nn.TensorProductBlockInstruction(1, 1, 0, connection_mode="uvu")
            ],
        )


def test_zero_dimensional_block_coupling_is_rejected():
    space = _space("dihedral", 6)
    scalar, e1 = space.trivial_repr, _e(space, 1)
    with pytest.raises(ValueError, match="no equivariant coupling"):
        nn.TensorProduct(
            nn.FieldType(space, [scalar]),
            nn.FieldType(space, [scalar]),
            nn.FieldType(space, [e1]),
            block_instructions=[nn.TensorProductBlockInstruction(0, 0, 0)],
        )


def test_block_coupling_weight_and_unweighted_path_match_field_oracle():
    torch.manual_seed(127)
    space = _space("cyclic", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    types = (
        nn.FieldType(space, [e1, e1]),
        nn.FieldType(space, [e1]),
        nn.FieldType(space, [e2, e2]),
    )
    assert full_coupling_basis(e1, e1, e2).shape[0] >= 2
    grouped = nn.TensorProduct(
        *types,
        block_instructions=[
            nn.TensorProductBlockInstruction(
                0, 0, 0, connection_mode="uvu", coupling=1, path_weight=0.25
            ),
            nn.TensorProductBlockInstruction(
                0,
                0,
                0,
                connection_mode="uvu",
                coupling=0,
                has_weight=False,
                path_weight=-0.5,
            ),
        ],
        internal_weights=False,
        shared_weights=False,
    ).double()
    reference = _configured_explicit_reference(grouped)
    assert grouped.weight_numel == 2
    assert [path.has_weight for path in grouped.instructions] == [
        True,
        True,
        False,
        False,
    ]

    left = torch.randn(3, types[0].size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    left_ref = left.detach().clone().requires_grad_()
    right_ref = right.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    actual = grouped(left, right, weights)
    expected = reference(left_ref, right_ref, weights_ref)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for new, old in ((left, left_ref), (right, right_ref), (weights, weights_ref)):
        torch.testing.assert_close(new.grad, old.grad, atol=3e-11, rtol=3e-11)


def test_block_coupling_validation_and_unweighted_ambiguity():
    space = _space("cyclic", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    types = tuple(nn.FieldType(space, [rep]) for rep in (e1, e1, e2))
    with pytest.raises(ValueError, match="coupling index"):
        nn.TensorProduct(
            *types,
            block_instructions=[
                nn.TensorProductBlockInstruction(0, 0, 0, coupling=99)
            ],
        )
    with pytest.raises(ValueError, match="must select one coupling"):
        nn.TensorProduct(
            *types,
            block_instructions=[
                nn.TensorProductBlockInstruction(0, 0, 0, has_weight=False)
            ],
        )


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
    block = grouped.blocks[0]
    minimum_valid_budget = (
        block.coupling_shape[0] * block.output.size * torch.float64.itemsize
    )
    plan = tensor_product_module._cg_chunk_plan(
        batch_size=3,
        left_multiplicity=block.left_pack.multiplicity,
        right_multiplicity=block.right_pack.multiplicity,
        coupling_multiplicity=block.coupling_shape[0],
        output_size=block.output.size,
        element_size=torch.float64.itemsize,
        max_intermediate_bytes=minimum_valid_budget,
    )
    assert plan.chunked
    assert plan.estimated_chunk_bytes <= minimum_valid_budget
    monkeypatch.setattr(
        tensor_product_module,
        "_CG_MAX_INTERMEDIATE_BYTES",
        minimum_valid_budget,
    )
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


@pytest.mark.parametrize(
    "kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)]
)
@pytest.mark.parametrize("regular_position", ["output", "left", "right", "all"])
def test_uvu_analytic_regular_paths_match_explicit_reference_and_gradients(
    kind, n, regular_position
):
    torch.manual_seed(107)
    space = _space(kind, n)
    vector, regular = _e(space, 1), space.regular_repr
    left_rep = regular if regular_position in {"left", "all"} else vector
    right_rep = regular if regular_position in {"right", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    types = (
        nn.FieldType(space, [left_rep] * 2),
        nn.FieldType(space, [right_rep] * 2),
        nn.FieldType(space, [output_rep] * 2),
    )
    grouped = nn.TensorProduct(
        *types,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=False,
    ).double()
    assert len(grouped.blocks) == 1
    assert grouped.blocks[0].kind == (
        "output_regular"
        if regular_position in {"output", "all"}
        else f"{regular_position}_regular"
    )
    left = torch.randn(3, types[0].size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


@pytest.mark.parametrize("shared_weights", [True, False])
def test_uvu_all_regular_shared_and_per_sample_weights(shared_weights):
    space = _space("dihedral", 6)
    regular = space.regular_repr
    type_ = nn.FieldType(space, [regular] * 2)
    grouped = nn.TensorProduct(
        type_,
        type_,
        type_,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=shared_weights,
    ).double()
    left = torch.randn(2, type_.size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, type_.size, dtype=torch.float64, requires_grad=True)
    shape = (grouped.weight_numel,) if shared_weights else (2, grouped.weight_numel)
    weights = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    _compare_external_with_reference(grouped, left, right, weights)


def test_uvu_internal_regular_weights_and_gradients_match_reference():
    torch.manual_seed(109)
    space = _space("dihedral", 5)
    regular = space.regular_repr
    type_ = nn.FieldType(space, [regular] * 2)
    grouped = nn.TensorProduct(type_, type_, type_, connection_mode="uvu").double()
    reference = nn.TensorProduct(
        type_,
        type_,
        type_,
        instructions=[(p.i_in1, p.i_in2, p.i_out) for p in grouped.instructions],
    ).double()
    flattened = torch.cat([block.weight.detach().reshape(-1) for block in grouped.blocks])
    offset = 0
    with torch.no_grad():
        for path in reference.paths:
            count = path.weight_numel
            path.weight.copy_(flattened[offset : offset + count].reshape(path.weight_shape))
            offset += count
    left = torch.randn(2, type_.size, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, type_.size, dtype=torch.float64, requires_grad=True)
    left_ref = left.detach().clone().requires_grad_()
    right_ref = right.detach().clone().requires_grad_()
    actual, expected = grouped(left, right), reference(left_ref, right_ref)
    torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(left.grad, left_ref.grad, atol=8e-11, rtol=8e-11)
    torch.testing.assert_close(right.grad, right_ref.grad, atol=8e-11, rtol=8e-11)
    grouped_grad = torch.cat([block.weight.grad.reshape(-1) for block in grouped.blocks])
    reference_grad = torch.cat([path.weight.grad.reshape(-1) for path in reference.paths])
    torch.testing.assert_close(grouped_grad, reference_grad, atol=8e-11, rtol=8e-11)


@pytest.mark.parametrize("regular_position", ["output", "left", "right", "all"])
def test_uvu_analytic_regular_paths_are_equivariant(regular_position):
    space = _space("dihedral", 5)
    vector, regular = _e(space, 1), space.regular_repr
    left_rep = regular if regular_position in {"left", "all"} else vector
    right_rep = regular if regular_position in {"right", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    types = (
        nn.FieldType(space, [left_rep] * 2),
        nn.FieldType(space, [right_rep] * 2),
        nn.FieldType(space, [output_rep] * 2),
    )
    module = nn.TensorProduct(*types, connection_mode="uvu").double()
    left = torch.randn(2, types[0].size, dtype=torch.float64)
    right = torch.randn(2, types[1].size, dtype=torch.float64)
    output = module(left, right)
    for element in space.fibergroup.elements:
        actual = module(
            types[0].transform_fibers(left, element),
            types[1].transform_fibers(right, element),
        )
        expected = types[2].transform_fibers(output, element)
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)


@pytest.mark.parametrize("shared_weights", [True, False])
def test_uvu_kernel_forward_basis_and_gradients_match_field_path_oracle(shared_weights):
    torch.manual_seed(113)
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    feature_type = nn.FieldType(space, [e1] * 3)
    filter_type = nn.FieldType(space, [e1] * 2)
    output_type = nn.FieldType(space, [e2] * 3)
    kernel = nn.KernelTensorProduct(
        feature_type,
        filter_type,
        output_type,
        connection_mode="uvu",
        shared_weights=shared_weights,
    ).double()
    reference = _explicit_reference(kernel.tensor_product)
    features = torch.randn(4, feature_type.size, dtype=torch.float64, requires_grad=True)
    filters = torch.randn(4, filter_type.size, dtype=torch.float64, requires_grad=True)
    weight_shape = (kernel.weight_numel,) if shared_weights else (4, kernel.weight_numel)
    weights = torch.randn(*weight_shape, dtype=torch.float64, requires_grad=True)
    features_ref = features.detach().clone().requires_grad_()
    filters_ref = filters.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()

    actual = kernel(features, filters, weights)
    expected = reference(features_ref, filters_ref, weights_ref)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    sampled = kernel.sample_kernel_basis(filters)
    assert sampled.shape == (4, kernel.weight_numel, output_type.size, feature_type.size)
    basis_output = torch.einsum("...poi,...i,...p->...o", sampled, features, weights)
    torch.testing.assert_close(basis_output, actual, atol=3e-12, rtol=3e-12)

    (actual.square().sum() + basis_output.square().sum()).backward()
    (2.0 * expected.square().sum()).backward()
    for new, old in (
        (features, features_ref),
        (filters, filters_ref),
        (weights, weights_ref),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=8e-11, rtol=8e-11)

    detached_output = actual.detach()
    for element in space.fibergroup.elements:
        transformed = kernel(
            feature_type.transform_fibers(features.detach(), element),
            filter_type.transform_fibers(filters.detach(), element),
            weights.detach(),
        )
        torch.testing.assert_close(
            transformed,
            output_type.transform_fibers(detached_output, element),
            atol=4e-10,
            rtol=4e-10,
        )


def test_mixed_kernel_basis_reconstructs_multiaxis_forward_and_matches_oracle(
    monkeypatch,
):
    torch.manual_seed(131)
    space = _space("dihedral", 6)
    types = _mixed_types(space)
    kernel = nn.KernelTensorProduct(
        *types,
        block_instructions=_mixed_block_instructions(),
        shared_weights=False,
    ).double()
    reference = _configured_explicit_reference(kernel.tensor_product)
    assert kernel.connection_mode is None
    assert len(kernel.tensor_product.blocks) == 3
    assert len(kernel.tensor_product.paths) == 0

    tensor_product_module = __import__(
        "we3nn.nn.tensor_product", fromlist=["_CG_MAX_INTERMEDIATE_BYTES"]
    )
    minimum_budget = max(
        block.coupling_shape[0] * block.output.size * torch.float64.itemsize
        for block in kernel.tensor_product.blocks
        if block.kind == "cg"
    )
    monkeypatch.setattr(
        tensor_product_module, "_CG_MAX_INTERMEDIATE_BYTES", minimum_budget
    )

    features = torch.randn(2, 3, types[0].size, dtype=torch.float64, requires_grad=True)
    filters = torch.randn(2, 3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(
        2, 3, kernel.weight_numel, dtype=torch.float64, requires_grad=True
    )
    features_ref = features.detach().clone().requires_grad_()
    filters_ref = filters.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()

    actual = kernel(features, filters, weights)
    expected = reference(features_ref, filters_ref, weights_ref)
    sampled = kernel.sample_kernel_basis(filters)
    reconstructed = torch.einsum(
        "...poi,...i,...p->...o", sampled, features, weights
    )
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(reconstructed, actual, atol=3e-12, rtol=3e-12)

    (actual.square().sum() + reconstructed.square().sum()).backward()
    (2.0 * expected.square().sum()).backward()
    for new, old in (
        (features, features_ref),
        (filters, filters_ref),
        (weights, weights_ref),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=8e-11, rtol=8e-11)

    for element in space.fibergroup.elements:
        transformed = kernel(
            types[0].transform_fibers(features.detach(), element),
            types[1].transform_fibers(filters.detach(), element),
            weights.detach(),
        )
        expected_transformed = types[2].transform_fibers(actual.detach(), element)
        torch.testing.assert_close(
            transformed, expected_transformed, atol=5e-10, rtol=5e-10
        )


@pytest.mark.parametrize("regular_position", ["output", "left", "right", "all"])
def test_uvu_regular_kernel_basis_reconstructs_analytic_forward(regular_position):
    space = _space("dihedral", 5)
    vector, regular = _e(space, 1), space.regular_repr
    feature_rep = regular if regular_position in {"left", "all"} else vector
    filter_rep = regular if regular_position in {"right", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    feature_type = nn.FieldType(space, [feature_rep] * 2)
    filter_type = nn.FieldType(space, [filter_rep] * 2)
    output_type = nn.FieldType(space, [output_rep] * 2)
    kernel = nn.KernelTensorProduct(
        feature_type, filter_type, output_type, connection_mode="uvu"
    ).double()
    features = torch.randn(3, feature_type.size, dtype=torch.float64)
    filters = torch.randn(3, filter_type.size, dtype=torch.float64)
    weights = torch.randn(3, kernel.weight_numel, dtype=torch.float64)
    basis = kernel.sample_kernel_basis(filters)
    torch.testing.assert_close(
        torch.einsum("...poi,...i,...p->...o", basis, features, weights),
        kernel(features, filters, weights),
        atol=8e-12,
        rtol=8e-12,
    )


def test_uvu_spherical_kernel_from_points_basis_gradients_and_equivariance():
    torch.manual_seed(127)
    group = gspaces.flipRot2dOnR2(6).fibergroup
    space = gspaces.no_base_space(group)
    harmonics = RestrictedSphericalHarmonics(
        group, degrees=[0, 1, 2], normalization="component"
    )
    e1 = _e(space, 1)
    feature_type = nn.FieldType(space, [e1] * 2)
    output_type = nn.FieldType(space, [e1] * 2)
    kernel = nn.SphericalKernelTensorProduct(
        feature_type,
        harmonics,
        output_type,
        connection_mode="uvu",
    ).double()
    reference = _explicit_reference(kernel.tensor_product)
    features = torch.randn(3, feature_type.size, dtype=torch.float64, requires_grad=True)
    points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, kernel.weight_numel, dtype=torch.float64, requires_grad=True)
    features_ref = features.detach().clone().requires_grad_()
    points_ref = points.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()

    actual = kernel.forward_from_points(features, points, weights)
    expected = reference(features_ref, harmonics(points_ref), weights_ref)
    torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-12)
    basis = kernel.sample_kernel_basis(points)
    from_basis = torch.einsum("...poi,...i,...p->...o", basis, features, weights)
    torch.testing.assert_close(from_basis, actual, atol=5e-12, rtol=5e-12)
    (actual.square().sum() + from_basis.square().sum()).backward()
    (2.0 * expected.square().sum()).backward()
    for new, old in (
        (features, features_ref),
        (points, points_ref),
        (weights, weights_ref),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=2e-10, rtol=2e-10)

    actual = actual.detach()
    for element in group.elements:
        matrix = harmonics.embedding.matrix(element, dtype=torch.float64)
        transformed = kernel.forward_from_points(
            feature_type.transform_fibers(features.detach(), element),
            points.detach() @ matrix.T,
            weights.detach(),
        )
        torch.testing.assert_close(
            transformed,
            output_type.transform_fibers(actual, element),
            atol=3e-6,
            rtol=3e-6,
        )


def test_mixed_spherical_kernel_preserves_points_api_and_basis_reconstruction():
    torch.manual_seed(137)
    group = gspaces.flipRot2dOnR2(6).fibergroup
    space = gspaces.no_base_space(group)
    harmonics = RestrictedSphericalHarmonics(
        group, degrees=[0, 1, 2], normalization="component"
    )
    e1 = _e(space, 1)
    feature_type = nn.FieldType(space, [e1] * 2)
    output_type = nn.FieldType(space, [e1] * 2)
    block_instructions = [
        nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(0, 2, 0, connection_mode="uvw"),
    ]
    kernel = nn.SphericalKernelTensorProduct(
        feature_type,
        harmonics,
        output_type,
        block_instructions=block_instructions,
    ).double()
    reference = _configured_explicit_reference(kernel.tensor_product)
    assert [block.connection_mode for block in kernel.tensor_product.blocks] == [
        "uvu",
        "uvw",
    ]

    features = torch.randn(2, feature_type.size, dtype=torch.float64, requires_grad=True)
    points = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, kernel.weight_numel, dtype=torch.float64, requires_grad=True)
    features_ref = features.detach().clone().requires_grad_()
    points_ref = points.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()

    actual = kernel.forward_from_points(features, points, weights)
    expected = reference(features_ref, harmonics(points_ref), weights_ref)
    basis = kernel.sample_kernel_basis(points)
    reconstructed = torch.einsum(
        "...poi,...i,...p->...o", basis, features, weights
    )
    torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-12)
    torch.testing.assert_close(reconstructed, actual, atol=5e-12, rtol=5e-12)
    (actual.square().sum() + reconstructed.square().sum()).backward()
    (2.0 * expected.square().sum()).backward()
    for new, old in (
        (features, features_ref),
        (points, points_ref),
        (weights, weights_ref),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=2e-10, rtol=2e-10)

    actual = actual.detach()
    for element in group.elements:
        matrix = harmonics.embedding.matrix(element, dtype=torch.float64)
        transformed = kernel.forward_from_points(
            feature_type.transform_fibers(features.detach(), element),
            points.detach() @ matrix.T,
            weights.detach(),
        )
        torch.testing.assert_close(
            transformed,
            output_type.transform_fibers(actual, element),
            atol=3e-6,
            rtol=3e-6,
        )


def test_uvu_preserves_representation_tensor_type_safety():
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    left_type = nn.FieldType(space, [e1] * 2)
    right_type = nn.FieldType(space, [e1])
    output_type = nn.FieldType(space, [e2] * 2)
    module = nn.TensorProduct(left_type, right_type, output_type, connection_mode="uvu")
    output = module(
        RepresentationTensor(torch.randn(3, left_type.size), left_type),
        RepresentationTensor(torch.randn(3, right_type.size), right_type),
    )
    assert isinstance(output, RepresentationTensor)
    assert output.field_type == output_type
    with pytest.raises(TypeError, match="representation mismatch"):
        module(
            RepresentationTensor(torch.randn(3, left_type.size), output_type),
            RepresentationTensor(torch.randn(3, right_type.size), right_type),
        )


def test_uvu_128_channel_kernel_scale_has_linear_weights_and_one_block():
    space = _space("dihedral", 6)
    e1, e2 = _e(space, 1), _e(space, 2)
    channels = 128
    left_type = nn.FieldType(space, [e1] * channels)
    filter_type = nn.FieldType(space, [e1])
    output_type = nn.FieldType(space, [e2] * channels)
    product = nn.TensorProduct(
        left_type,
        filter_type,
        output_type,
        connection_mode="uvu",
    )
    coupling_count = full_coupling_basis(e1, e1, e2).shape[0]
    assert product.weight_numel == channels * coupling_count
    assert len(product.blocks) == 1
    assert len(product.paths) == 0
    assert len(product.instructions) == channels
    assert sum(1 for _ in product.modules()) == 4
    assert sum(
        name.endswith("coupling_basis") and buffer.numel() > 0
        for name, buffer in product.named_buffers()
    ) == 1
    assert sum(parameter.numel() for parameter in product.parameters()) == product.weight_numel

    left = torch.randn(4, left_type.size, requires_grad=True)
    filters = torch.randn(4, filter_type.size, requires_grad=True)
    output = product(left, filters)
    assert output.shape == (4, output_type.size)
    output.square().mean().backward()
    assert left.grad is not None
    assert filters.grad is not None
    assert product.blocks[0].weight.grad is not None

    fully_mixed = nn.TensorProduct(
        left_type,
        filter_type,
        output_type,
        internal_weights=False,
    )
    assert fully_mixed.weight_numel == channels**2 * coupling_count
    assert fully_mixed.weight_numel == channels * product.weight_numel
