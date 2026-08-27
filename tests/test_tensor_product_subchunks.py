import pytest
import torch

from we3nn import RepresentationTensor, gspaces, nn


def _space(n=6):
    return gspaces.no_base_space(gspaces.flipRot2dOnR2(n).fibergroup)


def _e(space, frequency):
    return space.irrep(1, frequency)


def _spec(*indices):
    return nn.MultiplicityChunkSpec(indices)


def _explicit_reference(grouped):
    return nn.TensorProduct(
        grouped.in1_type,
        grouped.in2_type,
        grouped.out_type,
        instructions=[
            (path.i_in1, path.i_in2, path.i_out) for path in grouped.instructions
        ],
        internal_weights=False,
        shared_weights=grouped.shared_weights,
    ).double()


def _compare_external(grouped, left, right, weights):
    reference = _explicit_reference(grouped)
    left_ref = left.detach().clone().requires_grad_()
    right_ref = right.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    actual = grouped(left, right, weights)
    expected = reference(left_ref, right_ref, weights_ref)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for new, old in ((left, left_ref), (right, right_ref), (weights, weights_ref)):
        torch.testing.assert_close(new.grad, old.grad, atol=8e-11, rtol=8e-11)


def _validation_product(specs):
    space = _space()
    e1, scalar = _e(space, 1), space.trivial_repr
    left = nn.FieldType(space, [e1, e1, scalar])
    scalar_type = nn.FieldType(space, [scalar])
    return nn.TensorProduct(
        left,
        scalar_type,
        scalar_type,
        in1_chunks=specs,
        block_instructions=[],
    )


@pytest.mark.parametrize(
    "specs,error,match",
    [
        ([_spec()], ValueError, "empty"),
        ([_spec(-1, 0, 1, 2)], IndexError, "outside"),
        ([_spec(0, 1, 2, 3)], IndexError, "outside"),
        ([_spec(0, 0, 1), _spec(2)], ValueError, "repeats field"),
        ([_spec(0, 1), _spec(1), _spec(2)], ValueError, "belongs to both"),
        ([_spec(0, 2), _spec(1)], ValueError, "mixes representations"),
        ([_spec(0, 1)], ValueError, "omits field"),
        ([nn.MultiplicityChunkSpec((0, 1.5)), _spec(2)], TypeError, "noninteger"),
    ],
)
def test_explicit_chunk_partition_validation(specs, error, match):
    with pytest.raises(error, match=match):
        _validation_product(specs)


def test_duplicate_representation_chunks_drive_implicit_grouped_generation():
    space = _space()
    e1, scalar = _e(space, 1), space.trivial_repr
    left = nn.FieldType(space, [e1] * 4)
    right = nn.FieldType(space, [scalar])
    output = nn.FieldType(space, [e1] * 4)
    halves = [_spec(0, 1), _spec(2, 3)]
    product = nn.TensorProduct(
        left,
        right,
        output,
        in1_chunks=halves,
        out_chunks=halves,
        connection_mode="uvu",
        internal_weights=False,
    )

    assert [chunk.field_indices for chunk in product.in1_chunks] == [(0, 1), (2, 3)]
    assert product.in1_chunks[0].representation is product.in1_chunks[1].representation
    assert len(product.blocks) == 4
    assert len(product.paths) == 0
    assert len(product.instructions) == 8
    assert product.weight_layout[-1].weight_slice.stop == product.weight_numel


def test_crossed_permuted_subchunks_match_field_oracle_and_preserve_typing():
    torch.manual_seed(163)
    space = _space()
    e1, scalar = _e(space, 1), space.trivial_repr
    left = nn.FieldType(space, [e1] * 8)
    right = nn.FieldType(space, [scalar])
    output = nn.FieldType(space, [e1] * 8)
    left_specs = [_spec(3, 0, 2, 1), _spec(7, 4, 6, 5)]
    output_specs = [_spec(4, 7, 5, 6), _spec(0, 3, 1, 2)]
    product = nn.TensorProduct(
        left,
        right,
        output,
        in1_chunks=left_specs,
        out_chunks=output_specs,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 1, connection_mode="uvu"),
            nn.TensorProductBlockInstruction(1, 0, 0, connection_mode="uvu"),
        ],
        internal_weights=False,
        shared_weights=False,
    ).double()
    assert product.in1_chunks[0].field_indices == (3, 0, 2, 1)
    assert product.out_chunks[0].field_indices == (4, 7, 5, 6)
    assert all(block.left_pack.contiguous_slice is None for block in product.blocks)
    assert all(block.output_pack.contiguous_slice is None for block in product.blocks)

    x = torch.randn(2, 3, left.size, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 3, right.size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, product.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external(product, x, y, weights)
    typed = product(
        RepresentationTensor(x.detach(), left),
        RepresentationTensor(y.detach(), right),
        weights.detach(),
    )
    assert isinstance(typed, RepresentationTensor)
    assert typed.field_type == output


def test_uvu_checks_only_selected_subchunk_multiplicities():
    space = _space()
    e1, e2 = _e(space, 1), _e(space, 2)
    left = nn.FieldType(space, [e1] * 10)
    right = nn.FieldType(space, [e1])
    output = nn.FieldType(space, [e2] * 10)
    split = [_spec(*range(4)), _spec(*range(4, 10))]
    valid = nn.TensorProduct(
        left,
        right,
        output,
        in1_chunks=split,
        out_chunks=split,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu")
        ],
    )
    assert len(valid.blocks) == 1
    with pytest.raises(ValueError, match="matching output and left multiplicities"):
        nn.TensorProduct(
            left,
            right,
            output,
            in1_chunks=split,
            out_chunks=split,
            block_instructions=[
                nn.TensorProductBlockInstruction(0, 0, 1, connection_mode="uvu")
            ],
        )


def test_same_representation_subchunks_have_independent_internal_parameters():
    space = _space()
    e1, scalar = _e(space, 1), space.trivial_repr
    type_ = nn.FieldType(space, [e1] * 4)
    scalar_type = nn.FieldType(space, [scalar])
    halves = [_spec(0, 1), _spec(2, 3)]
    product = nn.TensorProduct(
        type_,
        scalar_type,
        type_,
        in1_chunks=halves,
        out_chunks=halves,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
            nn.TensorProductBlockInstruction(1, 0, 1, connection_mode="uvu"),
        ],
    )
    assert len(product.blocks) == 2
    assert product.blocks[0].weight is not product.blocks[1].weight
    assert product.blocks[0].weight.data_ptr() != product.blocks[1].weight.data_ptr()
