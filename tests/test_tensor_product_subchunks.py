import pytest
import torch

from we3nn import RestrictedSphericalHarmonics, RepresentationTensor, gspaces, nn


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


def _mixed_subchunk_case(space):
    e1, scalar = _e(space, 1), space.trivial_repr
    types = (
        nn.FieldType(space, [e1] * 8 + [scalar] * 3),
        nn.FieldType(space, [scalar, e1]),
        nn.FieldType(space, [e1] * 8),
    )
    left_specs = [_spec(*range(4)), _spec(*range(4, 8)), _spec(8, 9, 10)]
    output_specs = [_spec(*range(4)), _spec(*range(4, 8))]
    instructions = [
        nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(1, 0, 1, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(2, 1, 0, connection_mode="uvw"),
        nn.TensorProductBlockInstruction(2, 1, 1, connection_mode="uvw"),
    ]
    return types, left_specs, output_specs, instructions


def test_split_chunks_support_mixed_uvw_uvu_and_contiguous_weight_layout():
    torch.manual_seed(167)
    space = _space()
    types, left_specs, output_specs, instructions = _mixed_subchunk_case(space)
    product = nn.TensorProduct(
        *types,
        in1_chunks=left_specs,
        out_chunks=output_specs,
        block_instructions=instructions,
        internal_weights=False,
        shared_weights=False,
    ).double()
    assert [block.connection_mode for block in product.blocks] == [
        "uvu",
        "uvu",
        "uvw",
        "uvw",
    ]
    assert len(product.blocks) == 4 and len(product.paths) == 0
    assert all(
        first.weight_slice.stop == second.weight_slice.start
        for first, second in zip(product.weight_layout, product.weight_layout[1:])
    )
    x = torch.randn(2, 3, types[0].size, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, product.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external(product, x, y, weights)


def test_custom_implicit_uvw_uses_block_contiguous_layout_and_oracle_order():
    torch.manual_seed(169)
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
        internal_weights=False,
        shared_weights=False,
    ).double()
    assert product.weight_layout is not None
    assert product._uses_legacy_weight_layout is False
    x = torch.randn(2, left.size, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, right.size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, product.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external(product, x, y, weights)


def test_custom_regular_subchunks_keep_analytic_execution():
    torch.manual_seed(173)
    space = _space(5)
    regular, scalar = space.regular_repr, space.trivial_repr
    features = nn.FieldType(space, [regular] * 4)
    filters = nn.FieldType(space, [scalar])
    output = nn.FieldType(space, [regular] * 4)
    halves = [_spec(0, 1), _spec(2, 3)]
    product = nn.TensorProduct(
        features,
        filters,
        output,
        in1_chunks=halves,
        out_chunks=halves,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 1, connection_mode="uvu"),
            nn.TensorProductBlockInstruction(1, 0, 0, connection_mode="uvw"),
        ],
        internal_weights=False,
        shared_weights=False,
    ).double()
    assert [block.kind for block in product.blocks] == [
        "output_regular",
        "output_regular",
    ]
    assert all(block.coupling_basis.numel() == 0 for block in product.blocks)
    x = torch.randn(2, features.size, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, filters.size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, product.weight_numel, dtype=torch.float64, requires_grad=True)
    _compare_external(product, x, y, weights)


def test_kernel_custom_subchunks_forward_basis_and_gradients_match_oracle():
    torch.manual_seed(179)
    space = _space()
    types, left_specs, output_specs, _ = _mixed_subchunk_case(space)
    filter_specs = [_spec(1), _spec(0)]
    remapped_instructions = [
        nn.TensorProductBlockInstruction(0, 1, 0, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(1, 1, 1, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(2, 0, 0, connection_mode="uvw"),
        nn.TensorProductBlockInstruction(2, 0, 1, connection_mode="uvw"),
    ]
    kernel = nn.KernelTensorProduct(
        *types,
        in1_chunks=left_specs,
        in2_chunks=filter_specs,
        out_chunks=output_specs,
        block_instructions=remapped_instructions,
    ).double()
    reference = _explicit_reference(kernel.tensor_product)
    features = torch.randn(2, 3, types[0].size, dtype=torch.float64, requires_grad=True)
    filters = torch.randn(2, 3, types[1].size, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, kernel.weight_numel, dtype=torch.float64, requires_grad=True)
    features_ref = features.detach().clone().requires_grad_()
    filters_ref = filters.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    direct = kernel(features, filters, weights)
    expected = reference(features_ref, filters_ref, weights_ref)
    basis = kernel.sample_kernel_basis(filters)
    reconstructed = torch.einsum(
        "...poi,...i,...p->...o", basis, features, weights
    )
    torch.testing.assert_close(direct, expected, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(reconstructed, direct, atol=3e-12, rtol=3e-12)
    (direct.square().sum() + reconstructed.square().sum()).backward()
    (2 * expected.square().sum()).backward()
    for new, old in (
        (features, features_ref),
        (filters, filters_ref),
        (weights, weights_ref),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=8e-11, rtol=8e-11)
    assert kernel.in1_chunks == kernel.tensor_product.in1_chunks
    assert [chunk.field_indices for chunk in kernel.in2_chunks] == [(1,), (0,)]
    assert kernel.out_chunks == kernel.tensor_product.out_chunks


def test_spherical_kernel_forwards_custom_subchunks_and_remains_equivariant():
    torch.manual_seed(181)
    group = gspaces.flipRot2dOnR2(6).fibergroup
    space = gspaces.no_base_space(group)
    harmonics = RestrictedSphericalHarmonics(
        group, degrees=[0, 1, 2], normalization="component"
    )
    e1 = _e(space, 1)
    features = nn.FieldType(space, [e1] * 4)
    output = nn.FieldType(space, [e1] * 4)
    halves = [_spec(2, 0), _spec(3, 1)]
    kernel = nn.SphericalKernelTensorProduct(
        features,
        harmonics,
        output,
        in1_chunks=halves,
        out_chunks=halves,
        block_instructions=[
            nn.TensorProductBlockInstruction(0, 0, 1, connection_mode="uvu"),
            nn.TensorProductBlockInstruction(1, 2, 0, connection_mode="uvu"),
        ],
    ).double()
    x = torch.randn(3, features.size, dtype=torch.float64, requires_grad=True)
    points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, kernel.weight_numel, dtype=torch.float64, requires_grad=True)
    direct = kernel.forward_from_points(x, points, weights)
    basis = kernel.sample_kernel_basis(points)
    reconstructed = torch.einsum("...poi,...i,...p->...o", basis, x, weights)
    torch.testing.assert_close(reconstructed, direct, atol=5e-12, rtol=5e-12)
    direct.square().sum().backward()
    assert x.grad is not None and points.grad is not None and weights.grad is not None
    detached = direct.detach()
    for element in group.elements:
        matrix = harmonics.embedding.matrix(element, dtype=torch.float64)
        transformed = kernel.forward_from_points(
            features.transform_fibers(x.detach(), element),
            points.detach() @ matrix.T,
            weights.detach(),
        )
        torch.testing.assert_close(
            transformed,
            output.transform_fibers(detached, element),
            atol=3e-6,
            rtol=3e-6,
        )


def test_128_channels_split_into_four_blocks_without_path_expansion():
    space = _space()
    e1, scalar = _e(space, 1), space.trivial_repr
    features = nn.FieldType(space, [e1] * 128)
    filters = nn.FieldType(space, [scalar])
    output = nn.FieldType(space, [e1] * 128)
    quarters = [_spec(*range(start, start + 32)) for start in range(0, 128, 32)]
    product = nn.TensorProduct(
        features,
        filters,
        output,
        in1_chunks=quarters,
        out_chunks=quarters,
        block_instructions=[
            nn.TensorProductBlockInstruction(i, 0, (i + 1) % 4, connection_mode="uvu")
            for i in range(4)
        ],
        internal_weights=False,
    )
    assert len(product.in1_chunks) == 4
    assert len(product.blocks) == 4
    assert len(product.paths) == 0
    assert len(product.instructions) == 128
    assert sum(1 for _ in product.modules()) < 12
