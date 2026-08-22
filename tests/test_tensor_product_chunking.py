import importlib

import pytest
import torch

from we3nn import gspaces, nn


tensor_product_module = importlib.import_module("we3nn.nn.tensor_product")


def _types(kind, n, coupling, multiplicity=3):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    space = gspaces.no_base_space(base.fibergroup)
    scalar = space.trivial_repr
    vector = space.irrep(1) if kind == "cyclic" else space.irrep(1, 1)
    other = space.irrep(2) if kind == "cyclic" else space.irrep(1, 2)
    if coupling == "scalar_scalar":
        representations = (scalar, scalar, scalar)
    elif coupling == "scalar_vector":
        representations = (scalar, vector, vector)
    elif coupling == "vector_scalar":
        representations = (vector, vector, scalar)
    else:
        representations = (vector, vector, other)
    return tuple(
        nn.FieldType(space, [representation] * multiplicity)
        for representation in representations
    )


def _clone_inputs(left_type, right_type, *, dtype):
    left = torch.randn(5, left_type.size, dtype=dtype, requires_grad=True)
    right = torch.randn(5, right_type.size, dtype=dtype, requires_grad=True)
    return left, right


def _parameter_gradients(module):
    return torch.cat([parameter.grad.reshape(-1) for parameter in module.parameters()])


@pytest.mark.parametrize(
    "kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)]
)
@pytest.mark.parametrize(
    "coupling", ["scalar_scalar", "scalar_vector", "vector_scalar", "vector_other"]
)
@pytest.mark.parametrize("weight_mode", ["internal", "shared", "unshared"])
def test_forced_chunked_cg_matches_unchunked_outputs_and_gradients(
    monkeypatch, kind, n, coupling, weight_mode
):
    torch.manual_seed(47)
    left_type, right_type, output_type = _types(kind, n, coupling)
    internal = weight_mode == "internal"
    shared = weight_mode != "unshared"
    reference = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        internal_weights=internal,
        shared_weights=shared,
    ).double()
    chunked = nn.TensorProduct(
        left_type,
        right_type,
        output_type,
        internal_weights=internal,
        shared_weights=shared,
    ).double()
    if internal:
        chunked.load_state_dict(reference.state_dict())

    left_reference, right_reference = _clone_inputs(
        left_type, right_type, dtype=torch.float64
    )
    left_chunked = left_reference.detach().clone().requires_grad_()
    right_chunked = right_reference.detach().clone().requires_grad_()
    weight_shape = (reference.weight_numel,) if shared else (5, reference.weight_numel)
    weight_reference = (
        None
        if internal
        else torch.randn(*weight_shape, dtype=torch.float64, requires_grad=True)
    )
    weight_chunked = (
        None
        if internal
        else weight_reference.detach().clone().requires_grad_()
    )

    expected = reference(left_reference, right_reference, weight_reference)
    expected.square().sum().backward()
    monkeypatch.setattr(tensor_product_module, "_CG_MAX_INTERMEDIATE_BYTES", 64)
    plan = tensor_product_module._cg_chunk_plan(
        batch_size=5,
        left_multiplicity=3,
        right_multiplicity=3,
        coupling_multiplicity=chunked.blocks[0].coupling_shape[0],
        output_size=chunked.blocks[0].output.size,
        element_size=8,
    )
    assert plan.chunked
    assert plan.estimated_chunk_bytes <= plan.max_intermediate_bytes
    actual = chunked(left_chunked, right_chunked, weight_chunked)
    actual.square().sum().backward()

    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(left_chunked.grad, left_reference.grad, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(right_chunked.grad, right_reference.grad, atol=3e-11, rtol=3e-11)
    if internal:
        torch.testing.assert_close(
            _parameter_gradients(chunked),
            _parameter_gradients(reference),
            atol=3e-11,
            rtol=3e-11,
        )
    else:
        torch.testing.assert_close(
            weight_chunked.grad, weight_reference.grad, atol=3e-11, rtol=3e-11
        )


def test_forced_chunked_float32_and_equivariance(monkeypatch):
    left_type, right_type, output_type = _types("dihedral", 6, "vector_other", 4)
    reference = nn.TensorProduct(left_type, right_type, output_type)
    chunked = nn.TensorProduct(left_type, right_type, output_type)
    chunked.load_state_dict(reference.state_dict())
    left, right = _clone_inputs(left_type, right_type, dtype=torch.float32)
    expected = reference(left, right)
    monkeypatch.setattr(tensor_product_module, "_CG_MAX_INTERMEDIATE_BYTES", 64)
    actual = chunked(left, right)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    chunked.check_equivariance(atol=4e-5, rtol=4e-5)


@pytest.mark.parametrize("shared_weights", [True, False])
def test_kernel_tensor_product_uses_chunked_backend(monkeypatch, shared_weights):
    left_type, filter_type, output_type = _types("dihedral", 6, "vector_other", 3)
    reference = nn.KernelTensorProduct(
        left_type, filter_type, output_type, shared_weights=shared_weights
    ).double()
    chunked = nn.KernelTensorProduct(
        left_type, filter_type, output_type, shared_weights=shared_weights
    ).double()
    features, filters = _clone_inputs(left_type, filter_type, dtype=torch.float64)
    features_chunked = features.detach().clone().requires_grad_()
    filters_chunked = filters.detach().clone().requires_grad_()
    shape = (reference.weight_numel,) if shared_weights else (5, reference.weight_numel)
    weights = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    weights_chunked = weights.detach().clone().requires_grad_()
    expected = reference(features, filters, weights)
    expected.square().sum().backward()
    monkeypatch.setattr(tensor_product_module, "_CG_MAX_INTERMEDIATE_BYTES", 64)
    actual = chunked(features_chunked, filters_chunked, weights_chunked)
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    for actual_value, expected_value in (
        (features_chunked, features),
        (filters_chunked, filters),
        (weights_chunked, weights),
    ):
        torch.testing.assert_close(
            actual_value.grad, expected_value.grad, atol=3e-11, rtol=3e-11
        )


def test_chunk_planner_bounds_large_edge_and_single_sample_temporaries():
    edge_plan = tensor_product_module._cg_chunk_plan(
        batch_size=100_000,
        left_multiplicity=128,
        right_multiplicity=128,
        coupling_multiplicity=1,
        output_size=2,
        element_size=4,
    )
    assert edge_plan.estimated_unchunked_bytes > 12 * 2**30
    assert edge_plan.estimated_chunk_bytes <= 256 * 2**20
    assert edge_plan.batch_chunk < edge_plan.batch_size

    sample_plan = tensor_product_module._cg_chunk_plan(
        batch_size=1,
        left_multiplicity=128,
        right_multiplicity=128,
        coupling_multiplicity=1,
        output_size=2,
        element_size=4,
        max_intermediate_bytes=1024,
    )
    assert sample_plan.estimated_chunk_bytes <= 1024
    assert sample_plan.left_chunk < 128 or sample_plan.right_chunk < 128
