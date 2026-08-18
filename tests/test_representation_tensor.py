import warnings

import pytest
import torch

from we3nn import (
    MissingRepresentationMetadataWarning,
    RepresentationTensor,
    RestrictedSphericalHarmonics,
    gspaces,
    nn,
)


def _d6_types():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar = nn.FieldType(space, [space.trivial_repr])
    vector = nn.FieldType(space, [space.irrep(1, 1)])
    regular = nn.FieldType(space, [space.regular_repr])
    return scalar, vector, regular


def test_representation_tensor_validates_last_axis():
    _, vector, _ = _d6_types()
    wrapped = RepresentationTensor(torch.randn(4, 2), vector)
    assert wrapped.shape == (4, 2)
    assert wrapped.field_type == vector
    with pytest.raises(ValueError, match="last dimension"):
        RepresentationTensor(torch.randn(4, 3), vector)


def test_raw_tensor_product_warns_and_preserves_raw_tensor_api():
    scalar, vector, _ = _d6_types()
    product = nn.TensorProduct(vector, vector, scalar)
    with pytest.warns(MissingRepresentationMetadataWarning, match="input1, input2"):
        output = product(torch.randn(5, 2), torch.randn(5, 2))
    assert isinstance(output, torch.Tensor)


def test_typed_tensor_product_checks_and_propagates_metadata():
    scalar, vector, _ = _d6_types()
    product = nn.TensorProduct(vector, vector, scalar)
    left = vector.wrap(torch.randn(5, 2))
    right = vector.wrap(torch.randn(5, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingRepresentationMetadataWarning)
        output = product(left, right)
    assert isinstance(output, RepresentationTensor)
    assert output.field_type == scalar
    assert output.tensor.shape == (5, 1)


def test_partial_metadata_warns_and_returns_raw_output():
    scalar, vector, _ = _d6_types()
    product = nn.TensorProduct(vector, vector, scalar)
    left = RepresentationTensor(torch.randn(5, 2), vector)
    with pytest.warns(MissingRepresentationMetadataWarning, match="input2"):
        output = product(left, torch.randn(5, 2))
    assert isinstance(output, torch.Tensor)


def test_representation_mismatch_errors_even_when_dimensions_match():
    scalar, vector, _ = _d6_types()
    wrong_two_dimensional_type = nn.FieldType(
        scalar.gspace,
        [scalar.gspace.trivial_repr, scalar.gspace.trivial_repr],
    )
    product = nn.TensorProduct(vector, vector, scalar)
    wrong = RepresentationTensor(torch.randn(5, 2), wrong_two_dimensional_type)
    right = RepresentationTensor(torch.randn(5, 2), vector)
    with pytest.raises(TypeError, match="representation mismatch"):
        product(wrong, right)


def test_linear_activation_and_sequential_preserve_metadata():
    _, vector, regular = _d6_types()
    model = nn.SequentialModule(
        nn.WELinear(vector, regular),
        nn.PointActiv(regular, torch.relu),
        nn.WELinear(regular, vector),
    )
    output = model(RepresentationTensor(torch.randn(3, 2), vector))
    assert isinstance(output, RepresentationTensor)
    assert output.field_type == vector


def test_full_and_wigner_eckart_tensor_products_support_metadata():
    scalar, vector, _ = _d6_types()
    left = RepresentationTensor(torch.randn(4, 2), vector)
    right = RepresentationTensor(torch.randn(4, 2), vector)

    full_output = nn.FullTensorProduct(vector, vector)(left, right)
    assert isinstance(full_output, RepresentationTensor)

    product = nn.WETensorProduct(vector, vector, scalar)
    weights = torch.randn(4, product.weight_numel)
    output = product(left, right, weights)
    assert isinstance(output, RepresentationTensor)
    assert output.field_type == scalar


def test_restricted_wigner_eckart_propagates_known_harmonic_metadata():
    _, vector, _ = _d6_types()
    harmonics = RestrictedSphericalHarmonics(
        vector.fibergroup,
        degrees=[0, 1],
    )
    product = nn.RestrictedWETensorProduct(vector, harmonics, vector)
    features = vector.wrap(torch.randn(4, 2))
    points = torch.randn(4, 3)
    weights = torch.randn(4, product.weight_numel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingRepresentationMetadataWarning)
        output = product.forward_from_points(features, points, weights)
    assert isinstance(output, RepresentationTensor)
    assert output.field_type == vector


def test_wrapper_is_exported_from_group_namespace():
    from we3nn import group

    assert group.RepresentationTensor is RepresentationTensor
    assert group.MissingRepresentationMetadataWarning is MissingRepresentationMetadataWarning
