import pytest
import torch
from torch.nn import functional as F

from we3nn import MatrixFiniteGroup, RepresentationTensor, gspaces, nn


def _space(kind="dihedral", n=6):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    return gspaces.no_base_space(base.fibergroup)


def _irrep(space, frequency=1):
    if space.fibergroup.name.startswith("C"):
        return space.irrep(frequency)
    return space.irrep(1, frequency)


def _matched_layers(in_type, out_type, *, bias=True, backend="structured"):
    dense = nn.WELinear(
        in_type, out_type, bias=bias, backend=backend, execution="dense"
    ).double()
    direct = nn.WELinear(
        in_type, out_type, bias=bias, backend=backend, execution="direct"
    ).double()
    direct.load_state_dict(dense.state_dict(), strict=True)
    return dense, direct


@pytest.mark.parametrize("kind,n", [("cyclic", 5), ("dihedral", 6)])
@pytest.mark.parametrize(
    "source,target",
    [
        ("trivial", "trivial"),
        ("trivial", "regular"),
        ("regular", "trivial"),
        ("regular", "regular"),
        ("irrep", "regular"),
        ("regular", "irrep"),
        ("irrep", "irrep"),
    ],
)
def test_direct_pair_forward_matches_expanded_dense(kind, n, source, target):
    torch.manual_seed(211)
    space = _space(kind, n)
    reps = {
        "trivial": space.trivial_repr,
        "regular": space.regular_repr,
        "irrep": _irrep(space),
    }
    in_type = nn.FieldType(space, [reps[source]] * 2)
    out_type = nn.FieldType(space, [reps[target]] * 3)
    dense, direct = _matched_layers(in_type, out_type)
    x = torch.randn(2, 3, in_type.size, dtype=torch.float64)
    weight, bias = direct.expand_parameters()
    expected = F.linear(x, weight, bias)
    torch.testing.assert_close(direct(x), expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(dense(x), expected, atol=0, rtol=0)


def test_direct_mixed_noncontiguous_fields_forward_and_gradients_match_dense():
    torch.manual_seed(223)
    space = _space()
    scalar, vector, regular = space.trivial_repr, _irrep(space), space.regular_repr
    in_type = nn.FieldType(space, [vector, scalar, regular, vector, scalar])
    out_type = nn.FieldType(space, [regular, vector, scalar, regular, vector])
    dense, direct = _matched_layers(in_type, out_type)
    x_dense = torch.randn(2, 3, in_type.size, dtype=torch.float64, requires_grad=True)
    x_direct = x_dense.detach().clone().requires_grad_()
    expected = dense(x_dense)
    actual = direct(x_direct)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x_direct.grad, x_dense.grad, atol=2e-11, rtol=2e-11)
    for dense_parameter, direct_parameter in zip(dense.parameters(), direct.parameters()):
        torch.testing.assert_close(
            direct_parameter.grad, dense_parameter.grad, atol=3e-11, rtol=3e-11
        )


def test_direct_higher_derivatives_match_dense_force_workload():
    torch.manual_seed(227)
    space = _space()
    in_type = nn.FieldType(space, [space.trivial_repr] * 3 + [_irrep(space)] * 2)
    out_type = nn.FieldType(space, [space.regular_repr] * 2)
    dense, direct = _matched_layers(in_type, out_type)
    x_dense = torch.randn(4, in_type.size, dtype=torch.float64, requires_grad=True)
    x_direct = x_dense.detach().clone().requires_grad_()

    def derivatives(layer, value):
        energy = layer(value).square().sum()
        force = torch.autograd.grad(energy, value, create_graph=True)[0]
        second = torch.autograd.grad(
            force.square().sum(), (value, *layer.parameters()), allow_unused=True
        )
        return force, second

    force_dense, second_dense = derivatives(dense, x_dense)
    force_direct, second_direct = derivatives(direct, x_direct)
    torch.testing.assert_close(force_direct, force_dense, atol=3e-11, rtol=3e-11)
    for actual, expected in zip(second_direct, second_dense):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


def test_direct_generic_backend_and_arbitrary_finite_group():
    values = ("e", "a", "b", "c")
    group = MatrixFiniteGroup(
        values,
        [[i ^ j for j in range(4)] for i in range(4)],
        [0, 1, 2, 3],
        0,
        name="V4-direct",
        generators=[1, 2],
    )
    matrices = {
        "e": torch.eye(2),
        "a": torch.diag(torch.tensor([-1.0, 1.0])),
        "b": torch.diag(torch.tensor([1.0, -1.0])),
        "c": -torch.eye(2),
    }
    supplied = group.representation(matrices, name="two signs")
    space = gspaces.no_base_space(group)
    in_type = nn.FieldType(space, [supplied, group.regular_repr, supplied])
    out_type = nn.FieldType(space, [group.regular_repr, supplied])
    dense, direct = _matched_layers(in_type, out_type, backend="generic")
    x = torch.randn(3, in_type.size, dtype=torch.float64)
    torch.testing.assert_close(direct(x), dense(x), atol=3e-12, rtol=3e-12)


def test_execution_state_dict_expand_parameters_typing_and_equivariance():
    torch.manual_seed(229)
    space = _space()
    in_type = nn.FieldType(space, [space.trivial_repr, _irrep(space)] * 2)
    out_type = nn.FieldType(space, [space.regular_repr, _irrep(space)])
    dense, direct = _matched_layers(in_type, out_type)
    dense.load_state_dict(direct.state_dict(), strict=True)
    direct.load_state_dict(dense.state_dict(), strict=True)
    dense_weight, dense_bias = dense.expand_parameters()
    direct_weight, direct_bias = direct.expand_parameters()
    torch.testing.assert_close(direct_weight, dense_weight, atol=0, rtol=0)
    torch.testing.assert_close(direct_bias, dense_bias, atol=0, rtol=0)
    x = torch.randn(5, in_type.size, dtype=torch.float64)
    typed = direct(RepresentationTensor(x, in_type))
    assert isinstance(typed, RepresentationTensor) and typed.field_type == out_type
    for element in space.fibergroup.elements:
        torch.testing.assert_close(
            direct(in_type.transform_fibers(x, element)),
            out_type.transform_fibers(direct(x), element),
            atol=3e-12,
            rtol=3e-12,
        )


def test_direct_does_not_call_global_dense_expansion(monkeypatch):
    space = _space()
    type_ = nn.FieldType(space, [space.trivial_repr, space.regular_repr] * 2)
    layer = nn.WELinear(type_, type_, execution="direct")

    def fail():
        raise AssertionError("direct execution materialized the global dense weight")

    monkeypatch.setattr(layer, "expand_parameters", fail)
    assert layer(torch.randn(3, type_.size)).shape == (3, type_.size)


def test_auto_hybrid_mixes_pairs_without_global_expansion(monkeypatch):
    space = _space()
    in_type = nn.FieldType(
        space, [space.trivial_repr] * 16 + [space.regular_repr] * 2
    )
    out_type = nn.FieldType(space, [space.regular_repr] * 3)
    layer = nn.WELinear(in_type, out_type, execution="auto_hybrid")
    kinds = {pair.direct_kind: pair.auto_uses_direct() for pair in layer._pairs}
    assert kinds == {"trivial_regular": True, "regular_regular": False}

    def fail():
        raise AssertionError("auto execution materialized the global dense weight")

    monkeypatch.setattr(layer, "expand_parameters", fail)
    x = torch.randn(4, in_type.size)
    assert layer(x).shape == (4, out_type.size)
    with torch.no_grad():
        assert layer(x).shape == (4, out_type.size)


def test_auto_hybrid_mixed_edge_encoder_values_and_gradients_match_dense():
    torch.manual_seed(233)
    space = _space()
    in_type = nn.FieldType(
        space, [space.trivial_repr] * 16 + [space.regular_repr] * 2
    )
    out_type = nn.FieldType(space, [space.regular_repr] * 3)
    dense = nn.WELinear(in_type, out_type, execution="dense").double()
    hybrid = nn.WELinear(in_type, out_type, execution="auto_hybrid").double()
    hybrid.load_state_dict(dense.state_dict(), strict=True)
    assert [pair.auto_uses_direct() for pair in hybrid._pairs] == [True, False]

    dense_input = torch.randn(
        2, 3, in_type.size, dtype=torch.float64, requires_grad=True
    )
    hybrid_input = dense_input.detach().clone().requires_grad_()
    dense_output = dense(dense_input)
    hybrid_output = hybrid(hybrid_input)
    torch.testing.assert_close(hybrid_output, dense_output, atol=3e-12, rtol=3e-12)

    dense_output.square().sum().backward()
    hybrid_output.square().sum().backward()
    torch.testing.assert_close(
        hybrid_input.grad, dense_input.grad, atol=3e-11, rtol=3e-11
    )
    for dense_parameter, hybrid_parameter in zip(
        dense.parameters(), hybrid.parameters()
    ):
        torch.testing.assert_close(
            hybrid_parameter.grad,
            dense_parameter.grad,
            atol=4e-11,
            rtol=4e-11,
        )


def test_default_is_dense_and_auto_is_conservative_whole_layer_selection():
    space = _space()
    in_type = nn.FieldType(
        space, [space.trivial_repr] * 16 + [space.regular_repr] * 2
    )
    out_type = nn.FieldType(space, [space.regular_repr] * 3)
    default = nn.WELinear(in_type, out_type)
    assert default.execution == "dense"

    automatic = nn.WELinear(in_type, out_type, execution="auto")
    x = torch.randn(4, in_type.size)
    automatic(x)
    assert automatic._inference_weight.numel() == 0
    with torch.no_grad():
        automatic(x)
    assert automatic._inference_weight.shape == (out_type.size, in_type.size)


def test_regular_inverse_permutation_is_cached_without_large_group_duplication():
    small = _space()
    small_layer = nn.WELinear(
        small.regular_repr, small.regular_repr, execution="direct"
    )
    assert small_layer._pairs[0].inverse_relative.shape == (12, 12)

    large = _space(n=128)
    large_layer = nn.WELinear(
        large.regular_repr, large.regular_repr, execution="direct"
    )
    assert large_layer._pairs[0].relative.shape == (256, 256)
    assert large_layer._pairs[0].inverse_relative.numel() == 0


def test_execution_validation_auto_policy_and_parameterless_map():
    space = _space()
    scalar, vector, other = space.trivial_repr, _irrep(space, 1), _irrep(space, 2)
    with pytest.raises(ValueError, match="execution"):
        nn.WELinear(scalar, scalar, execution="unknown")
    channel_layer = nn.WELinear(scalar, space.regular_repr, execution="auto")
    assert all(pair.auto_uses_direct() for pair in channel_layer._pairs)
    convolution = nn.WELinear(
        space.regular_repr, space.regular_repr, execution="auto"
    )
    assert not convolution._pairs[0].auto_uses_direct()
    zero = nn.WELinear(vector, other, execution="direct", bias=True).double()
    x = torch.randn(2, vector.size, dtype=torch.float64)
    assert torch.equal(zero(x), torch.zeros(2, other.size, dtype=torch.float64))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_direct_execution_torch_compile_smoke():
    space = _space()
    in_type = nn.FieldType(space, [space.trivial_repr] * 4)
    out_type = nn.FieldType(space, [space.regular_repr] * 2)
    layer = nn.WELinear(in_type, out_type, execution="direct")
    compiled = torch.compile(layer, backend="eager")
    x = torch.randn(3, in_type.size, requires_grad=True)
    expected = layer(x)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None
