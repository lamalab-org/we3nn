# e3nn_WE

Validated development base: e3nn `0.6.0` (upstream tag commit
`2aa7f58440a06b15352a2cbce01fa4c26f824969`) and PyTorch `2.13.0`. The finite
subsystem is parallel to `e3nn.o3`; it does not modify O(3)'s `Irrep`,
`Irreps`, or `TensorProduct` behavior.

The public API is available directly beside upstream e3nn:

```python
from e3nn import group

G = group.DihedralGroup(6)
layer = group.nn.Linear(G.standard_representation(), G.regular_representation())
```

All finite-group modules consume and return ordinary `torch.Tensor` objects.
Representation metadata is stored on the modules; no tensor wrapper is part
of the library API.

The extension applies e3nn's representation approach to the finite planar
groups C_n (rotations) and D_n (rotations and reflections). `mp_example.py`
uses `e3nn.group` directly and does not require escnn-style spaces, field
types, or typed tensor wrappers.

The implementation uses real irreducible representations, complete
intertwiner bases for linear maps, invariant biases, and literal permutation
regular representations. Applying ReLU, ELU, or another scalar function to a
regular representation is therefore exactly equivariant.

It also supplies the parts of e3nn's representation algebra needed to build
nonlinear equivariant models:

- normalized real Clebsch--Gordan coupling tensors for all C_n/D_n irrep
  triples;
- `TensorProduct` and `FullyConnectedTensorProduct`, with internal, shared
  external, or per-sample external weights;
- circular harmonics `cos(k theta), sin(k theta)`, the natural angular basis
  for finite subgroups of O(2);
- ordinary 3D spherical harmonics computed by `e3nn.o3.spherical_harmonics`
  and restricted to the C_n/D_n action which fixes the z axis.

## Example

```python
import torch
from e3nn import group

G = group.DihedralGroup(6)
scalar = G.trivial_irrep
vector = G.standard_representation()
scalars_and_vectors = 8 * scalar + 2 * vector
regular = 4 * G.regular_representation()

model = torch.nn.Sequential(
    group.nn.Linear(scalars_and_vectors, regular),
    group.nn.PointwiseActivation(regular, torch.relu),
    group.nn.Linear(regular, scalars_and_vectors),
)
x = torch.randn(32, scalars_and_vectors.dim)
y = model(x)
```

## Harmonics and tensor products

```python
from e3nn import group

# Angular O(2) harmonics through the non-aliased D6 frequency limit.
angular = group.CircularHarmonics(G, max_frequency=3)
y_theta = angular(torch.linspace(0, 2 * torch.pi, 32))

# Actual e3nn O(3) harmonics, with their representation restricted to D6.
spatial = group.RestrictedSphericalHarmonics(G, degrees=[0, 1, 2, 3])
y_lm = spatial(torch.randn(32, 3))

product = group.nn.FullyConnectedTensorProduct(
    scalars_and_vectors,
    scalars_and_vectors,
    regular,
)
z = product(x, x)
```

Restricted Wigner--Eckart products can own the spherical-harmonic evaluator
and expose the complete sampled matrix-valued kernel basis:

```python
restricted_tp = group.nn.RestrictedWignerEckartTensorProduct(
    vector,
    spatial,
    vector,
)

weights = radial_mlp(radial_features)
messages = restricted_tp.forward_from_points(node_features, edge_vectors, weights)
kernel_basis = restricted_tp.sample_kernel_basis(edge_vectors)
```

The sampled basis has shape
`(..., weight_numel, output_dim, input_dim)`. It contains the full finite-group
coupling space, not only paths inherited from the O(3) parent symmetry.

`clebsch_gordan(left, right, output)` returns one normalized tensor for each
representation copy, with shape
`(multiplicity, output_dim, left_dim, right_dim)`.
`tensor_product_multiplicity(left, right, output)` returns that copy
multiplicity directly. `finite_group_couplings(left, right, output)` instead
returns every independent real Hom-space path. For a two-dimensional C_n
irrep, one representation copy corresponds to two such real paths because
its real endomorphism algebra is the complex numbers. Neural tensor products
use this complete real Hom space.

For C_n use `group.CyclicGroup(n)`. Its real irreps are indexed by
`G.irrep(k)`. D_n irreps use `G.irrep(j, k)`: `(0, 0)` is scalar and `(1, 1)`
is the standard xy vector for n > 2.

The constructors `group.CyclicGroup(n)` and `group.DihedralGroup(n)` expose
all elements, irreps, the regular representation, group operations, character
tensor-product decomposition, and random sampling. `Irreps` provides an
e3nn-like multiplicity container.

`MatrixFiniteGroup` supports arbitrary finite groups from deterministic
multiplication and inverse tables. User-provided orthogonal representations
then work with `intertwiner_basis`, `Linear`, CG construction, and tensor
products without automatic character-theory or irrep enumeration.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/linear_vs_escnn.py
```

The benchmark reports median forward latency plus persistent parameter and
buffer storage. If escnn is unavailable it still reports e3nn_WE and explains
how to enable the direct comparison. escnn 1.0.x's legacy dependencies do not
currently build on Python 3.12; Python 3.10 can be used for that optional
reference environment.
