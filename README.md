# e3nn_WE

Validated development base: e3nn `0.6.0` (upstream tag commit
`2aa7f58440a06b15352a2cbce01fa4c26f824969`) and PyTorch `2.13.0`. The finite
subsystem is parallel to `e3nn.o3`; it does not modify O(3)'s `Irrep`,
`Irreps`, or `TensorProduct` behavior.

The preferred generic API is available directly beside upstream e3nn:

```python
from e3nn import group

G = group.DihedralGroup(6)
layer = group.nn.Linear(G.standard_representation(), G.regular_representation())
```

The historical `e3nn_WE` imports remain supported for the escnn-compatible
surface used by `mp_example.py`.

All finite-group modules consume and return ordinary `torch.Tensor` objects.
Representation metadata is stored on the modules; no tensor wrapper is part
of the library API.

`e3nn_WE` extends e3nn's representation approach to the finite planar
groups C_n (rotations) and D_n (rotations and reflections). Its no-base-space
API intentionally mirrors the small part of escnn used by `mp_example.py`, so
that example only needs this import change:

```python
from e3nn_WE import gspaces, nn as enn
```

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
from e3nn_WE import gspaces, nn

space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
scalars_and_vectors = nn.FieldType(
    space,
    8 * [space.irrep(0, 0)] + 2 * [space.irrep(1, 1)],
)
regular = nn.FieldType(space, 4 * [space.regular_repr])

model = torch.nn.Sequential(
    nn.Linear(scalars_and_vectors, regular),
    nn.ReLU(regular),
    nn.Linear(regular, scalars_and_vectors),
)
x = torch.randn(32, scalars_and_vectors.size)
y = model(x)
```

## Harmonics and tensor products

```python
from e3nn_WE import CircularHarmonics, RestrictedSphericalHarmonics

# Angular O(2) harmonics through the non-aliased D6 frequency limit.
angular = CircularHarmonics(space, max_frequency=3)
y_theta = angular(torch.linspace(0, 2 * torch.pi, 32))

# Actual e3nn O(3) harmonics, with their representation restricted to D6.
spatial = RestrictedSphericalHarmonics(space, degrees=[0, 1, 2, 3])
y_lm = spatial(torch.randn(32, 3))

product = nn.FullyConnectedTensorProduct(
    scalars_and_vectors,
    scalars_and_vectors,
    regular,
)
z = product(x, x)
```

`clebsch_gordan(left, right, output)` returns a tensor with shape
`(paths, output_dim, left_dim, right_dim)`. These are real intertwiner paths.
For a two-dimensional C_n irrep, one representation copy may correspond to
two real paths because its real endomorphism algebra is the complex numbers.

For C_n use `gspaces.rot2dOnR2(N=n)`. Its real irreps are indexed by
`space.irrep(k)`. D_n irreps use `space.irrep(j, k)`, matching escnn: `(0, 0)`
is scalar and `(1, 1)` is the standard xy vector for n > 2.

The lower-level constructors `cyclic_group(n)` and `dihedral_group(n)` expose
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
