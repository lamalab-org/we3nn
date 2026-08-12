# e3nn_WE

`e3nn_WE` extends e3nn's typed-representation approach to the finite planar
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

model = nn.SequentialModule(
    nn.Linear(scalars_and_vectors, regular),
    nn.ReLU(regular),
    nn.Linear(regular, scalars_and_vectors),
)
x = nn.GeometricTensor(torch.randn(32, scalars_and_vectors.size), scalars_and_vectors)
y = model(x)
```

For C_n use `gspaces.rot2dOnR2(N=n)`. Its real irreps are indexed by
`space.irrep(k)`. D_n irreps use `space.irrep(j, k)`, matching escnn: `(0, 0)`
is scalar and `(1, 1)` is the standard xy vector for n > 2.

The lower-level constructors `cyclic_group(n)` and `dihedral_group(n)` expose
all elements, irreps, the regular representation, group operations, character
tensor-product decomposition, and random sampling. `Irreps` provides an
e3nn-like multiplicity container.

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
