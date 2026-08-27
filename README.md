# we3nn

Validated development base: e3nn `0.6.0` and PyTorch `2.13.0`. The finite
subsystem is parallel to `e3nn.o3`; it does not modify O(3)'s `Irrep`,
`Irreps`, or `TensorProduct` behavior.

The public API is available directly beside upstream e3nn:

```python
from we3nn import group

G = group.DihedralGroup(6)
layer = group.nn.WELinear(G.standard_representation(), G.regular_representation())
```

All finite-group modules accept ordinary `torch.Tensor` objects.
For runtime representation checks, tensors can optionally be wrapped with
`RepresentationTensor` or `FieldType.wrap()`. Tensor products warn when raw
inputs have no metadata and raise an error when wrapped metadata disagrees
with their declared input representations. Typed inputs propagate typed
outputs through `WELinear`, `PointActiv`, and tensor products.

The extension applies e3nn's representation approach to the finite planar
groups C_n (rotations) and D_n (rotations and reflections).

The implementation uses real irreducible representations, complete
intertwiner bases for linear maps, invariant biases, and literal permutation
regular representations. 

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

### Choosing a tensor product

| Module | Purpose | Weights | Output basis |
| --- | --- | --- | --- |
| `TensorProduct` | Configurable equivariant field couplings | Internal or external | Requested representations |
| `FullyConnectedTensorProduct` | Every compatible coupling path | Internal or external | Requested representations |
| `FullTensorProduct` | Complete unprojected Kronecker product | None | Product-coordinate basis |
| `KernelTensorProduct` | Fixed coupling tensors plus reduced coefficients | External | Requested representations |
| `SphericalKernelTensorProduct` | Kernels from restricted e3nn spherical harmonics | External/radial | Requested representations |

All five validate `RepresentationTensor` metadata. Raw feature tensors emit
`MissingRepresentationMetadataWarning` and continue through the compatible
raw-tensor path.

## Example

```python
import torch
from we3nn import group

G = group.DihedralGroup(6)
scalar = G.trivial_irrep
vector = G.standard_representation()
scalars_and_vectors = 8 * scalar + 2 * vector
regular = 4 * G.regular_representation()

model = torch.nn.Sequential(
    group.nn.WELinear(scalars_and_vectors, regular),
    group.nn.PointActiv(regular, torch.relu),
    group.nn.WELinear(regular, scalars_and_vectors),
)
x = torch.randn(32, scalars_and_vectors.dim)
y = model(x)
```

## Harmonics and tensor products

Omitting the band selects the full finite-group default

$$
L_{\mathrm{full}}(C_n)=\lfloor n/2\rfloor,
\qquad
L_{\mathrm{full}}(D_n)=n.
$$

The selected circular frequencies and spherical degrees are
`0, ..., L_full`. Pass `max_frequency=` or `degrees=` to request a custom
band.

```python
from we3nn import group

# Defaults to the full D6 band, frequencies 0 through 6.
angular = group.CircularHarmonics(G)
y_theta = angular(torch.linspace(0, 2 * torch.pi, 32))

# Actual e3nn O(3) harmonics, with their representation restricted to D6.
spatial = group.RestrictedSphericalHarmonics(G)
y_lm = spatial(torch.randn(32, 3))

# Custom smaller bands remain available.
angular_small = group.CircularHarmonics(G, max_frequency=3)
spatial_small = group.RestrictedSphericalHarmonics(G, degrees=[0, 1, 2, 3])

# Change to explicit finite-irrep coordinates without dropping multiplicities.
# For C6, degree 3 remains seven-dimensional and contains two k=3 copies.
C6 = group.CyclicGroup(6)
spatial_irreps = group.RestrictedSphericalHarmonics(
    C6, degrees=3, basis="finite_irreps"
)

product = group.nn.FullyConnectedTensorProduct(
    scalars_and_vectors,
    scalars_and_vectors,
    regular,
)
z = product(x, x)
```

To enable representation checking and metadata propagation:

```python
from we3nn import RepresentationTensor

typed_x = RepresentationTensor(x, scalars_and_vectors)
typed_z = product(typed_x, typed_x)
assert typed_z.field_type == product.out_type
z = typed_z.tensor
```

Passing raw tensors to a tensor product returns a raw tensor and emits
`MissingRepresentationMetadataWarning`. If only some inputs are wrapped, the
operation warns and returns a raw tensor. When every representation-carrying
input is wrapped, the result is a `RepresentationTensor`.

Spherical kernel tensor products can own the restricted spherical-harmonic
evaluator and expose the complete sampled matrix-valued kernel basis:

```python
spherical_tp = group.nn.SphericalKernelTensorProduct(
    vector,
    spatial,
    vector,
)

weights = radial_mlp(radial_features)
messages = spherical_tp.forward_from_points(node_features, edge_vectors, weights)
kernel_basis = spherical_tp.sample_kernel_basis(edge_vectors)
```

Connection modes can also be selected per representation-triple multiplicity
block. Chunk indices are inspectable through `in1_chunks`, `in2_chunks`, and
`out_chunks`; they never refer to individual fields.

```python
mixed = group.nn.KernelTensorProduct(
    node_type,
    edge_type,
    output_type,
    block_instructions=[
        # Preserve the E1 node-channel index: W = U is required.
        group.nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        # Independently mix scalar inputs into every E1 output channel.
        group.nn.TensorProductBlockInstruction(1, 1, 0, connection_mode="uvw"),
    ],
)

for index, chunk in enumerate(mixed.in1_chunks):
    print(index, chunk.representation.name, chunk.multiplicity)
print(mixed.weight_layout)
```

For explicit mixed blocks, the flattened reduced-weight vector is the
concatenation of block weights in instruction order. Each `weight_layout`
entry reports the block-local shape and slice. Reordering block instructions
therefore changes external-weight and checkpoint semantics. Legacy implicit
`"uvw"` products retain their historical flattened ordering.

Equal-representation fields can optionally be split into independently
addressable multiplicity subchunks. `FieldType` still specifies how every
field transforms; `MultiplicityChunkSpec` supplies a validated field-index
partition; the compiled read-only `MultiplicityChunk` adds the inferred
representation and coordinate starts. `TensorProductBlockInstruction` then
selects one left, right, and output chunk.

```python
in_chunks = [
    group.nn.MultiplicityChunkSpec(tuple(range(0, 64))),
    group.nn.MultiplicityChunkSpec(tuple(range(64, 128))),
]
out_chunks = [
    group.nn.MultiplicityChunkSpec(tuple(range(0, 64))),
    group.nn.MultiplicityChunkSpec(tuple(range(64, 128))),
]

tp = group.nn.TensorProduct(
    in_type,
    filter_type,
    out_type,
    in1_chunks=in_chunks,
    out_chunks=out_chunks,
    block_instructions=[
        group.nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        group.nn.TensorProductBlockInstruction(1, 0, 1, connection_mode="uvu"),
    ],
)
```

Both input chunks above carry the same representation, but have independent
channel wiring and parameters. Explicit specs must cover every field exactly
once. Their chunk order and the field order inside each chunk are preserved.
Automatic chunking is unchanged when no specs are supplied: all occurrences
of an equal representation still form one chunk, including noncontiguous
occurrences. Custom chunks use block-contiguous reduced-weight layout; the
historical implicit `"uvw"` layout remains reserved for automatic chunks.

The sampled basis has shape
`(..., weight_numel, output_dim, input_dim)`. `clebsch_gordan(left, right, output)` returns one normalized tensor for each
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
then work with `intertwiner_basis`, `WELinear`, CG construction, and tensor
products without automatic character-theory or irrep enumeration.

## Development

```bash
python -m pip install -e '.[test]'
pytest
python benchmarks/linear_vs_escnn.py
```

The benchmark reports median forward latency plus persistent parameter and
buffer storage. If escnn is unavailable it still reports we3nn and explains
how to enable the direct comparison. escnn 1.0.x's legacy dependencies do not
currently build on Python 3.12; Python 3.10 can be used for that optional
reference environment.
