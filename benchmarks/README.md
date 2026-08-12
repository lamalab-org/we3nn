# D6 message-trunk benchmark

`linear_vs_escnn.py` builds the four-linear, three-ReLU regular-representation
trunk from `mp_example.py`. It uses a 2048-edge batch, 16 scalar input
channels, 32 hidden channels, float32 CPU tensors, and one PyTorch thread.

Reference run on Apple Silicon with PyTorch 2.13.0 and escnn 1.0.11:

| implementation | median forward | persistent tensor storage |
| --- | ---: | ---: |
| e3nn_WE | 0.687 ms | 78.7 KiB |
| escnn | 0.696 ms | 570.1 KiB |

The script exits unsuccessfully if e3nn_WE exceeds escnn by more than a 5%
timing-noise allowance or uses more persistent parameter/buffer storage.
Numbers above are one reference run; use the script for results on the current
machine and PyTorch build.

Persistent storage counts parameters and registered buffers. The script also
reports peak Python allocations via `tracemalloc`; tensor allocator peak
memory is backend-specific and should be measured with the relevant CPU/GPU
profiler for production workloads.

`finite_ops.py` additionally benchmarks construction, forward, and backward
for 1/4/16 D6 regular-field linear maps plus E1 x E1 and regular x E1 tensor
products. It also reports learnable parameter counts and tensor storage.
The same script benchmarks construction, forward, and backward for the full
D6 message layer (including radial features and both scatter conventions).
