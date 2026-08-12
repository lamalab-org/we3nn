# D6 message-trunk benchmark

`linear_vs_escnn.py` builds the four-linear, three-ReLU regular-representation
trunk from `mp_example.py`. It uses a 2048-edge batch, 16 scalar input
channels, 32 hidden channels, float32 CPU tensors, and one PyTorch thread.

Reference run on Apple Silicon with PyTorch 2.13.0 and escnn 1.0.11:

| implementation | median forward | persistent tensor storage |
| --- | ---: | ---: |
| e3nn_WE | 0.700 ms | 78.7 KiB |
| escnn | 0.699 ms | 570.1 KiB |

The script exits unsuccessfully if e3nn_WE exceeds escnn by more than a 5%
timing-noise allowance or uses more persistent parameter/buffer storage.
Numbers above are one reference run; use the script for results on the current
machine and PyTorch build.
