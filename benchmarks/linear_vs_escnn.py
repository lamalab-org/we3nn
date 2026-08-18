"""Compare the D6 WELinear used by the message-passing example.

Run in an environment containing the optional ``reference`` dependencies.
The timing deliberately excludes construction and uses one CPU thread.
"""

from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from we3nn import group


def storage_bytes(module: torch.nn.Module) -> int:
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def median_forward_ms(callable_, warmup: int = 20, repeats: int = 100) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            callable_()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            callable_()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def peak_python_bytes(callable_) -> int:
    """Peak Python allocation; tensor storage is reported separately."""
    tracemalloc.start()
    callable_()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    edges, in_channels, hidden_channels = 2048, 16, 32
    scalar_inputs = 2 * in_channels + 1 + 16 + 1

    symmetry = group.DihedralGroup(6)
    our_in = scalar_inputs * symmetry.trivial_irrep + 2 * symmetry.standard_representation()
    our_out = (hidden_channels // 2) * symmetry.regular_representation()
    our_layer = torch.nn.Sequential(
        group.nn.WELinear(our_in, our_out),
        group.nn.PointActiv(our_out, torch.relu),
        group.nn.WELinear(our_out, our_out),
        group.nn.PointActiv(our_out, torch.relu),
        group.nn.WELinear(our_out, our_out),
        group.nn.PointActiv(our_out, torch.relu),
        group.nn.WELinear(our_out, our_out),
    )
    tensor = torch.randn(edges, our_in.dim)
    our_input = tensor
    our_ms = median_forward_ms(lambda: our_layer(our_input))
    our_storage = storage_bytes(our_layer)
    our_python_peak = peak_python_bytes(lambda: our_layer(our_input))
    print(
        f"we3nn: {our_ms:.3f} ms, {our_storage / 1024:.1f} KiB persistent storage, "
        f"{our_python_peak / 1024:.1f} KiB peak Python allocation"
    )

    try:
        from escnn import gspaces as esc_gspaces, nn as esc_nn
    except ImportError:
        print("escnn: unavailable; install the 'reference' extra in a compatible Python environment")
        return

    esc_space = esc_gspaces.no_base_space(esc_gspaces.flipRot2dOnR2(N=6).fibergroup)
    esc_in = esc_nn.FieldType(esc_space, scalar_inputs * [esc_space.irrep(0, 0)] + 2 * [esc_space.irrep(1, 1)])
    esc_out = esc_nn.FieldType(esc_space, (hidden_channels // 2) * [esc_space.regular_repr])
    esc_layer = esc_nn.SequentialModule(
        esc_nn.Linear(esc_in, esc_out),
        esc_nn.ReLU(esc_out, inplace=False),
        esc_nn.Linear(esc_out, esc_out),
        esc_nn.ReLU(esc_out, inplace=False),
        esc_nn.Linear(esc_out, esc_out),
        esc_nn.ReLU(esc_out, inplace=False),
        esc_nn.Linear(esc_out, esc_out),
    )
    esc_input = esc_nn.GeometricTensor(tensor, esc_in)
    esc_ms = median_forward_ms(lambda: esc_layer(esc_input))
    esc_storage = storage_bytes(esc_layer)
    esc_python_peak = peak_python_bytes(lambda: esc_layer(esc_input))
    print(
        f"escnn:  {esc_ms:.3f} ms, {esc_storage / 1024:.1f} KiB persistent storage, "
        f"{esc_python_peak / 1024:.1f} KiB peak Python allocation"
    )
    print(f"ratios: {our_ms / esc_ms:.3f}x time, {our_storage / esc_storage:.3f}x storage")

    if our_ms > esc_ms * 1.05:
        raise SystemExit("we3nn is more than 5% slower than escnn")
    if our_storage > esc_storage:
        raise SystemExit("we3nn uses more persistent tensor storage than escnn")


if __name__ == "__main__":
    main()
