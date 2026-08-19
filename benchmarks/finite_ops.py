"""Construction, forward, and backward benchmarks for core finite operations."""

from __future__ import annotations

import statistics
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from we3nn import group


def timed(function, repeats=30):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def storage(module):
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


def benchmark_linear(fields):
    symmetry = group.DihedralGroup(6)
    representation = fields * symmetry.regular_representation()
    start = time.perf_counter_ns()
    module = group.nn.WELinear(representation, representation)
    construction = (time.perf_counter_ns() - start) / 1e6
    tensor = torch.randn(512, representation.dim, requires_grad=True)
    value = tensor
    forward = timed(lambda: module(value))

    def backward():
        module.zero_grad(set_to_none=True)
        tensor.grad = None
        module(value).square().mean().backward()

    backward_ms = timed(backward, repeats=10)
    print(
        f"WELinear {fields} regular fields: construct={construction:.3f} ms, "
        f"forward={forward:.3f} ms, backward={backward_ms:.3f} ms, "
        f"parameters={sum(p.numel() for p in module.parameters())}, storage={storage(module)/1024:.1f} KiB"
    )


def benchmark_tensor_product():
    symmetry = group.DihedralGroup(6)
    vector = symmetry.standard_representation()
    regular = symmetry.regular_representation()
    for name, left, right, output in (
        ("E1 x E1", vector, vector, regular),
        ("regular x E1", regular, vector, regular),
    ):
        start = time.perf_counter_ns()
        module = group.nn.TensorProduct(left, right, output)
        construction = (time.perf_counter_ns() - start) / 1e6
        x = torch.randn(512, left.dim)
        y = torch.randn(512, right.dim)
        print(
            f"TensorProduct {name}: construct={construction:.3f} ms, "
            f"forward={timed(lambda: module(x,y)):.3f} ms, "
            f"parameters={sum(p.numel() for p in module.parameters())}, storage={storage(module)/1024:.1f} KiB"
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    for count in (1, 4, 16):
        benchmark_linear(count)
    benchmark_tensor_product()
