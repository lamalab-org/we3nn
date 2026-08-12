"""Construction, forward, and backward benchmarks for core finite operations."""

from __future__ import annotations

import statistics
import time

import torch

from e3nn_WE import gspaces, nn


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
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    type_ = nn.FieldType(space, fields * [space.regular_repr])
    start = time.perf_counter_ns()
    module = nn.Linear(type_, type_)
    construction = (time.perf_counter_ns() - start) / 1e6
    tensor = torch.randn(512, type_.size, requires_grad=True)
    value = nn.GeometricTensor(tensor, type_)
    forward = timed(lambda: module(value))

    def backward():
        module.zero_grad(set_to_none=True)
        tensor.grad = None
        module(value).tensor.square().mean().backward()

    backward_ms = timed(backward, repeats=10)
    print(
        f"Linear {fields} regular fields: construct={construction:.3f} ms, "
        f"forward={forward:.3f} ms, backward={backward_ms:.3f} ms, "
        f"parameters={sum(p.numel() for p in module.parameters())}, storage={storage(module)/1024:.1f} KiB"
    )


def benchmark_tensor_product():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    vector = nn.FieldType(space, [space.irrep(1, 1)])
    regular = nn.FieldType(space, [space.regular_repr])
    for name, left, right, output in (
        ("E1 x E1", vector, vector, regular),
        ("regular x E1", regular, vector, regular),
    ):
        start = time.perf_counter_ns()
        module = nn.TensorProduct(left, right, output)
        construction = (time.perf_counter_ns() - start) / 1e6
        x = nn.GeometricTensor(torch.randn(512, left.size), left)
        y = nn.GeometricTensor(torch.randn(512, right.size), right)
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
