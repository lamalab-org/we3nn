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
    module = group.nn.Linear(representation, representation)
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
        f"Linear {fields} regular fields: construct={construction:.3f} ms, "
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


def benchmark_message_layer():
    from mp_example import EquivariantMPLayer

    nodes, edges = 256, 1024
    start = time.perf_counter_ns()
    module = EquivariantMPLayer(16, 32, torch.nn.ReLU())
    construction = (time.perf_counter_ns() - start) / 1e6
    h = torch.randn(nodes, 16, requires_grad=True)
    pos = torch.randn(nodes, 3, requires_grad=True)
    anchor = torch.randn(nodes, 2, requires_grad=True)
    edge_index = torch.randint(nodes, (2, edges))
    forward = timed(lambda: module(h, pos, anchor, edge_index), repeats=10)

    def backward():
        module.zero_grad(set_to_none=True)
        h.grad = pos.grad = anchor.grad = None
        scalar, force = module(h, pos, anchor, edge_index)
        (scalar.square().mean() + force.square().mean()).backward()

    backward_ms = timed(backward, repeats=5)
    print(
        f"D6 message layer: construct={construction:.3f} ms, forward={forward:.3f} ms, "
        f"backward={backward_ms:.3f} ms, parameters={sum(p.numel() for p in module.parameters())}, "
        f"storage={storage(module)/1024:.1f} KiB"
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    for count in (1, 4, 16):
        benchmark_linear(count)
    benchmark_tensor_product()
    benchmark_message_layer()
