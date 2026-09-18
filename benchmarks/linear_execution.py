"""Benchmark dense, direct, and automatic WELinear execution.

Examples::

    python benchmarks/linear_execution.py --device cpu
    python benchmarks/linear_execution.py --device cuda --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from we3nn import gspaces, nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(function, device: torch.device, repeats: int) -> float:
    for _ in range(3):
        function()
    _synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        _synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def _measure(
    in_type,
    out_type,
    execution: str,
    device: torch.device,
    batch: int,
    repeats: int,
) -> dict[str, float | str]:
    layer = nn.WELinear(in_type, out_type, execution=execution).to(device)
    x = torch.randn(batch, in_type.size, device=device)

    def inference():
        with torch.no_grad():
            layer(x)

    def training_forward():
        with torch.enable_grad():
            layer(x)

    x_grad = x.detach().requires_grad_()

    def forward_backward():
        layer.zero_grad(set_to_none=True)
        x_grad.grad = None
        layer(x_grad).square().mean().backward()

    inference_ms = _median_ms(inference, device, repeats)
    training_ms = _median_ms(training_forward, device, repeats)
    backward_ms = _median_ms(forward_backward, device, max(3, repeats // 3))

    peak_mib = float("nan")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        forward_backward()
        _synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / 2**20

    return {
        "execution": execution,
        "inference_ms": inference_ms,
        "training_forward_ms": training_ms,
        "forward_backward_ms": backward_ms,
        "peak_tensor_mib": peak_mib,
    }


def _second_derivative_ms(
    in_type,
    out_type,
    execution: str,
    device: torch.device,
    batch: int,
    repeats: int,
) -> float:
    layer = nn.WELinear(in_type, out_type, execution=execution).to(device)
    x = torch.randn(batch, in_type.size, device=device, requires_grad=True)

    def workload():
        layer.zero_grad(set_to_none=True)
        force = torch.autograd.grad(layer(x).square().mean(), x, create_graph=True)[0]
        torch.autograd.grad(force.square().sum(), tuple(layer.parameters()), allow_unused=True)

    return _median_ms(workload, device, max(3, repeats // 5))


def _print_case(name: str, rows: list[dict[str, float | str]]) -> None:
    print(f"\n{name}")
    print(
        "execution     inference_ms  training_forward_ms  forward_backward_ms  "
        "peak_tensor_mib"
    )
    for row in rows:
        peak = row["peak_tensor_mib"]
        peak_text = "n/a" if isinstance(peak, float) and peak != peak else f"{peak:.2f}"
        print(
            f"{row['execution']:>12}  {row['inference_ms']:>12.4f}  "
            f"{row['training_forward_ms']:>19.4f}  "
            f"{row['forward_backward_ms']:>19.4f}  {peak_text:>15}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.set_num_threads(1)
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar, regular = space.trivial_repr, space.regular_repr
    executions = ("dense", "direct", "auto", "auto_hybrid")

    edge_in = nn.FieldType(space, [scalar] * 160 + [regular] * 6)
    edge_out = nn.FieldType(space, [regular] * 3)
    rows = [
        _measure(edge_in, edge_out, mode, device, args.batch, args.repeats)
        for mode in executions
    ]
    _print_case("A: 160 trivial + 6 regular -> 3 regular", rows)
    print("second derivative (ms):", end="")
    for mode in executions:
        value = _second_derivative_ms(
            edge_in, edge_out, mode, device, min(args.batch, 32), args.repeats
        )
        print(f"  {mode}={value:.4f}", end="")
    print()

    for channels in (1, 4, 8, 16, 32):
        type_ = nn.FieldType(space, [regular] * channels)
        rows = [
            _measure(type_, type_, mode, device, args.batch, args.repeats)
            for mode in executions
        ]
        _print_case(f"B: {channels} regular -> {channels} regular", rows)

    for scalars in (16, 64, 160, 512):
        in_type = nn.FieldType(space, [scalar] * scalars)
        out_type = nn.FieldType(space, [regular] * 3)
        rows = [
            _measure(in_type, out_type, mode, device, args.batch, args.repeats)
            for mode in executions
        ]
        _print_case(f"C: {scalars} trivial -> 3 regular", rows)


if __name__ == "__main__":
    main()
