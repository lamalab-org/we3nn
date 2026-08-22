"""Benchmark grouped multiplicity tensor products against legacy paths.

The legacy implementation is selected by supplying the fully connected logical
instructions explicitly. Large legacy multiplicities are disabled by default
because their purpose is precisely to demonstrate the Python object explosion;
pass ``--legacy-max 128`` only on a machine where constructing millions of
modules is acceptable.
"""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
import warnings

import torch

from we3nn import gspaces, nn
from we3nn.nn.representation_tensor import MissingRepresentationMetadataWarning


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median_milliseconds(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(torch.tensor(samples).median())


def construct(multiplicity: int, *, legacy: bool, device: torch.device):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    e1, e2 = space.irrep(1, 1), space.irrep(1, 2)
    left = nn.FieldType(space, [e1] * multiplicity)
    right = nn.FieldType(space, [e1] * multiplicity)
    output = nn.FieldType(space, [e2] * multiplicity)
    if legacy:
        compact = nn.TensorProduct(left, right, output, internal_weights=False)
        instructions = list(compact.instructions)
        del compact
        module = nn.TensorProduct(left, right, output, instructions=instructions)
    else:
        module = nn.TensorProduct(left, right, output)
    return module.to(device)


def benchmark(
    multiplicity: int,
    *,
    legacy: bool,
    batch_size: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    gc.collect()
    tracemalloc.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    start = time.perf_counter_ns()
    module = construct(multiplicity, legacy=legacy, device=device)
    synchronize()
    construction_ms = (time.perf_counter_ns() - start) / 1e6
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    device_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    left = torch.randn(batch_size, module.in1_type.size, device=device, requires_grad=True)
    right = torch.randn(batch_size, module.in2_type.size, device=device, requires_grad=True)
    def forward():
        return module(left, right)

    def forward_backward():
        module.zero_grad(set_to_none=True)
        left.grad = right.grad = None
        forward().square().mean().backward()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingRepresentationMetadataWarning)
        forward()
        forward_backward()
        forward_ms = median_milliseconds(forward, repeats)
        forward_backward_ms = median_milliseconds(forward_backward, repeats)

    buffers = [buffer for buffer in module.buffers() if buffer.numel()]
    tensor_storage = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*tuple(module.parameters()), *tuple(buffers))
    )
    return {
        "implementation": "legacy" if legacy else "grouped",
        "m": multiplicity,
        "constructor_ms": construction_ms,
        "forward_ms": forward_ms,
        "forward_backward_ms": forward_backward_ms,
        "python_peak_mib": peak_python / 2**20,
        "device_peak_mib": device_peak / 2**20,
        "tensor_storage_mib": tensor_storage / 2**20,
        "modules": sum(1 for _ in module.modules()),
        "grouped_blocks": len(module.blocks),
        "legacy_paths": len(module.paths),
        "buffers": len(buffers),
        "coupling_buffers": sum(
            name.endswith("coupling_basis") and buffer.numel() > 0
            for name, buffer in module.named_buffers()
        ),
        "weight_numel": module.weight_numel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplicities", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--legacy-max", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(
        "implementation,m,constructor_ms,forward_ms,forward_backward_ms,"
        "python_peak_mib,device_peak_mib,tensor_storage_mib,modules,grouped_blocks,legacy_paths,buffers,"
        "coupling_buffers,weight_numel"
    )
    for multiplicity in args.multiplicities:
        modes = (False, True) if multiplicity <= args.legacy_max else (False,)
        for legacy in modes:
            result = benchmark(
                multiplicity,
                legacy=legacy,
                batch_size=args.batch_size,
                repeats=args.repeats,
                device=device,
            )
            print(",".join(str(result[key]) for key in result))


if __name__ == "__main__":
    main()
