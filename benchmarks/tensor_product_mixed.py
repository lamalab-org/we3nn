"""Benchmark mixed ``uvw``/``uvu`` grouped tensor-product plans.

Constructor/forward scaling is measured for every requested channel count.
Kernel-basis sampling is limited by ``--basis-max-channels`` because its
materialized output necessarily scales as ``weight_numel * out_dim * in_dim``.
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

from we3nn import MissingRepresentationMetadataWarning, gspaces, nn


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def milliseconds(function, repeats: int, device: torch.device) -> float:
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        function()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(torch.tensor(samples).median())


def construct(channels: int, device: torch.device) -> nn.KernelTensorProduct:
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar, e1 = space.trivial_repr, space.irrep(1, 1)
    scalar_channels = max(1, channels // 2)
    features = nn.FieldType(space, [e1] * channels + [scalar] * scalar_channels)
    filters = nn.FieldType(space, [scalar, e1])
    output = nn.FieldType(space, [e1] * channels + [scalar] * scalar_channels)
    instructions = [
        nn.TensorProductBlockInstruction(0, 0, 0, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(1, 0, 1, connection_mode="uvu"),
        nn.TensorProductBlockInstruction(1, 1, 0, connection_mode="uvw"),
        nn.TensorProductBlockInstruction(0, 1, 1, connection_mode="uvw"),
    ]
    return nn.KernelTensorProduct(
        features,
        filters,
        output,
        block_instructions=instructions,
        shared_weights=False,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--basis-max-channels", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=MissingRepresentationMetadataWarning)
    device = torch.device(args.device)
    print(
        "channels,constructor_ms,forward_ms,kernel_basis_ms,weight_numel,"
        "logical_paths,modules,blocks,paths,coupling_buffers,basis_elements"
    )
    for channels in args.channels:
        synchronize(device)
        start = time.perf_counter_ns()
        kernel = construct(channels, device)
        synchronize(device)
        constructor_ms = (time.perf_counter_ns() - start) / 1e6
        features = torch.randn(
            args.batch_size, kernel.in1_type.size, device=device
        )
        filters = torch.randn(args.batch_size, kernel.in2_type.size, device=device)
        weights = torch.randn(
            args.batch_size, kernel.weight_numel, device=device
        )
        forward_ms = milliseconds(
            lambda: kernel(features, filters, weights), args.repeats, device
        )
        if channels <= args.basis_max_channels:
            basis_ms = milliseconds(
                lambda: kernel.sample_kernel_basis(filters), args.repeats, device
            )
            basis_elements = (
                args.batch_size
                * kernel.weight_numel
                * kernel.out_type.size
                * kernel.in1_type.size
            )
        else:
            basis_ms = float("nan")
            basis_elements = 0
        product = kernel.tensor_product
        coupling_buffers = sum(
            name.endswith("coupling_basis") and buffer.numel() > 0
            for name, buffer in product.named_buffers()
        )
        print(
            f"{channels},{constructor_ms},{forward_ms},{basis_ms},"
            f"{kernel.weight_numel},{len(product.instructions)},"
            f"{sum(1 for _ in product.modules())},{len(product.blocks)},"
            f"{len(product.paths)},{coupling_buffers},{basis_elements}"
        )


if __name__ == "__main__":
    main()
