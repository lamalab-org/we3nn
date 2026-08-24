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
import importlib
import math
import time
import tracemalloc
import warnings

import psutil
import torch

from we3nn import gspaces, nn
from we3nn.nn.representation_tensor import MissingRepresentationMetadataWarning


tensor_product_module = importlib.import_module("we3nn.nn.tensor_product")


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


_PROCESS = psutil.Process()


def process_current_rss_bytes() -> int:
    """Return current resident memory, not the process lifetime high-water mark."""
    return int(_PROCESS.memory_info().rss)


def construct(
    multiplicity: int,
    *,
    legacy: bool,
    device: torch.device,
    internal_weights: bool = True,
):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    e1, e2 = space.irrep(1, 1), space.irrep(1, 2)
    left = nn.FieldType(space, [e1] * multiplicity)
    right = nn.FieldType(space, [e1] * multiplicity)
    output = nn.FieldType(space, [e2] * multiplicity)
    if legacy:
        compact = nn.TensorProduct(left, right, output, internal_weights=False)
        instructions = list(compact.instructions)
        del compact
        module = nn.TensorProduct(
            left,
            right,
            output,
            instructions=instructions,
            internal_weights=internal_weights,
        )
    else:
        module = nn.TensorProduct(
            left, right, output, internal_weights=internal_weights
        )
    return module.to(device)


def storage_statistics(module: nn.TensorProduct) -> dict[str, int]:
    parameters = tuple(module.named_parameters())
    buffers = tuple((name, buffer) for name, buffer in module.named_buffers() if buffer.numel())
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for _, tensor in parameters
    )
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in buffers)
    parameter_names = {name for name, _ in parameters}
    persistent_buffer_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in module.state_dict().items()
        if name not in parameter_names
    )
    legacy_index_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in buffers
        if "legacy_" in name
    )
    layout_metadata_names = {
        "left_indices",
        "right_indices",
        "output_indices",
        "legacy_output_offsets",
        "legacy_left_offsets",
        "legacy_right_offsets",
    }
    layout_metadata_elements = sum(
        tensor.numel()
        for name, tensor in buffers
        if name.rsplit(".", 1)[-1] in layout_metadata_names
    )
    return {
        "parameter_bytes": parameter_bytes,
        "parameter_elements": sum(tensor.numel() for _, tensor in parameters),
        "buffer_bytes": buffer_bytes,
        "buffer_elements": sum(tensor.numel() for _, tensor in buffers),
        "persistent_buffer_bytes": persistent_buffer_bytes,
        "legacy_index_bytes": legacy_index_bytes,
        "layout_metadata_elements": layout_metadata_elements,
        "buffers": len(buffers),
    }


def grouped_cg_intermediate_bytes(
    module: nn.TensorProduct, batch_size: int, element_size: int
) -> int:
    """Size of the explicit ``[..., U, V, P, O]`` grouped CG intermediates."""
    if not module._grouped:
        return 0
    return sum(
        batch_size
        * block.left_pack.multiplicity
        * block.right_pack.multiplicity
        * block.coupling_shape[0]
        * block.output.size
        * element_size
        for block in module.blocks
        if block.kind == "cg"
    )


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
    rss_before = process_current_rss_bytes()
    start = time.perf_counter_ns()
    module = construct(multiplicity, legacy=legacy, device=device)
    synchronize()
    construction_ms = (time.perf_counter_ns() - start) / 1e6
    rss_after_construction = process_current_rss_bytes()
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    constructor_device_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

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
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        forward()
        forward_device_peak = (
            torch.cuda.max_memory_allocated(device) - baseline
            if device.type == "cuda"
            else 0
        )
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        forward_backward()
        forward_backward_device_peak = (
            torch.cuda.max_memory_allocated(device) - baseline
            if device.type == "cuda"
            else 0
        )
        forward_ms = median_milliseconds(forward, repeats)
        forward_backward_ms = median_milliseconds(forward_backward, repeats)

    storage = storage_statistics(module)
    tensor_storage = storage["parameter_bytes"] + storage["buffer_bytes"]
    return {
        "implementation": "legacy" if legacy else "grouped",
        "m": multiplicity,
        "constructor_ms": construction_ms,
        "forward_ms": forward_ms,
        "forward_backward_ms": forward_backward_ms,
        "python_peak_mib": peak_python / 2**20,
        "constructor_device_peak_mib": constructor_device_peak / 2**20,
        "forward_device_peak_mib": forward_device_peak / 2**20,
        "forward_backward_device_peak_mib": forward_backward_device_peak / 2**20,
        "process_current_rss_mib": process_current_rss_bytes() / 2**20,
        "constructor_retained_rss_delta_mib": (
            rss_after_construction - rss_before
        )
        / 2**20,
        "tensor_storage_mib": tensor_storage / 2**20,
        "parameter_mib": storage["parameter_bytes"] / 2**20,
        "parameter_elements": storage["parameter_elements"],
        "persistent_buffer_mib": storage["persistent_buffer_bytes"] / 2**20,
        "legacy_index_mib": storage["legacy_index_bytes"] / 2**20,
        "registered_buffer_elements": storage["buffer_elements"],
        "layout_metadata_elements": storage["layout_metadata_elements"],
        "grouped_cg_intermediate_mib": grouped_cg_intermediate_bytes(
            module, batch_size, left.element_size()
        )
        / 2**20,
        "modules": sum(1 for _ in module.modules()),
        "grouped_blocks": len(module.blocks),
        "legacy_paths": len(module.paths),
        "buffers": storage["buffers"],
        "coupling_buffers": sum(
            name.endswith("coupling_basis") and buffer.numel() > 0
            for name, buffer in module.named_buffers()
        ),
        "weight_numel": module.weight_numel,
    }


def benchmark_external_constructor(
    multiplicity: int, *, device: torch.device
) -> dict[str, float | int | str]:
    """Measure metadata construction without allocating the O(m^3) weights."""
    gc.collect()
    tracemalloc.start()
    rss_before = process_current_rss_bytes()
    start = time.perf_counter_ns()
    module = construct(
        multiplicity,
        legacy=False,
        device=device,
        internal_weights=False,
    )
    synchronize()
    construction_ms = (time.perf_counter_ns() - start) / 1e6
    rss_after = process_current_rss_bytes()
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    storage = storage_statistics(module)
    return {
        "implementation": "grouped-external-constructor",
        "m": multiplicity,
        "constructor_ms": construction_ms,
        "forward_ms": float("nan"),
        "forward_backward_ms": float("nan"),
        "python_peak_mib": peak_python / 2**20,
        "constructor_device_peak_mib": 0.0,
        "forward_device_peak_mib": float("nan"),
        "forward_backward_device_peak_mib": float("nan"),
        "process_current_rss_mib": rss_after / 2**20,
        "constructor_retained_rss_delta_mib": (rss_after - rss_before) / 2**20,
        "tensor_storage_mib": (storage["parameter_bytes"] + storage["buffer_bytes"]) / 2**20,
        "parameter_mib": storage["parameter_bytes"] / 2**20,
        "parameter_elements": storage["parameter_elements"],
        "persistent_buffer_mib": storage["persistent_buffer_bytes"] / 2**20,
        "legacy_index_mib": storage["legacy_index_bytes"] / 2**20,
        "registered_buffer_elements": storage["buffer_elements"],
        "layout_metadata_elements": storage["layout_metadata_elements"],
        "grouped_cg_intermediate_mib": float("nan"),
        "modules": sum(1 for _ in module.modules()),
        "grouped_blocks": len(module.blocks),
        "legacy_paths": len(module.paths),
        "buffers": storage["buffers"],
        "coupling_buffers": sum(
            name.endswith("coupling_basis") and buffer.numel() > 0
            for name, buffer in module.named_buffers()
        ),
        "weight_numel": module.weight_numel,
    }


def construct_edge_product(
    left_multiplicity: int,
    right_multiplicity: int,
    output_multiplicity: int,
    *,
    device: torch.device,
):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    e1, e2 = space.irrep(1, 1), space.irrep(1, 2)
    return nn.TensorProduct(
        nn.FieldType(space, [e1] * left_multiplicity),
        nn.FieldType(space, [e1] * right_multiplicity),
        nn.FieldType(space, [e2] * output_multiplicity),
    ).to(device)


def message_passing_estimate(
    edge_count: int,
    module: nn.TensorProduct,
    *,
    budget_bytes: int,
) -> dict[str, float | int]:
    block = module.blocks[0]
    plan = tensor_product_module._cg_chunk_plan(
        batch_size=edge_count,
        left_multiplicity=block.left_pack.multiplicity,
        right_multiplicity=block.right_pack.multiplicity,
        coupling_multiplicity=block.coupling_shape[0],
        output_size=block.output.size,
        element_size=block.weight.element_size(),
        max_intermediate_bytes=budget_bytes,
    )
    chunks = (
        math.ceil(edge_count / plan.batch_chunk)
        * math.ceil(plan.left_multiplicity / plan.left_chunk)
        * math.ceil(plan.right_multiplicity / plan.right_chunk)
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in module.parameters()
    )
    return {
        "edges": edge_count,
        "weight_numel": module.weight_numel,
        "parameter_mib": parameter_bytes / 2**20,
        "shared_external_weight_mib": parameter_bytes / 2**20,
        "per_edge_external_weight_mib": edge_count * parameter_bytes / 2**20,
        "unbounded_cg_temporary_mib": plan.estimated_unchunked_bytes / 2**20,
        "bounded_cg_temporary_mib": plan.estimated_chunk_bytes / 2**20,
        "chunk_contractions": chunks,
        "batch_chunk": plan.batch_chunk,
        "left_chunk": plan.left_chunk,
        "right_chunk": plan.right_chunk,
    }


def benchmark_edge_mode(
    module: nn.TensorProduct,
    edge_count: int,
    *,
    budget_bytes: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    left = torch.randn(
        edge_count, module.in1_type.size, device=device, requires_grad=True
    )
    right = torch.randn(
        edge_count, module.in2_type.size, device=device, requires_grad=True
    )
    previous_budget = tensor_product_module._CG_MAX_INTERMEDIATE_BYTES
    tensor_product_module._CG_MAX_INTERMEDIATE_BYTES = budget_bytes

    def forward():
        return module(left, right)

    def forward_backward():
        module.zero_grad(set_to_none=True)
        left.grad = right.grad = None
        forward().square().mean().backward()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingRepresentationMetadataWarning)
            forward()
            forward_backward()
            if device.type == "cuda":
                baseline_allocated = torch.cuda.memory_allocated(device)
                baseline_reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            forward_ms = median_milliseconds(forward, repeats)
            forward_backward_ms = median_milliseconds(forward_backward, repeats)
            if device.type == "cuda":
                allocated_peak = max(
                    0, torch.cuda.max_memory_allocated(device) - baseline_allocated
                )
                reserved_peak = max(
                    0, torch.cuda.max_memory_reserved(device) - baseline_reserved
                )
            else:
                allocated_peak = reserved_peak = 0
    finally:
        tensor_product_module._CG_MAX_INTERMEDIATE_BYTES = previous_budget
    return {
        "forward_ms": forward_ms,
        "forward_backward_ms": forward_backward_ms,
        "cuda_allocated_peak_mib": allocated_peak / 2**20,
        "cuda_reserved_peak_mib": reserved_peak / 2**20,
        "process_current_rss_mib": process_current_rss_bytes() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplicities", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument(
        "--constructor-multiplicities",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument("--legacy-max", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge-counts", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--edge-left-multiplicity", type=int, default=128)
    parser.add_argument("--edge-right-multiplicity", type=int, default=128)
    parser.add_argument("--edge-output-multiplicity", type=int, default=8)
    parser.add_argument("--edge-budget-mib", type=float, default=256.0)
    parser.add_argument(
        "--edge-run-max",
        type=int,
        default=0,
        help="execute bounded and unbounded modes only for edge counts at or below this value",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    print(
        "implementation,m,constructor_ms,forward_ms,forward_backward_ms,"
        "python_peak_mib,constructor_device_peak_mib,forward_device_peak_mib,"
        "forward_backward_device_peak_mib,process_current_rss_mib,"
        "constructor_retained_rss_delta_mib,"
        "tensor_storage_mib,parameter_mib,parameter_elements,persistent_buffer_mib,"
        "legacy_index_mib,registered_buffer_elements,layout_metadata_elements,"
        "grouped_cg_intermediate_mib,"
        "modules,grouped_blocks,legacy_paths,buffers,"
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
    for multiplicity in args.constructor_multiplicities:
        result = benchmark_external_constructor(multiplicity, device=device)
        print(",".join(str(result[key]) for key in result))

    edge_module = construct_edge_product(
        args.edge_left_multiplicity,
        args.edge_right_multiplicity,
        args.edge_output_multiplicity,
        device=device,
    )
    edge_budget = int(args.edge_budget_mib * 2**20)
    print(
        "edge_mode,edges,weight_numel,parameter_mib,shared_external_weight_mib,"
        "per_edge_external_weight_mib,unbounded_cg_temporary_mib,"
        "bounded_cg_temporary_mib,chunk_contractions,batch_chunk,left_chunk,right_chunk,"
        "forward_ms,forward_backward_ms,cuda_allocated_peak_mib,"
        "cuda_reserved_peak_mib,process_current_rss_mib"
    )
    for edge_count in args.edge_counts:
        estimate = message_passing_estimate(
            edge_count, edge_module, budget_bytes=edge_budget
        )
        execution_modes = ()
        if edge_count <= args.edge_run_max:
            execution_modes = (
                ("bounded", edge_budget),
                ("unbounded", estimate["unbounded_cg_temporary_mib"] * 2**20 + 1),
            )
        if not execution_modes:
            execution_modes = (("estimate", None),)
        for mode, mode_budget in execution_modes:
            execution = (
                {
                    "forward_ms": float("nan"),
                    "forward_backward_ms": float("nan"),
                    "cuda_allocated_peak_mib": float("nan"),
                    "cuda_reserved_peak_mib": float("nan"),
                    "process_current_rss_mib": process_current_rss_bytes() / 2**20,
                }
                if mode_budget is None
                else benchmark_edge_mode(
                    edge_module,
                    edge_count,
                    budget_bytes=int(mode_budget),
                    repeats=args.repeats,
                    device=device,
                )
            )
            row = {"edge_mode": mode, **estimate, **execution}
            print(",".join(str(row[key]) for key in row))


if __name__ == "__main__":
    main()
