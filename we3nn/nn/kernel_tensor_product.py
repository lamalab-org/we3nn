"""Finite-group tensor products specialized for steerable kernels."""

from __future__ import annotations

import torch
from torch import nn

from ..embedding import RestrictedO3Representation
from ..harmonics import RestrictedSphericalHarmonics
from ..representations import Representation
from .field_type import FieldType, as_field_type
from .tensor_product import TensorProduct
from .representation_tensor import (
    RepresentationTensor,
    unpack_tensor_product_inputs,
)


class KernelTensorProduct(nn.Module):
    r"""Tensor product specialized for externally sampled filter features.

    ``features`` transform under ``rep_in`` and ``filter_features`` under
    ``rep_filter``. The module constructs the complete finite-group coupling
    basis

    .. math::

        C_p \in \operatorname{Hom}_G
        (V_{\mathrm{in}} \otimes V_{\mathrm{filter}}, V_{\mathrm{out}})

    once, then evaluates it using externally supplied reduced weights. With
    ``shared_weights=False`` (the default), ``reduced_weights`` has shape
    ``(..., weight_numel)`` matching the inputs' leading axes. Shared weights
    have shape ``(weight_numel,)``.

    This separation is useful for steerable kernels: angular/filter features
    determine the fixed equivariant basis while a radial network predicts the
    reduced coefficients. :meth:`sample_kernel_basis` expands filter samples
    into matrices with shape
    ``(..., weight_numel, output_dim, input_dim)``.

    Both :meth:`forward` and :meth:`sample_kernel_basis` use the grouped
    multiplicity execution plan owned by the underlying :class:`TensorProduct`.
    Sampling iterates over unique representation triples, not logical field
    triples, while preserving the historical flattened reduced-weight order.

    Raw feature/filter tensors emit
    :class:`MissingRepresentationMetadataWarning`. If both are
    :class:`RepresentationTensor` objects, metadata is validated and the
    output is typed. Reduced weights are invariant scalars and need no wrapper.
    """

    def __init__(self, rep_in: FieldType | Representation, rep_filter: FieldType | Representation, rep_out: FieldType | Representation, *, shared_weights: bool = False):
        super().__init__()
        if isinstance(rep_in, Representation):
            rep_in = as_field_type(rep_in)
        if isinstance(rep_filter, Representation):
            rep_filter = as_field_type(rep_filter)
        if isinstance(rep_out, Representation):
            rep_out = as_field_type(rep_out)
        self.rep_in = rep_in
        self.rep_filter = rep_filter
        self.rep_out = rep_out
        self.tensor_product = TensorProduct(
            rep_in,
            rep_filter,
            rep_out,
            internal_weights=False,
            shared_weights=shared_weights,
        )
        self.weight_numel = self.tensor_product.weight_numel
        self.in1_type = rep_in
        self.in2_type = rep_filter
        self.out_type = rep_out

    def forward(
        self,
        features: torch.Tensor | RepresentationTensor,
        filter_features: torch.Tensor | RepresentationTensor,
        reduced_weights: torch.Tensor,
    ) -> torch.Tensor | RepresentationTensor:
        return self.tensor_product(features, filter_features, reduced_weights)

    def sample_kernel_basis(
        self,
        filter_features: torch.Tensor | RepresentationTensor,
    ) -> torch.Tensor:
        """Expand filter features into the physical kernel basis.

        The result has shape ``(..., weight_numel, out_dim, in_dim)``. Its
        ``p``-th slice is the matrix-valued kernel obtained by setting the
        ``p``-th reduced weight to one and every other reduced weight to zero.
        This operation is intended for kernel inspection, oracle comparison,
        and basis precomputation; the regular forward path remains fused.
        """
        (filter_features,), _ = unpack_tensor_product_inputs(
            (("filter_features", filter_features, self.rep_filter),),
            f"{type(self).__name__}.sample_kernel_basis",
        )
        if filter_features.shape[-1] != self.rep_filter.size:
            raise ValueError(
                f"expected filter dimension {self.rep_filter.size}, got {filter_features.shape[-1]}"
            )

        leading_shape = filter_features.shape[:-1]
        kernels = filter_features.new_zeros(
            *leading_shape,
            self.weight_numel,
            self.rep_out.size,
            self.rep_in.size,
        )
        if self.tensor_product._grouped:
            kernels_flat = kernels.flatten(-2)
            for block in self.tensor_product.blocks:
                weight_indices, matrix_indices, values = block.kernel_basis_entries(
                    filter_features, self.rep_in.size
                )
                kernels_flat[
                    ...,
                    weight_indices[:, None],
                    matrix_indices,
                ] = values
            return kernels

        weight_offset = 0
        for instruction, path in zip(
            self.tensor_product.instructions, self.tensor_product.paths
        ):
            if not path.has_weight:
                raise RuntimeError(
                    "kernel-basis sampling requires weighted tensor-product paths"
                )
            in_start = self.rep_in.fields_start[instruction.i_in1]
            in_end = self.rep_in.fields_end[instruction.i_in1]
            filter_start = self.rep_filter.fields_start[instruction.i_in2]
            filter_end = self.rep_filter.fields_end[instruction.i_in2]
            out_start = self.rep_out.fields_start[instruction.i_out]
            out_end = self.rep_out.fields_end[instruction.i_out]

            local_filters = filter_features[..., filter_start:filter_end]
            identity = torch.eye(
                path.left.size,
                dtype=filter_features.dtype,
                device=filter_features.device,
            )
            left = identity.expand(*leading_shape, path.left.size, path.left.size)
            right = local_filters.unsqueeze(-2).expand(
                *leading_shape, path.left.size, path.right.size
            )
            for local_weight in range(path.weight_numel):
                coefficients = filter_features.new_zeros(path.weight_shape)
                coefficients.reshape(-1)[local_weight] = 1.0
                # The additional leading axis enumerates input basis vectors.
                # Transposing the result turns these responses into a matrix.
                values = path(left, right, coefficients).transpose(-1, -2)
                kernels[
                    ...,
                    weight_offset + local_weight,
                    out_start:out_end,
                    in_start:in_end,
                ] = values
            weight_offset += path.weight_numel

        if weight_offset != self.weight_numel:
            raise RuntimeError("tensor-product path weights were not fully sampled")
        return kernels


class SphericalKernelTensorProduct(KernelTensorProduct):
    r"""Kernel tensor product using restricted e3nn spherical harmonics.

    Pass a :class:`RestrictedSphericalHarmonics` module as ``rep_filter`` to
    couple directly from points with :meth:`forward_from_points` and to sample
    the physical matrix-valued kernel basis with :meth:`sample_kernel_basis`.
    A restricted filter ``FieldType`` is also accepted for compatibility with
    callers which evaluate the harmonics separately.

    Couplings always span the full finite-group Hom space. O(3) Wigner
    coefficients therefore describe an inherited subspace but never truncate
    subgroup-only paths.

    :meth:`forward_from_points` evaluates the owned e3nn spherical harmonics
    at 3D ``points`` and combines them with input features and reduced radial
    weights. :meth:`sample_kernel_basis` evaluates every resulting
    matrix-valued kernel path at supplied points. The harmonic filter created
    internally carries known metadata, so only an unwrapped feature input can
    trigger :class:`MissingRepresentationMetadataWarning`.

    Args:
        rep_in: Input feature representation.
        rep_filter: Restricted harmonic evaluator or its output type.
        rep_out: Desired kernel output representation.
        harmonics: Optional evaluator when ``rep_filter`` is supplied as a
            representation rather than as a module.
        shared_weights: Share one reduced-weight vector over leading axes.

    The output is a :class:`RepresentationTensor` when the feature input is
    typed; representation mismatches are rejected before harmonic contraction.
    """

    def __init__(
        self,
        rep_in: FieldType | Representation,
        rep_filter: FieldType | Representation | RestrictedSphericalHarmonics,
        rep_out: FieldType | Representation,
        *,
        harmonics: RestrictedSphericalHarmonics | None = None,
        shared_weights: bool = False,
    ):
        if isinstance(rep_filter, RestrictedSphericalHarmonics):
            if harmonics is not None:
                raise ValueError(
                    "pass the harmonic evaluator either positionally or with harmonics=, not both"
                )
            harmonics = rep_filter
            rep_filter = harmonics.out_type
        if harmonics is not None:
            if not isinstance(harmonics, RestrictedSphericalHarmonics):
                raise TypeError("harmonics must be RestrictedSphericalHarmonics")
            expected = (
                as_field_type(rep_filter)
                if isinstance(rep_filter, Representation)
                else rep_filter
            )
            if expected != harmonics.out_type:
                raise ValueError(
                    "the harmonic evaluator output does not match rep_filter"
                )
        else:
            fields = (
                as_field_type(rep_filter)
                if isinstance(rep_filter, Representation)
                else rep_filter
            )
            if not all(isinstance(rep, RestrictedO3Representation) for rep in fields):
                raise TypeError(
                    "restricted Wigner--Eckart filters must be restricted O(3) "
                    "representations or come from RestrictedSphericalHarmonics"
                )

        super().__init__(
            rep_in,
            rep_filter,
            rep_out,
            shared_weights=shared_weights,
        )
        self.harmonics = harmonics
        self.embedding = harmonics.embedding if harmonics is not None else None
        self.degrees = harmonics.degrees if harmonics is not None else None

    def evaluate_filter(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the restricted parent-group harmonic filter features."""
        if self.harmonics is None:
            raise RuntimeError(
                "point evaluation requires a RestrictedSphericalHarmonics evaluator"
            )
        return self.harmonics(points)

    def forward_from_points(
        self,
        features: torch.Tensor | RepresentationTensor,
        points: torch.Tensor,
        reduced_weights: torch.Tensor,
    ) -> torch.Tensor | RepresentationTensor:
        """Evaluate restricted harmonics and apply the tensor product."""
        filters = RepresentationTensor(self.evaluate_filter(points), self.rep_filter)
        return self(features, filters, reduced_weights)

    def sample_kernel_basis(self, points: torch.Tensor) -> torch.Tensor:
        """Sample all restricted finite-group kernel paths at 3D points."""
        filters = RepresentationTensor(self.evaluate_filter(points), self.rep_filter)
        return super().sample_kernel_basis(filters)

    def sample(self, points: torch.Tensor) -> torch.Tensor:
        """Alias matching escnn's analytical kernel-basis interface."""
        return self.sample_kernel_basis(points)
