"""Finite-group Wigner--Eckart tensor products."""

from __future__ import annotations

import torch
from torch import nn

from .field_type import FieldType
from .geometric_tensor import GeometricTensor
from .tensor_product import TensorProduct
from ..representations import Representation
from ..gspaces import no_base_space


class WignerEckartTensorProduct(nn.Module):
    """Separate fixed finite-group coupling tensors from reduced weights."""

    def __init__(self, rep_in: FieldType | Representation, rep_filter: FieldType | Representation, rep_out: FieldType | Representation, *, shared_weights: bool = False):
        super().__init__()
        self._raw_tensor_api = all(isinstance(rep, Representation) for rep in (rep_in, rep_filter, rep_out))
        if isinstance(rep_in, Representation):
            rep_in = FieldType(no_base_space(rep_in.group), [rep_in])
        if isinstance(rep_filter, Representation):
            rep_filter = FieldType(no_base_space(rep_filter.group), [rep_filter])
        if isinstance(rep_out, Representation):
            rep_out = FieldType(no_base_space(rep_out.group), [rep_out])
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
        features: GeometricTensor,
        filter_features: GeometricTensor,
        reduced_weights: torch.Tensor,
    ) -> GeometricTensor:
        raw = isinstance(features, torch.Tensor) and isinstance(filter_features, torch.Tensor)
        if raw:
            if not self._raw_tensor_api:
                raise TypeError("raw tensors require construction from Representations")
            features = GeometricTensor(features, self.rep_in)
            filter_features = GeometricTensor(filter_features, self.rep_filter)
        output = self.tensor_product(features, filter_features, reduced_weights)
        return output.tensor if raw else output


class RestrictedWignerEckartTensorProduct(WignerEckartTensorProduct):
    """Semantic alias for Wigner--Eckart products using restricted O(3) filters.

    Couplings are still computed from the full finite-group Hom space; this
    class never truncates paths to inherited O(3) Wigner coefficients.
    """
