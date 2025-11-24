"""Test for torch ensemble module."""

from __future__ import annotations

import pytest
from torch import nn

from probly.transformation import ensemble
from pytraverse import lazydispatch_traverser
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")
reset_traverser = lazydispatch_traverser[object](name="reset_traverser")


def test_return_type_and_length(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Ensure that `ensemble()` returns a `ModuleList` of the expected length."""
    k = 3
    ens = ensemble(torch_model_small_2d_2d, num_members=k)
    assert isinstance(ens, nn.ModuleList)
    assert len(ens) == k


def test_number_of_members(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Verify that the ensemble contains exactly the requested number of members."""
    num_members = 3
    model = ensemble(torch_model_small_2d_2d, num_members=num_members)
    assert len(model) == num_members


def test_returns_module_list(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that the ensemble returned by `ensemble()` is a `torch.nn.ModuleList`."""
    num_members = 3
    model = ensemble(torch_model_small_2d_2d, num_members=num_members)
    assert isinstance(model, nn.ModuleList)


def test_layer_counts_scale_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Check that linear-network layer counts scale linearly with the number of ensemble members."""
    k = 3
    base = torch_model_small_2d_2d
    ens = ensemble(base, num_members=k)

    assert count_layers(ens, nn.Linear) == k * count_layers(base, nn.Linear)
    assert count_layers(ens, nn.Sequential) == k * count_layers(base, nn.Sequential)
    assert count_layers(ens, nn.Dropout) == k * count_layers(base, nn.Dropout)


def test_layer_counts_scale_conv(torch_conv_linear_model: nn.Sequential) -> None:
    """Check that convolutional layer counts scale linearly with the number of ensemble members."""
    k = 4
    base = torch_conv_linear_model
    ens = ensemble(base, num_members=k)
    assert count_layers(ens, nn.Linear) == k * count_layers(base, nn.Linear)
    assert count_layers(ens, nn.Conv2d) == k * count_layers(base, nn.Conv2d)
    assert count_layers(ens, nn.Sequential) == k * count_layers(base, nn.Sequential)


def test_deep_copy(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that ensemble members are deep copies and do not share parameter memory."""
    k = 2
    ens = ensemble(torch_model_small_2d_2d, num_members=k)

    assert isinstance(ens, nn.ModuleList)
    assert len(ens) == k

    p0 = next(ens[0].parameters())
    p1 = next(ens[1].parameters())
    assert p0.data_ptr() != p1.data_ptr()
    params0 = list(ens[0].parameters())
    params1 = list(ens[1].parameters())
    assert len(params0) == len(params1)
