"""Tests for the Flax module."""

from __future__ import annotations

import numpy as np
import pytest

from probly.transformation.ensemble.flax import generate_flax_ensemble
from pytraverse import singledispatch_traverser
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
from jax import numpy as jnp  # noqa: E402

reset_traverser = singledispatch_traverser[object](name="reset_traverser")


class TestEnsembleModule:
    """Test class for different aspects of Ensemble Flax."""

    def test_return_type_and_length_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks the generated ensemble module's return type and length.

        Args:
            flax_model_small_2d_2d (nnx.Sequential):
                Takes a Flax Fixture as a base model for generating the ensemble module.

        Asserts:
            - The returned value is a list.
            - The list length matches the requested number of ensemble members.
            - Every element in the list is an instance of `nnx.Module`.
        """
        k = 3
        ens = generate_flax_ensemble(flax_model_small_2d_2d, num_members=k, reset_params=True)

        assert isinstance(ens, list), f"Expected list, got {type(ens)}."
        assert len(ens) == k, f"Expected {k} members, got {len(ens)}."
        assert all(isinstance(m, nnx.Module) for m in ens), "All members must be nnx.Modules."

    def test_is_cloned(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks that the model and the ensemble module are not the same.

        Args:
            flax_model_small_2d_2d (nnx.Sequential):
                Takes a Flax Fixture as a base model for generating the ensemble module.

        Asserts:
            - The cloned ensemble module is not the same as the model.
        """
        k = 2
        model = flax_model_small_2d_2d
        clone = generate_flax_ensemble(model, num_members=k, reset_params=True)
        assert clone is not model, "Clone is the same as the model."

    def test_layer_counts_scale_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the number of layers is correct.

        Args:
            flax_model_small_2d_2d (nnx.Sequential):
                Takes a Flax Fixture as a base model for generating the ensemble module.

        Asserts:
            - The number of linear layers is correct.
            - The number of sequential layers is correct. Should be 0.
        """
        k = 3
        base = flax_model_small_2d_2d
        ens = generate_flax_ensemble(base, num_members=k)

        linear_layers = sum([count_layers(m, nnx.Linear) for m in ens])
        assert linear_layers == k * count_layers(base, nnx.Linear), (
            f"Expected {linear_layers} layers, got {linear_layers}, got {count_layers(base, nnx.Linear)}."
        )

        sequential_layers = sum([count_layers(m, nnx.Sequential) for m in ens])
        assert sequential_layers == k * count_layers(base, nnx.Sequential), (
            f"Expected {sequential_layers} layers, got {count_layers(base, nnx.Sequential)}."
        )

    def extract_kernels(self, model: nnx.Module) -> list[np.array]:
        """Helper method to extract kernels from a flax model.

        Args:
            model: nnx.Module
                Takes any nnx.Module.
        """
        kernel_values = [np.array(layer.kernel) for layer in model.layers if hasattr(layer, "kernel")]
        return kernel_values

    def test_ensemble_kernels_not_empty(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks that the array of kernel values is not empty.

        Args:
            flax_model_small_2d_2d (nnx.Sequential):
                Takes a Flax Fixture as a base model for generating the ensemble module.

        Asserts:
            - The number of kernel arrays is the same as num_members.
            - The Array is not empty.
        """
        k = 3
        ens = generate_flax_ensemble(flax_model_small_2d_2d, num_members=k, reset_params=True)

        all_kernel_arrays = [self.extract_kernels(m) for m in ens]
        combined_kernels = np.stack(all_kernel_arrays, axis=0)  # Shape: (3, total_kernel_values)
        assert combined_kernels.shape[0] == 3
        assert combined_kernels.size > 0

    def test_clone_and_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks that the Ensemble model's parameters have been reset correctly.

        Args:
              flax_model_small_2d_2d (nnx.Sequential):
                  Takes a Flax Fixture as a base model for generating the ensemble module.

          Asserts:
              - The in_features are not the same.
        """
        k = 2
        model = flax_model_small_2d_2d
        original_params = jax.tree_util.tree_leaves(model)

        ens = generate_flax_ensemble(model, num_members=k, reset_params=True)
        for i in range(k - 1):
            ens_params = jax.tree_util.tree_leaves(ens[i])
        assert not jnp.array_equal(ens_params[1], original_params[1])
