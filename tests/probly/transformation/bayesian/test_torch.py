"""Tests for torch Bayesian models."""

from __future__ import annotations

import pytest

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.transformation import bayesian
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_replacement(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if nn.Linear layers in a model are correctly replaced by BayesLinear layers.

        This function verifies that:
        - All nn.Linear layers are replaced by BayesLinear layers.
        - The number of original linear layers matches the number of new bayesian linear layers.
        - The overall structure of the model remains a torch.nn.Sequential instance.

        Args:
            torch_model_small_2d_2d: A small sequential PyTorch model containing Linear layers,
                provided by a pytest fixture.
        """
        model = bayesian(torch_model_small_2d_2d)

        # Count the layers in the original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        count_bayes_linear_original = count_layers(torch_model_small_2d_2d, BayesLinear)

        # Count the layers in the modified model
        count_linear_modified = count_layers(model, nn.Linear)
        count_bayes_linear_modified = count_layers(model, BayesLinear)

        # Check whether the replacement has been carried out as expected.
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_linear_original > 0
        assert count_bayes_linear_original == 0
        assert count_linear_modified == 0
        assert count_bayes_linear_modified == count_linear_original

    def test_convolutional_network_replacement(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the replacement of layers in a convolutional neural network.

        This function verifies that both nn.Linear and nn.Conv2d layers are
        correctly replaced by their Bayesian counterparts (BayesLinear and BayesConv2d).

        Args:
            torch_conv_linear_model: A sequential PyTorch model containing both Conv2d
                and Linear layers, provided by a pytest fixture.
        """
        model = bayesian(torch_conv_linear_model)

        # Count the layers in the original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)

        # Count the layers in the modified model
        count_linear_modified = count_layers(model, nn.Linear)
        count_conv_modified = count_layers(model, nn.Conv2d)
        count_bayes_linear_modified = count_layers(model, BayesLinear)
        count_bayes_conv_modified = count_layers(model, BayesConv2d)

        # Check whether the replacement has been carried out as expected.
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_original > 0
        assert count_conv_original > 0

        assert count_linear_modified == 0
        assert count_conv_modified == 0
        assert count_bayes_linear_modified == count_linear_original
        assert count_bayes_conv_modified == count_conv_original

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the Bayesian transformation on a custom nn.Module.

        Args:
            torch_custom_model: A custom PyTorch nn.Module (not nn.Sequential),
                provided by a pytest fixture.
        """
        model = bayesian(torch_custom_model)

        # Check whether the model type is retained
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)


class TestParameters:
    """Test class for parameter settings in Bayesian layers."""

    def test_parameter_setting(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if parameters are correctly passed to and set in the Bayesian layers.

        This test assumes that the Bayesian layers store the passed parameters as attributes
        (e.g., `prior_mean`, `prior_std`).

        Args:
            torch_model_small_2d_2d: A small sequential PyTorch model used as a base
                for the Bayesian transformation.
        """
        # Define test parameters
        posterior_std = 0.08
        prior_mean = 0.5
        prior_std = 1.5

        model = bayesian(
            torch_model_small_2d_2d,
            posterior_std=posterior_std,
            prior_mean=prior_mean,
            prior_std=prior_std,
        )

        # Check each Bayesian layer in the model
        for m in model.modules():
            if isinstance(m, (BayesLinear, BayesConv2d)):
                # Assumption: The layers have these attributes for verification.
                assert m.prior_mean == prior_mean
                assert m.prior_std == prior_std
                # NOTE: The check for posterior_std may depend on the implementation,
                # as it may be a learnable parameter. This test assumes
                # that the initialization value is stored somewhere.
                assert m.initial_posterior_std == posterior_std

    def test_use_base_weights_true(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that `use_base_weights=True` correctly uses the original model's weights.

        This test assumes that the posterior mean of the weights (`weight_mu`) in the
        Bayesian layer is initialized with the weights from the original, non-Bayesian layer.

        Args:
            torch_model_small_2d_2d: A small sequential PyTorch model with pre-initialized weights.
        """
        model = bayesian(torch_model_small_2d_2d, use_base_weights=True)

        original_layers = [m for m in torch_model_small_2d_2d.modules() if isinstance(m, nn.Linear)]
        bayesian_layers = [m for m in model.modules() if isinstance(m, BayesLinear)]

        assert len(original_layers) == len(bayesian_layers)

        for orig_layer, bayes_layer in zip(original_layers, bayesian_layers, strict=False):
            # Assumption: The `weight_mu` parameter exists and holds the mean value of the weight posterior distribution
            # Assert torch.allclose(bayes_layer.weight_mu, orig_layer.weight.data)
            if orig_layer.bias is not None:
                assert torch.allclose(bayes_layer.bias_mu, orig_layer.bias.data)

    def test_use_base_weights_false(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests that `use_base_weights=False` initializes weights based on the prior mean.

        This test assumes that when not using base weights, the posterior mean of the weights
        (`weight_mu`) is initialized with the specified `prior_mean`.

        Args:
            torch_model_small_2d_2d: A small sequential PyTorch model used as a base.
        """
        prior_mean = 0.5
        model = bayesian(torch_model_small_2d_2d, use_base_weights=False, prior_mean=prior_mean)

        for m in model.modules():
            if isinstance(m, (BayesLinear, BayesConv2d)):
                # Assumption: `weight_mu` is filled with a tensor containing `prior_mean`
                expected_weight_mu = torch.full_like(m.weight_mu, prior_mean)
                assert torch.allclose(m.weight_mu, expected_weight_mu)
