"""Common tests for the Bayesian transformation."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import bayesian


@pytest.mark.parametrize("invalid_std", [-1.0, -0.01, -100.0])
def test_invalid_posterior_std(dummy_predictor: Predictor, invalid_std: float) -> None:
    """Tests that the bayesian function raises a ValueError for an invalid posterior_std.

    A negative standard deviation is physically and mathematically not meaningful,
    so the function should reject such values.

    Args:
        dummy_predictor: A mock or dummy predictor instance used for testing.
        invalid_std: A negative float value representing an invalid standard deviation.
    """
    # We expect a ValueError, which will likely be raised in the constructor of the Bayesian layer.
    with pytest.raises(ValueError, match="standard deviation cannot be negative"):
        bayesian(dummy_predictor, posterior_std=invalid_std)


@pytest.mark.parametrize("invalid_std", [-1.0, -0.01, -100.0])
def test_invalid_prior_std(dummy_predictor: Predictor, invalid_std: float) -> None:
    """Tests that the bayesian function raises a ValueError for an invalid prior_std.

    A negative standard deviation is physically and mathematically not meaningful,
    so the function should reject such values.

    Args:
        dummy_predictor: A mock or dummy predictor instance used for testing.
        invalid_std: A negative float value representing an invalid standard deviation.
    """
    # Wir erwarten einen ValueError, der wahrscheinlich im Konstruktor des bayesianischen Layers ausgelöst wird
    with pytest.raises(ValueError, match="standard deviation cannot be negative"):
        bayesian(dummy_predictor, prior_std=invalid_std)


def test_valid_parameters(dummy_predictor: Predictor) -> None:
    """Tests that valid parameters do not raise an error.

    Args:
        dummy_predictor: A mock or dummy predictor instance used for testing.
    """
    try:
        bayesian(
            dummy_predictor,
            use_base_weights=True,
            posterior_std=0.1,
            prior_mean=0.0,
            prior_std=1.0,
        )
    except ValueError:
        pytest.fail("bayesian() raised ValueError unexpectedly with valid parameters.")
