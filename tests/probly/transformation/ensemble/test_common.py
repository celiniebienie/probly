"""Tests for the ensemble module."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.ensemble.common import ensemble_generator, register


class InvalidDummyPredictor(Predictor):
    def __call__(self, x: int) -> int:
        return x


class ValidDummyPredictor(Predictor):
    pass


def test_for_right_calls() -> None:
    """ensemble_generator should call the registered generator for this predictor type."""
    base = ValidDummyPredictor()

    def dummy_generator(
        base: ValidDummyPredictor,
        num_members: int,
        reset_params: bool = True,
    ) -> tuple:
        return (base, num_members, reset_params)

    register(cls=ValidDummyPredictor, generator=dummy_generator)
    result = ensemble_generator(base, num_members=3, reset_params=False)

    assert result[0] is base
    assert result[1] == 3
    assert result[2] is False


def test_for_invalid_type() -> None:
    """Calling ensemble_generator on an unregistered predictor type should raise."""
    base = InvalidDummyPredictor()
    with pytest.raises(NotImplementedError):
        ensemble_generator(base)
