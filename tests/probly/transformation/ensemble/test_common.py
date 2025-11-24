
"""Tests for the ensemble module."""
from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import *

class Invalid_Dummy_Predictor(Predictor):
    def __call__(self, x:int) -> int:
        return x

class Valid_Dummy_Predictor(Predictor):
    pass


def test_base() -> None:
    def dummy_generator(base, num_members):
        return (base, num_members)
    register(cls = Valid_Dummy_Predictor, generator = dummy_generator)

def test_for_invalid_type() -> None:
    base = Invalid_Dummy_Predictor()
    with pytest.raises(NotImplementedError):
        ensemble_generator(base)