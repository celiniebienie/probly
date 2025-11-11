"""Flax dropout implementation."""

from __future__ import annotations

from flax import nnx
import jax

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Linear) -> nnx.Module:
    rng = nnx.Rngs(params=jax.random.key(42))
    return nnx.Linear(obj.in_features, obj.out_features, rngs=rng)


@reset_traverser.register
def _(obj: nnx.Conv) -> nnx.Module:
    rng = nnx.Rngs(params=jax.random.key(0))
    return nnx.Conv(obj.in_features, obj.out_features, obj.kernel_size, obj.padding, rngs=rng)


def _reset_copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_traverser, init={CLONE: True})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool = True,
) -> list[nnx.Module]:
    """Build a torch ensemble by copying the base model num_members times, resetting the parameters of each member."""
    if reset_params:
        return [_reset_copy(obj) for _ in range(num_members)]
    return [_copy(obj) for _ in range(num_members)]


register(nnx.Module, generate_flax_ensemble)
