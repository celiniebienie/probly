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
    """Creates a new object with new random arguments for Linear Layers.

    Args:
         obj: nnx.Linear:
         The original linear layer, based on which a new linear layer is created.
    """
    rng_key = jax.random.key(42)
    rng_key, subkey = jax.random.split(rng_key)
    subkey = nnx.Rngs(params=subkey)
    return nnx.Linear(obj.in_features, obj.out_features, rngs=subkey)


@reset_traverser.register
def _(obj: nnx.Conv) -> nnx.Module:
    """Creates a new object with new random arguments for Linear Layers.

    Args:
         obj: nnx.Conv:
         The original convoluted layer, based on which a new linear layer is created.
    """
    rng_key = jax.random.key(42)
    rng_key, subkey = jax.random.split(rng_key)
    subkey = nnx.Rngs(params=subkey)
    return nnx.Conv(obj.in_features, obj.out_features, obj.kernel_size, obj.padding, rngs=subkey)


def _reset_copy(module: nnx.Module) -> nnx.Module:
    """Copies the layer and activated decorator to reset parameters.

    Args:
        module: nnx.Module:
        Takes any nnx.Module as argument.
    """
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nnx.Module) -> nnx.Module:
    """Copies the layer without resetting.

    Args:
        module: nnx.Module:
        Takes any nnx.Module as argument.
    """
    return traverse(module, nn_traverser, init={CLONE: True})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool = True,
) -> list[nnx.Module]:
    """Build a torch ensemble by copying the base model num_members times, resetting the parameters of each member.

    Args:
       obj: nnx.Module: Takes any nnx.Module as argument
       num_members: The number of members to generate.
       reset_params: If true, resets the parameters of each member.
    """
    if reset_params:
        return [_reset_copy(obj) for _ in range(num_members)]
    return [_copy(obj) for _ in range(num_members)]


register(nnx.Module, generate_flax_ensemble)
