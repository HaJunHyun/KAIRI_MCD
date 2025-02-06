import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, List
from type_alias import Params, KeyArray, PyTree


def get_batch_idxs(
    num_data: int,
    batch_size: int,
    key: Optional[KeyArray] = None,
    num_batches: Optional[int] = None,
) -> jax.Array:
    assert num_data % batch_size == 0
    if key is None:
        return jnp.arange(num_data).reshape((-1, batch_size))[:num_batches]
    else:
        batch_idxs = jax.random.permutation(key, num_data)
        batch_idxs = batch_idxs.reshape((-1, batch_size))[:num_batches]
        return batch_idxs


def get_batch(dataset: jax.Array, batch_idx: jax.Array) -> jax.Array:
    return jax.tree.map(lambda d: d[batch_idx], dataset)


def get_sliding_batch_start_idxs(num_data, batch_size, overlap):
    step = batch_size - overlap
    if step <= 0:
        raise ValueError("step = b - o must be positive for valid shifting.")

    last_start = num_data - batch_size
    if last_start < 0:
        # No valid windows
        return []

    # Determine how many windows fit
    num_windows = (last_start // step) + 1

    # Generate all start indices
    return jnp.arange(num_windows) * step


def l2_norm(params: Params) -> float:
    return sum([jnp.sum(p**2) for p in jax.tree.leaves(params)])


def normal_like_tree(tree: PyTree, rngs: nnx.Rngs, mean=0.0, std=1.0) -> PyTree:
    return jax.tree.map(
        lambda p: mean + std * jax.random.normal(rngs(), shape=p.shape), tree
    )


def tree_scale(tree: PyTree, c: float) -> PyTree:
    return jax.tree.map(lambda a: c * a, tree)


def tree_add(tree_a: PyTree, tree_b: PyTree) -> PyTree:
    return jax.tree.map(lambda a, b: a + b, tree_a, tree_b)


def tree_dot(tree_a: PyTree, tree_b: PyTree) -> PyTree:
    return jax.tree.map(lambda a, b: a * b, tree_a, tree_b)


def build_masks(model: nnx.Module, to_freeze: List[str] = None, filter=nnx.Param):
    to_freeze = [] if to_freeze is None else to_freeze
    state = nnx.state(model).filter(filter)
    path_and_values, treedef = jax.tree_util.tree_flatten_with_path(state)

    mask_values = []
    for path, _ in path_and_values:
        mask_values.append(0.0 if path[0].key in to_freeze else 1.0)
    return jax.tree_util.tree_unflatten(treedef, mask_values)
