import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple


def generate_data(
    rngs: nnx.Rngs, n: int, d: int, lamb: float, rho0: float
) -> jax.Array:
    theta = jax.random.normal(rngs(), shape=(1, d)) / jnp.sqrt(rho0)
    x = theta + jax.random.normal(rngs(), shape=(n, d)) / jnp.sqrt(lamb)
    return x


# unnormalized negative log likelihood
def neg_log_likel(theta: jax.Array, x: jax.Array, lamb: float) -> float:
    return 0.5 * lamb * jnp.sum((x - theta) ** 2)


# unnormalized negative log prior
def neg_log_prior(theta: jax.Array, rho0: float) -> float:
    return 0.5 * rho0 * jnp.sum(theta**2)


# log marginal likelihood computed from unnormalized log joint density
def log_Z(x: jax.Array, lamb: float, rho0: float) -> float:
    mu, rho = posterior_params(x, lamb, rho0)
    d = x.shape[-1]
    return (
        0.5 * d * jnp.log(2 * jnp.pi)
        - 0.5 * d * jnp.log(rho)
        - 0.5 * lamb * jnp.sum(x**2)
        + 0.5 * rho * jnp.sum(mu**2)
    )


# gaussian log probability with mean mu and precision rho
def log_prob(theta: jax.Array, mu: jax.Array, rho: float) -> float:
    d = theta.shape[-1]
    return (
        -0.5 * d * jnp.log(2 * jnp.pi)
        + 0.5 * d * jnp.log(rho)
        - 0.5 * rho * jnp.sum((theta - mu) ** 2)
    )


# compute posterior parameters given data x, prior precision lamb, and prior precision rho0
def posterior_params(x: jax.Array, lamb: float, rho0: float) -> Tuple[jax.Array, float]:
    n = len(x)
    post_prec = rho0 + lamb * n
    post_mean = lamb * n * jnp.mean(x, 0) / post_prec
    return post_mean, post_prec
