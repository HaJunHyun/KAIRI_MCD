import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import nnx
from typing import NamedTuple, Optional, Tuple
from type_alias import Params, Batch, LossFunction, KeyArray, PyTree
from utils import normal_like_tree, l2_norm, get_sliding_batch_start_idxs, get_batch

import optax


class Particle(NamedTuple):
    params: Params 
    momentum: Params
    log_gamma_0: float
    log_trans: float
    log_gamma_k: float


def build(
    base_params: Params, 
    base_momentum: jax.Array, 
    graphdef: nnx.GraphDef,
    masks: nnx.State,
    train_ds: Batch,
    init_loss_fn: LossFunction,
    loss_fn: LossFunction, 
    damper: float, 
    leapfrog_updates: float, 
):  
    
    def energy_fn(params, batch): 
        loss, _ = loss_fn(nnx.merge(graphdef, params), batch) 
        return loss 
    
    def log_prob_momentum(momentum_particle: PyTree) -> float:
        leaves = jax.tree_leaves(momentum_particle)
        total = 0.0
        for leaf in leaves:
            size = leaf.size
            total += -0.5 * jnp.sum(leaf ** 2) - 0.5 * size * jnp.log(2 * jnp.pi)
        return total

    log_prob_momentum_v = jax.vmap(log_prob_momentum)
    
    def leapfrog_update(params, momentum, step_size, batch):
        energy, grad = jax.value_and_grad(energy_fn)(params, batch)
        momentum_half = jax.tree_map(
            lambda m, g, mask: m - 0.5 * step_size * mask * g,
            momentum, grad, masks
        )
        params_new = jax.tree_map(
            lambda p, m, mask: p + step_size * mask * m,
            params, momentum_half, masks
        )
        energy_new, grad_new = jax.value_and_grad(energy_fn)(params_new, batch)
        momentum_new = jax.tree_map(
            lambda m, g, mask: m - 0.5 * step_size * mask * g,
            momentum_half, grad_new, masks
        )
        return params_new, momentum_new, energy_new

    def multiple_leapfrog_updates(params, momentum, step_size, batch, L):
        def body_fn(carry, _):
            p, m = carry
            p, m, _ = leapfrog_update(p, m, step_size, batch)
            return (p, m), None
        (p_final, m_final), _ = jax.lax.scan(body_fn, (params, momentum), None, length=L)
        return p_final, m_final 
    
    def gaussian_log_prob(tree, tree_mean, sigma2):
        leaves_x, _ = jax.tree_flatten(tree)
        leaves_mu, _ = jax.tree_flatten(tree_mean)
        log_prob = 0.0
        for lx, lmu in zip(leaves_x, leaves_mu):
            size = lx.size
            log_prob += -0.5 * jnp.sum((lx - lmu)**2) / sigma2 - 0.5 * size * jnp.log(2 * jnp.pi * sigma2)
        return log_prob

    def init_momentum(key, params):
        return normal_like_tree(params, rngs=nnx.Rngs(key), mean=0.0, std=1.0)

    def init_particles(rngs: nnx.Rngs, num_particles: int) -> Particle:
        momentum = init_momentum(rngs(), base_params)
        stacked_momentum= jax.tree_map(lambda m: jnp.stack([m]* num_particles), momentum)
        return Particle(
            params=jax.tree_map(lambda p: jnp.stack([p] * num_particles), base_params),
            momentum= stacked_momentum,
            log_gamma_0=jnp.full([num_particles], -init_loss_fn(base_params)),
            log_trans=jnp.full([num_particles], 0.0),
            log_gamma_k=jnp.full([num_particles], 0.0),
        )

    @nnx.vmap(in_axes=(0, 0, None, None, None, None))
    def forward_particle(
        key: KeyArray,
        particle: Particle,
        batch: Batch,
        step_size: float, 
        damper: float, 
        leapfrog_updates: int,
    ) -> Particle: 

        std = jnp.sqrt(1 - damper**2)
        noise = normal_like_tree(particle.momentum, rngs=nnx.Rngs(key), mean=0.0, std=1.0)
        momentum_tilda = jax.tree_map(
            lambda m, n, mask: mask * (m * damper + std * n) + (1 - mask) * m,
            particle.momentum, noise, masks
        )
        
        logF = gaussian_log_prob(
            particle.momentum,
            jax.tree_map(lambda m: damper * m, momentum_tilda),
            std**2
        )
        logB = gaussian_log_prob(
            momentum_tilda,
            jax.tree_map(lambda m: damper * m, particle.momentum),
            std**2
        ) 

        params_new, momentum_new= multiple_leapfrog_updates(
            particle.params, momentum_tilda, step_size, batch, leapfrog_updates
        ) 
        energy_new, _ = jax.value_and_grad(energy_fn)(params_new, batch)

        return Particle(
            params=params_new, 
            momentum= momentum_new, 
            log_gamma_0=particle.log_gamma_0,
            log_trans=particle.log_trans + logF- logB,
            log_gamma_k=-energy_new,
        )

    def forward_particles(
        rngs: nnx.Rngs,
        particles: Particle,
        batch: Batch,
        step_size: float, 
        damper: float, 
        leapfrog_updates: int, 
    ) -> Particle:
        num_particles = len(particles.log_gamma_k)
        keys = jax.random.split(rngs(), num_particles)
        return forward_particle(keys, particles, batch, step_size, damper, leapfrog_updates)

    def resample_if_needed(
        rngs: nnx.Rngs,
        particles: Particle,
        thres: float,
    ) -> Particle:
        log_prob_v= log_prob_momentum_v(particles.momentum)
        log_w = log_prob_v+ particles.log_gamma_k + particles.log_trans - particles.log_gamma_0
        num_particles = len(log_w)
        ess = jnp.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w))

        idxs = jax.random.categorical(rngs(), log_w, shape=(num_particles,))

        resampled_params = jax.tree_map(
            lambda p: jnp.take(p, idxs, axis=0), particles.params
        ) 
        resampled_momentum = jax.tree_map(
            lambda m: jnp.take(m, idxs, axis=0), particles.momentum
        )
        resampled_log_gamma_0 = jnp.take(log_prob_v + particles.log_gamma_k, idxs)
        resampled_particles = Particle(
            params=resampled_params, 
            momentum=resampled_momentum,
            log_gamma_0= resampled_log_gamma_0,
            log_trans=jnp.zeros_like(log_w),
            log_gamma_k=jnp.zeros_like(log_w),
        )
        log_Z_ratio_est = logsumexp(log_w) - jnp.log(num_particles)

        return jax.lax.cond(
            ess < thres * num_particles,
            lambda _: (resampled_particles, log_Z_ratio_est, 1),
            lambda _: (particles, 0.0, 0),
            operand= 0,
        )

    num_train = len(train_ds[0])

    def run_uha(
        rngs: nnx.Rngs,
        num_particles: int,
        batch_size: int,
        overlap: int,
        num_cycles: int,
        damper: float, 
        leapfrog_updates: int, 
        step_size: float, 
        resample_thres: Optional[float] = 0.5,
    ) -> Tuple[Particle, float, int]:

        start_idxs = jnp.concatenate(
            [
                get_sliding_batch_start_idxs(num_train, batch_size, overlap)
                for _ in range(num_cycles)
            ],
            0,
        )
        num_batches = len(start_idxs)
        particles = init_particles(rngs, num_particles) 
        log_Z_est= 0.0
        resample_cnt= 0

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def step(carry, start_idx):
            k, rngs, particles, log_Z_est, resample_cnt = carry

            batch = jax.tree_map(
                lambda d: jax.lax.dynamic_slice(
                    d, (start_idx,) + (0,) * (d.ndim - 1), (batch_size,) + d.shape[1:]
                ),
                train_ds,
            )
            particles = forward_particles(rngs, particles, batch, step_size, damper, leapfrog_updates)
            particles, log_Z_ratio_est, resampled = resample_if_needed(
                rngs, particles, resample_thres
            )

            return (
                k + 1,
                rngs,
                particles,
                log_Z_est + log_Z_ratio_est,
                resample_cnt + resampled,
            )

        _, rngs, particles, log_Z_est, resample_cnt = step(
            (0, rngs, particles, 0.0, 0), start_idxs
        )

        particles = forward_particles(rngs, particles, train_ds, step_size, damper, leapfrog_updates)
        log_prob_v= log_prob_momentum_v(particles.momentum)
        log_w = log_prob_v+ particles.log_gamma_k + particles.log_trans - particles.log_gamma_0
        log_Z_est += logsumexp(log_w) - jnp.log(num_particles)

        return particles, log_Z_est, resample_cnt

    return run_uha
