import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import nnx
from typing import NamedTuple, Optional, Tuple
from type_alias import Params, Batch, LossFunction, KeyArray
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
    
    def leapfrog_update(params, momentum, step_size, batch): 
        loss, grad_params= jax.value_and_grad(energy_fn)(params, batch) 
        momentum_half= jax.tree_map(lambda m,g: m- 0.5*step_size*g, momentum, grad_params) 
        params_new= jax.tree_map(lambda p, m: p+ step_size*m, params, momentum_half) 
        loss_new, grad_params_new= jax.value_and_grad(energy_fn)(params_new, batch) 
        momentum_new= jax.tree_map(lambda m,g: m- 0.5*step_size*g, momentum_half, grad_params_new)
        return params_new, momentum_new, loss_new 
    
    def multiple_leapfrog_updates(params, momentum, step_size, batch, L):
        def body_fn(carry, _):
            params, momentum = carry
            params, momentum, _ = leapfrog_update(params, momentum, step_size, batch)
            return (params, momentum), None
        (params_final, momentum_final), _ = jax.lax.scan(body_fn, (params, momentum), None, length=L)
        return params_final, momentum_final 
    
    def gaussian_log_prob(tree, tree_mean, sigma2):
        leaves_x, _ = jax.tree_flatten(tree)
        leaves_mu, _ = jax.tree_flatten(tree_mean)
        log_prob = 0.0
        for lx, lmu in zip(leaves_x, leaves_mu):
            size = lx.size
            log_prob += -0.5 * jnp.sum((lx - lmu)**2) / sigma2 - 0.5 * size * jnp.log(2 * jnp.pi * sigma2)
        return log_prob

    def init_momentum(key, params): 
        return jax.tree_map(lambda p: jax.random.normal(key, shape=p.shape), params)

    def init_particles(rngs: nnx.Rngs, num_particles: int) -> Particle:
        momentum = init_momentum(rngs(), base_params)
        stacked_momentum= jax.tree_map(lambda m: jnp.stack([m]* num_particles), momentum)
        return Particle(
            params=jax.tree.map(lambda p: jnp.stack([p] * num_particles), base_params),
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
        def refresh(m): 
            noise= jax.random.normal(key, shape= m.shape) 
            return damper*m + jnp.sqrt(1-damper**2)*noise 
        
        momentum_refreshed= jax.tree_map(refresh, particle.momentum) 

        sigma2 = 1 - damper**2
        logF = gaussian_log_prob(
            particle.momentum,
            jax.tree_map(lambda m: damper * m, momentum_refreshed),
            sigma2
        )
        logB = gaussian_log_prob(
            momentum_refreshed,
            jax.tree_map(lambda m: damper * m, particle.momentum),
            sigma2
        ) 

        params_new, momentum_new= multiple_leapfrog_updates(
            particle.params, momentum_refreshed, step_size, batch, leapfrog_updates
        ) 
        energy_new= energy_fn(params_new, batch)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        std = jnp.sqrt(2 * step_size)
        noise = normal_like_tree(particle.params, rngs=nnx.Rngs(key))
        _, grads = grad_fn(nnx.merge(graphdef, particle.params), batch)

        params_new = jax.tree.map(
            lambda p, g, m, e: p - step_size * m * g + std * m * e,
            particle.params,
            grads,
            masks,
            noise,
        )
        log_F = -0.5 * l2_norm(noise)

        (energy, _), bgrads = grad_fn(nnx.merge(graphdef, params_new), batch)
        bnoise = jax.tree_map(
            lambda p_new, g, m, p, e: m * (p - (p_new - step_size * g)) / std
            + (1 - m) * e,
            params_new,
            bgrads,
            masks,
            particle.params,
            noise,
        )
        log_B = -0.5 * l2_norm(bnoise)

        return Particle(
            params=params_new,
            log_gamma_0=particle.log_gamma_0,
            log_trans=particle.log_trans + log_B - log_F,
            log_gamma_k=-energy,
        )

    def forward_particles(
        rngs: nnx.Rngs,
        particles: Particle,
        batch: Batch,
        step_size: float,
    ) -> Particle:
        num_particles = len(particles.log_gamma_k)
        keys = jax.random.split(rngs(), num_particles)
        return forward_particle(keys, particles, batch, step_size)

    def resample_if_needed(
        rngs: nnx.Rngs,
        particles: Particle,
        thres: float,
    ) -> Particle:

        log_w = particles.log_gamma_k + particles.log_trans - particles.log_gamma_0
        num_particles = len(log_w)
        ess = jnp.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w))

        idxs = jax.random.categorical(rngs(), log_w, shape=(num_particles,))

        resampled_params = jax.tree_map(
            lambda p: jnp.take(p, idxs, axis=0), particles.params
        )

        resampled_particles = Particle(
            params=resampled_params,
            log_gamma_0=jnp.take(particles.log_gamma_k, idxs),
            log_trans=jnp.zeros_like(log_w),
            log_gamma_k=jnp.zeros_like(log_w),
        )
        log_Z_ratio_est = logsumexp(log_w) - jnp.log(num_particles)

        return jax.lax.cond(
            ess < thres * num_particles,
            lambda _: (resampled_particles, log_Z_ratio_est, 1),
            lambda _: (particles, 0.0, 0),
            operand=None,
        )

    num_train = len(train_ds[0])

    def run_ais(
        rngs: nnx.Rngs,
        num_particles: int,
        batch_size: int,
        overlap: int,
        num_cycles: int,
        init_step_size: float,
        final_step_size: Optional[float] = 1.0e-6,
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
        step_size_fn = optax.cosine_decay_schedule(
            init_step_size, num_batches, alpha=final_step_size / init_step_size
        )
        particles = init_particles(num_particles)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def step(carry, start_idx):
            k, rngs, particles, log_Z_est, resample_cnt = carry

            batch = jax.tree.map(
                lambda d: jax.lax.dynamic_slice(
                    d, (start_idx,) + (0,) * (d.ndim - 1), (batch_size,) + d.shape[1:]
                ),
                train_ds,
            )
            step_size = step_size_fn(k)
            particles = forward_particles(rngs, particles, batch, step_size)
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

        particles = forward_particles(rngs, particles, train_ds, final_step_size)
        log_w = particles.log_gamma_k + particles.log_trans - particles.log_gamma_0
        log_Z_est += logsumexp(log_w) - jnp.log(num_particles)

        return particles, log_Z_est, resample_cnt

    return run_ais
