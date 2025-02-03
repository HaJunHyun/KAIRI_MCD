import jax
import jax.numpy as jnp 
from jax import random
import equinox as eqx
import optax
from jax.random import PRNGKey, split
from typing import Optional, Callable, List
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from tqdm import trange
import math 
import jax.scipy.special  # For gammaln

class MultivariateStudentT(eqx.Module):
    loc: jnp.ndarray
    scale_diag: jnp.ndarray
    df: float = 3.0 

    def sample(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        samples = dist.StudentT(df=self.df, loc=0.0, scale=1.0).sample(key, sample_shape=self.loc.shape)
        return self.loc + self.scale_diag * samples

    def log_prob(self, x: jnp.ndarray) -> float:
        df = self.df
        d = self.loc.size 
        const_term = d * (jax.scipy.special.gammaln((df + 1) / 2) -
                            jax.scipy.special.gammaln(df / 2) -
                            0.5 * jnp.log(df * jnp.pi))
        scale_term = - jnp.sum(jnp.log(self.scale_diag))
        tail_term = - ((df + 1) / 2) * jnp.sum(jnp.log(1 + (((x - self.loc) / self.scale_diag) ** 2) / df))
        
        return const_term + scale_term + tail_term

class MultivariateNormalDiag(eqx.Module):
    loc: jnp.ndarray
    scale_diag: jnp.ndarray

    def sample(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return self.loc + self.scale_diag * random.normal(key, shape=self.loc.shape)

    def log_prob(self, x: jnp.ndarray) -> float:
        d = x.size
        log_det = jnp.sum(jnp.log(self.scale_diag**2))
        log_norm= -0.5*d*jnp.log(2.0*jnp.pi) -0.5*log_det
        return log_norm - 0.5*jnp.sum(((x - self.loc)/self.scale_diag)**2)


class GaussianMixture(eqx.Module):
    means: jnp.ndarray
    std: float= 1.0

    def __init__(self, dim, n_components=8, key=jax.random.PRNGKey(0)):
        k1, _= random.split(key)
        base= random.normal(k1, (n_components, dim))
        self.means= 3.0 + base
        self.std= 1.0

    def log_prob(self, x: jnp.ndarray) -> float:
        d= x.size
        def comp_lp(m):
            log_norm= -0.5*d*jnp.log(2.0*jnp.pi*self.std**2)
            return log_norm - 0.5*jnp.sum(((x - m)/self.std)**2)
        log_probs= jax.vmap(comp_lp)(self.means)
        return jax.scipy.special.logsumexp(log_probs) - jnp.log(self.means.shape[0]) 
    
def sigmoid(x):
    return 1.0/(1.0 + jnp.exp(-x)) 

class AnnealingSchedule(eqx.Module):
    b: jnp.ndarray  

    def __init__(self, n_steps: int, key: jax.random.PRNGKey):
        k1, _= random.split(key)
        self.b= 0.01* random.normal(k1, shape=(n_steps,)) # Need to be fixed..

    def compute_betas(self) -> jnp.ndarray:
        s= sigmoid(self.b)
        part= jnp.cumsum(s)
        total= part[-1]
        betas_interior= part/(total+1e-8)
        return jnp.concatenate([jnp.array([0.0]), betas_interior], axis=0) 
    
class StepSizeMLP(eqx.Module):
    embed: eqx.nn.Embedding
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, n_steps: int, key: jax.random.PRNGKey):
        k1, k2, k3= random.split(key, 3)
        self.embed= eqx.nn.Embedding(num_embeddings=n_steps+1, embedding_size=32, key=k1)
        self.l1= eqx.nn.Linear(32, 32, key=k2)
        self.l2= eqx.nn.Linear(32, 1, key=k3)

    def __call__(self, k: int) -> float:
        x= self.embed(jnp.array(k, dtype=jnp.int32))
        x= jax.nn.swish(x)
        x= self.l1(x)
        x= jax.nn.swish(x)
        out= self.l2(x)[0]
        return 0.25* sigmoid(out) 
    
class ResidualBlock(eqx.Module):
    layernorm: eqx.nn.LayerNorm
    linear_in: eqx.nn.Linear
    linear_time: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    d_h: int = 512
    d_t: int = 16

    def __init__(self, d_h: int, d_t: int, key: jax.random.PRNGKey):
        k1, k2, k3, k4= random.split(key,4)
        self.layernorm= eqx.nn.LayerNorm(d_h)
        self.linear_in= eqx.nn.Linear(d_h, 2*d_h, key=k1)
        self.linear_time= eqx.nn.Linear(d_t, 2*d_h, key=k2)
        self.linear_out= eqx.nn.Linear(2*d_h, d_h, key=k3)

    def __call__(self, h: jnp.ndarray, t_emb: jnp.ndarray):
        r= self.layernorm(h)
        r= jax.nn.swish(r)
        rin= self.linear_in(r)
        rtime= self.linear_time(t_emb)
        r_all= rin+ rtime
        r_all= jax.nn.swish(r_all)
        r_out= self.linear_out(r_all)
        return h+ r_out


class ScoreNetwork(eqx.Module):
    x_proj: eqx.nn.Linear
    t_proj: eqx.nn.Linear
    blocks: List[ResidualBlock]
    final: eqx.nn.Linear
    x_dim: int
    d_h: int
    d_t: int
    k: int

    def __init__(
        self,
        x_dim: int,
        d_h: int=512,
        d_t: int=16,
        k: int=3,
        key: jax.random.PRNGKey=jax.random.PRNGKey(0),
    ):
        super().__init__()
        self.x_dim = x_dim
        self.d_h = d_h
        self.d_t = d_t
        self.k = k

        k1, k2, kblocks, kfinal= random.split(key,4)
        self.x_proj= eqx.nn.Linear(x_dim, self.d_h, key=k1)
        self.t_proj= eqx.nn.Linear(1, self.d_t, key=k2)
        keys_blocks= random.split(kblocks, self.k)
        self.blocks= []
        for bkey in keys_blocks:
            self.blocks.append(ResidualBlock(d_h=self.d_h, d_t=self.d_t, key=bkey))
        self.final= eqx.nn.Linear(self.d_h, x_dim, key=kfinal) 
        # initializing final layers to zero. 
        self.final= eqx.tree_at(
            lambda m: m.weight, 
            self.final, 
            jnp.zeros((x_dim, self.d_h)) 
        ) 
        self.final= eqx.tree_at(
            lambda m: m.bias, 
            self.final, 
            jnp.zeros((x_dim,)) 
        )

    def __call__(self, t: int, x: jnp.ndarray) -> jnp.ndarray:
        hx= self.x_proj(x)
        hx= jax.nn.swish(hx)
        tval= jnp.array(t, dtype=jnp.float32).reshape([1])
        ht= self.t_proj(tval)
        ht= jax.nn.swish(ht)
        for block in self.blocks:
            hx= block(hx, ht)
        hx= jax.nn.swish(hx)
        out= self.final(hx)
        return out