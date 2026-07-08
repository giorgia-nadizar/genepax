import jax
import jax.numpy as jnp


class CGPPolicy:
    def __init__(self, genome, cgp_structure, obs_mean, obs_std):
        self.genome = genome
        self.cgp = cgp_structure
        self.obs_mean = jnp.asarray(obs_mean)
        self.obs_std = jnp.asarray(obs_std) + 1e-8

        # compiled forward pass for speed
        self._apply = jax.jit(self._forward)

    def _forward(self, obs):
        obs = (obs - self.obs_mean) / self.obs_std
        return self.cgp.apply(self.genome, obs)

    def act(self, obs):
        a = self._apply(obs)
        return jnp.asarray(a)