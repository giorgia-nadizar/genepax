import jax.numpy as jnp


class Normalizer:
    """
    Simple running normalization (fixed stats version).
    For BC we just use dataset statistics.
    """

    def __init__(self, obs_mean, obs_std):
        self.obs_mean = jnp.array(obs_mean)
        self.obs_std = jnp.array(obs_std) + 1e-8

    def normalize(self, obs):
        return (obs - self.obs_mean) / self.obs_std