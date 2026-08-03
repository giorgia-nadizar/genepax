import json
from pathlib import Path

import jax
import flax.linen as nn
import jax.numpy as jnp
from flax import serialization


class StudentPolicy(nn.Module):
    action_size: int
    hidden_sizes: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_size)(x)

        # Match TD3 actor output range
        x = nn.tanh(x)

        return x
