import flax.linen as nn
import jax.numpy as jnp


class JaxActor(nn.Module):

    hidden_sizes: tuple
    action_dim: int

    @nn.compact
    def __call__(self, obs):

        x = obs

        for i, size in enumerate(self.hidden_sizes):

            x = nn.Dense(
                size,
                name=f"fc{i+1}"
            )(x)

            x = nn.relu(x)

        mu = nn.Dense(
            self.action_dim,
            name="mu"
        )(x)

        # deterministic SAC policy
        action = jnp.tanh(mu)

        return action