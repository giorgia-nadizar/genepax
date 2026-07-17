import flax.linen as nn
import jax.numpy as jnp


class JaxCritic(nn.Module):

    hidden_sizes: tuple

    @nn.compact
    def __call__(self, obs, action):

        x = jnp.concatenate(
            [obs, action],
            axis=-1
        )

        for i, size in enumerate(self.hidden_sizes):

            x = nn.Dense(
                size,
                name=f"fc{i+1}"
            )(x)

            x = nn.relu(x)

        q = nn.Dense(
            1,
            name="q"
        )(x)

        return q.squeeze(-1)