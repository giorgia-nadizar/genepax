import flax.linen as nn


class MLPPolicy(nn.Module):
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x