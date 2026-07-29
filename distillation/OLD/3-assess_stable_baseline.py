import pickle

import jax
import jax.numpy as jnp

from brax import envs

from distillation.networks.jax_actor import JaxActor

# ============================================================
# Load trained SB3 model
# ============================================================

ENV = "inverted_double_pendulum"
MODEL_PATH = f"./sac_jax_{ENV}.pkl"

# Load parameters

with open(MODEL_PATH, "rb") as f:
    sac_data = pickle.load(f)

actor_data = sac_data["actor"]
architecture = actor_data["architecture"]
actor_params = actor_data["params"]

# Recreate network

actor = JaxActor(
    hidden_sizes=tuple(
        architecture["hidden_sizes"]
    ),
    action_dim=architecture["action_dim"],
)

# ============================================================
# Brax environment
# ============================================================

env = envs.create(
    env_name=ENV,
    backend="generalized",
)

print("obs size:", env.observation_size)
print("action size:", env.action_size)


# ============================================================
# Rollout with scan
# ============================================================

@jax.jit
def rollout(actor_params, seed):
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)

    def step_fn(carry, _):
        state, key = carry

        # symbolic policy
        action = actor.apply(
            actor_params,
            state.obs
        )

        # make sure action has shape (action_size,)
        action = jnp.asarray(action)

        next_state = env.step(
            state,
            action,
        )

        return (
            next_state,
            key,
        ), (next_state.reward, next_state.done)

    (_, _), (rewards, dones) = jax.lax.scan(
        step_fn,
        (state, key),
        None,
        length=1000,
    )
    alive = 1.0 - jnp.concatenate(
        [jnp.array([0]), jnp.cumsum(dones[:-1])]
    )
    alive = jnp.clip(alive, 0.0, 1.0)

    return jnp.sum(rewards * alive)


# ============================================================
# Evaluation
# ============================================================

vmapped_rollout = jax.vmap(rollout, in_axes=(None, 0))
rewards = vmapped_rollout(actor_params, jnp.arange(10))
print(rewards)
