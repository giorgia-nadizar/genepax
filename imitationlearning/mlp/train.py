import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training import train_state

from imitationlearning.dataset import load_d4rl_dataset
from policy import MLPPolicy
from flax import serialization


# -------------------------
# Train state
# -------------------------
class TrainState(train_state.TrainState):
    pass


# -------------------------
# BC loss
# -------------------------
def bc_loss(params, apply_fn, obs, actions):
    pred = apply_fn({"params": params}, obs)
    return jnp.mean((pred - actions) ** 2)


# -------------------------
# JIT training step
# -------------------------
@jax.jit
def train_step(state, obs, actions):
    def loss_fn(params):
        return bc_loss(params, state.apply_fn, obs, actions)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


# -------------------------
# Batch generator (FIXED)
# -------------------------
def get_batches(obs, actions, batch_size, key):
    n = obs.shape[0]
    perm = jax.random.permutation(key, n)

    num_batches = n // batch_size

    for i in range(num_batches):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        yield obs[idx], actions[idx]


# -------------------------
# Main
# -------------------------
def main():
    dataset_id = "mujoco/invertedpendulum/expert-v0"
    obs, actions = load_d4rl_dataset(dataset_id)

    obs = np.array(obs)
    actions = np.array(actions)

    # -------------------------
    # NORMALIZATION (obs only)
    # -------------------------
    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0)

    def norm_obs(x):
        return (x - obs_mean) / (obs_std + 1e-8)

    obs = norm_obs(obs)
    # actions = norm_act(actions)

    obs = jnp.array(obs)
    actions = jnp.array(actions)

    # -------------------------
    # model
    # -------------------------
    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]

    model = MLPPolicy(action_dim=act_dim)

    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, obs_dim)))["params"]

    # -------------------------
    # optimizer (IMPORTANT: slightly higher LR)
    # -------------------------
    tx = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    # -------------------------
    # training loop
    # -------------------------
    batch_size = 256
    epochs = 20

    print("Starting training...")

    for epoch in range(epochs):

        key, subkey = jax.random.split(key)

        losses = []

        for ob, ac in get_batches(obs, actions, batch_size, subkey):
            state, loss = train_step(state, ob, ac)
            losses.append(loss)

        print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.6f}")

    # -------------------------
    # save
    # -------------------------
    # Save model parameters
    with open("bc_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))

    # Save normalization statistics
    np.savez(
        "normalization.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

    print("Checkpoint saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
