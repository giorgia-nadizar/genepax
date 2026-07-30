"""Behavioral cloning of a deterministic SAC teacher."""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from brax import envs
from flax import serialization

from distillation.networks.neural_student import StudentPolicy


class StudentTrainer:
    def __init__(self, model, obs_size, learning_rate=3e-4, seed=0):
        self.model = model
        self.seed = seed
        self.params = model.init(
            jax.random.key(seed), jnp.zeros((1, obs_size), dtype=jnp.float32)
        )
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def loss_fn(self, params, X, y):
        return jnp.mean((self.model.apply(params, X) - y) ** 2)

    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, params, opt_state, X, y):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, X, y)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @functools.partial(jax.jit, static_argnums=0)
    def predict(self, obs):
        return self.model.apply(self.params, obs)

    def train(
            self, X, y, epochs=100, batch_size=256,
            validation_fraction=0.1, seed=None,
    ):
        """Fits BC and restores the parameters with the best validation MSE."""
        X, y = jnp.asarray(X, jnp.float32), jnp.asarray(y, jnp.float32)
        if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be rank-2 arrays with matching rows")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        n_samples = int(X.shape[0])
        if n_samples < 2:
            raise ValueError("At least two demonstrations are required")
        key = jax.random.key(self.seed if seed is None else seed)
        split = jax.random.permutation(key, n_samples)
        n_validation = max(1, int(n_samples * validation_fraction))
        validation_indices, training_indices = split[:n_validation], split[n_validation:]
        X_train, y_train = X[training_indices], y[training_indices]
        X_validation, y_validation = X[validation_indices], y[validation_indices]
        if X_train.shape[0] < batch_size:
            raise ValueError("Training split must contain at least one full batch")

        best_params, best_validation_loss = self.params, float("inf")
        history = {"train_loss": [], "validation_loss": []}
        n_full_batches = X_train.shape[0] // batch_size
        for epoch in range(epochs):
            permutation = jax.random.permutation(
                jax.random.fold_in(key, epoch + 1), X_train.shape[0]
            )
            X_epoch, y_epoch = X_train[permutation], y_train[permutation]
            losses = []
            for batch_index in range(n_full_batches):
                start = batch_index * batch_size
                self.params, self.opt_state, loss = self.train_step(
                    self.params, self.opt_state,
                    X_epoch[start:start + batch_size],
                    y_epoch[start:start + batch_size],
                )
                losses.append(loss)

            train_loss = float(jnp.mean(jnp.asarray(losses)))
            validation_loss = float(self.loss_fn(
                self.params, X_validation, y_validation
            ))
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            if validation_loss < best_validation_loss:
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), self.params)
                best_validation_loss = validation_loss
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print(
                    f"epoch={epoch + 1:4d} train_mse={train_loss:.6f} "
                    f"validation_mse={validation_loss:.6f}"
                )

        self.params = best_params
        return history


def evaluate_student_policy(seed, trainer, env, num_steps=1000):
    """Returns one episode's return, masking rewards after termination."""
    state = env.reset(jax.random.key(seed))

    def step_fn(carry, _):
        state, active = carry
        action = trainer.predict(state.obs[None])[0]
        next_state = env.step(state, action)
        reward = jnp.where(active, next_state.reward, 0.0)
        active = active & jnp.logical_not(next_state.done)
        return (next_state, active), reward

    _, rewards = jax.lax.scan(step_fn, (state, jnp.asarray(True)), None, num_steps)
    return jnp.sum(rewards)


def save_student_checkpoint(
        path, trainer, obs_size, action_size, hidden_sizes, env_name=None,
):
    """Persists the selected BC policy and sufficient reconstruction metadata."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy.msgpack").write_bytes(serialization.to_bytes(trainer.params))
    metadata = {
        "format_version": 1,
        "observation_size": int(obs_size),
        "action_size": int(action_size),
        "hidden_sizes": list(hidden_sizes),
        "activation": "relu",
        "output_activation": "tanh",
        "env_name": env_name,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == '__main__':
    env_name = "inverted_pendulum"
    data = jnp.load(f"expert_datasets/expert_{env_name}.npz")
    expert_X = jnp.asarray(data["X"], dtype=jnp.float32)
    expert_y = jnp.asarray(data["y"], dtype=jnp.float32)
    if not jnp.isfinite(expert_X).all() or not jnp.isfinite(expert_y).all():
        raise ValueError("Dataset contains non-finite values")
    if jnp.max(jnp.abs(expert_y)) > 1.0 + 1e-5:
        raise ValueError("StudentPolicy uses tanh and requires actions in [-1, 1]")
    print(f"dataset: X={expert_X.shape}, y={expert_y.shape}")

    hidden_sizes = (128, 128, 128)
    trainer = StudentTrainer(
        StudentPolicy(action_size=expert_y.shape[1], hidden_sizes=hidden_sizes),
        obs_size=expert_X.shape[1], learning_rate=3e-4,
    )
    history = trainer.train(expert_X, expert_y, epochs=500, batch_size=256)
    print(f"best validation action MSE: {min(history['validation_loss']):.6f}")
    checkpoint_path = Path("student_checkpoints") / env_name / "final"
    save_student_checkpoint(
        checkpoint_path, trainer, expert_X.shape[1], expert_y.shape[1],
        hidden_sizes, env_name=env_name,
    )
    print(f"saved student checkpoint: {checkpoint_path}")

    eval_env = envs.create(env_name=env_name, backend="generalized")
    eval_fn = functools.partial(
        evaluate_student_policy, trainer=trainer, env=eval_env, num_steps=1000
    )
    print("student returns:", jax.vmap(eval_fn)(jnp.arange(10)))
