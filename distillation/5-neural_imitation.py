import functools
from functools import partial

import jax
import optax

import jax.numpy as jnp
from brax import envs

from distillation.networks.neural_student import StudentPolicy


class StudentTrainer:
    def __init__(
            self,
            model,
            obs_size,
            action_size,
            learning_rate=3e-4,
            seed=0,
    ):
        self.model = model

        key = jax.random.key(seed)

        dummy_obs = jnp.zeros(
            (1, obs_size),
            dtype=jnp.float32,
        )

        self.params = self.model.init(
            key,
            dummy_obs,
        )

        self.optimizer = optax.adam(
            learning_rate
        )

        self.opt_state = self.optimizer.init(
            self.params
        )

    def loss_fn(
            self,
            params,
            X,
            y,
    ):
        pred = self.model.apply(
            params,
            X,
        )

        loss = jnp.mean(
            (pred - y) ** 2
        )

        return loss

    @partial(jax.jit, static_argnums=0)
    def train_step(
            self,
            params,
            opt_state,
            X,
            y,
    ):

        loss, grads = jax.value_and_grad(
            self.loss_fn
        )(
            params,
            X,
            y,
        )

        updates, opt_state = self.optimizer.update(
            grads,
            opt_state,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        return params, opt_state, loss

    def train(
            self,
            X,
            y,
            epochs=100,
            batch_size=256,
            shuffle=True,
    ):

        n_samples = X.shape[0]

        for epoch in range(epochs):

            if shuffle:
                key = jax.random.key(epoch)

                indices = jax.random.permutation(
                    key,
                    n_samples,
                )

                X_epoch = X[indices]
                y_epoch = y[indices]

            else:
                X_epoch = X
                y_epoch = y

            losses = []

            for i in range(0, n_samples, batch_size):
                X_batch = X_epoch[
                    i:i + batch_size
                ]

                y_batch = y_epoch[
                    i:i + batch_size
                ]

                self.params, self.opt_state, loss = self.train_step(
                    self.params,
                    self.opt_state,
                    X_batch,
                    y_batch,
                )

                losses.append(loss)

            if epoch % 10 == 0:
                print(
                    f"epoch {epoch}, "
                    f"loss {float(jnp.mean(jnp.array(losses)))}"
                )

    def predict(
            self,
            obs,
    ):
        return self.model.apply(
            self.params,
            obs,
        )


def evaluate_student_policy(
        seed,
        trainer,
        env,
        num_steps=1000,
):
    key = jax.random.key(seed)

    state = env.reset(key)

    def step_fn(state, _):
        action = trainer.predict(
            state.obs[None]
        )[0]

        next_state = env.step(
            state,
            action,
        )

        return next_state, (
            next_state.reward,
            next_state.done,
        )

    _, (rewards, dones) = jax.lax.scan(
        step_fn,
        state,
        None,
        length=num_steps,
    )

    alive = 1.0 - jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.cumsum(
                dones[:-1].astype(jnp.float32)
            ),
        ]
    )

    alive = jnp.clip(
        alive,
        0,
        1,
    )

    return jnp.sum(
        rewards * alive
    )


if __name__ == '__main__':
    epochs = 200
    ENV_NAME = "inverted_double_pendulum"
    eval_env = envs.get_environment(
        env_name=ENV_NAME,
    )

    # we have already collected the initial dataset at expert_dataset.npz
    dataset_path = f"expert_dataset_{ENV_NAME}.npz"
    data = jnp.load(dataset_path)

    expert_X = data["X"]
    expert_y = data["y"]

    student = StudentPolicy(
        action_size=expert_y.shape[1],
        hidden_sizes=(128, 128, 128),
    )

    trainer = StudentTrainer(
        model=student,
        obs_size=expert_X.shape[1],
        action_size=expert_y.shape[1],
        learning_rate=3e-4,
    )

    trainer.train(
        expert_X,
        expert_y,
        epochs=epochs,
        batch_size=256,
    )

    print("moving to evaluation")
    eval_fn = functools.partial(
        evaluate_student_policy,
        trainer=trainer,
        env=eval_env,
        num_steps=1000,
    )
    rewards = jax.vmap(eval_fn)(jnp.arange(10))
    print(rewards)
