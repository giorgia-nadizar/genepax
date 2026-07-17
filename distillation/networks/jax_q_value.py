import jax.numpy as jnp


class JaxQValueEstimator:

    def __init__(
            self,
            q1,
            q2,
            q1_params,
            q2_params
    ):
        self.q1 = q1
        self.q2 = q2
        self.q1_params = q1_params
        self.q2_params = q2_params

    def __call__(
            self,
            obs,
            action
    ):
        q1 = self.q1.apply(
            self.q1_params,
            obs,
            action
        )

        q2 = self.q2.apply(
            self.q2_params,
            obs,
            action
        )

        return jnp.minimum(
            q1,
            q2
        )

    def batched_q(self, obs, actions, batch_size=4096):
        qs = []

        for i in range(0, len(obs), batch_size):
            q = self.__call__(
                obs[i:i + batch_size],
                actions[i:i + batch_size]
            )

            qs.append(q)

        return jnp.concatenate(qs)
