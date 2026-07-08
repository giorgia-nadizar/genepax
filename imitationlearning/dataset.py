import minari
import numpy as np



def load_d4rl_dataset(dataset_id: str):
    """
    Returns flattened (observations, actions)
    from a Minari dataset.
    """
    print(dataset_id)

    dataset = minari.load_dataset(dataset_id, download=True)

    observations = []
    actions = []

    for episode in dataset:
        obs = episode.observations
        act = episode.actions

        observations.append(obs[:-1])
        actions.append(act)

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)

    return observations, actions