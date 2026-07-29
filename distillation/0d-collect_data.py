from distillation.dataset_collection import generate_expert_dataset

if __name__ == "__main__":
    for env_name in [
        "inverted_pendulum",
        "inverted_double_pendulum",
        "hopper",
        "walker2d",
        "halfcheetah",
    ]:

        checkpoint_path = f"./checkpoints/{env_name}/final"

        dataset_path = f"expert_datasets/expert_{env_name}.npz"

        X, y = generate_expert_dataset(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            num_envs=10,
            episode_length=1000,
            seed=0,
        )
