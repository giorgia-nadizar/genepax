from distillation.dataset_collection import generate_expert_dataset
from pathlib import Path

if __name__ == "__main__":
    experiments_dir = Path(__file__).resolve().parents[1]
    NUM_ENVS = 32
    NUM_ROLLOUTS = 4
    EPISODE_LENGTH = 1000

    for env_name in [
        "inverted_pendulum",
        "inverted_double_pendulum",
        "hopper",
        "walker2d",
        "halfcheetah",
    ]:

        checkpoint_path = experiments_dir / "expert_models" / env_name / "final"
        dataset_path = experiments_dir / "expert_datasets" / f"expert_{env_name}.npz"

        X, y = generate_expert_dataset(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            num_envs=NUM_ENVS,
            num_rollouts=NUM_ROLLOUTS,
            episode_length=EPISODE_LENGTH,
            seed=0,
            verbose=True,
        )
