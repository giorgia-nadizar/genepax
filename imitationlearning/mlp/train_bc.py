from imitationlearning.dataset import load_d4rl_dataset


def main():
    dataset_id = "mujoco/invertedpendulum/expert-v0"  # or Minari equivalent

    obs, act = load_d4rl_dataset(dataset_id)

    print("observations:", obs.shape)
    print("actions:", act.shape)

    print("obs mean:", obs.mean(axis=0))
    print("obs std:", obs.std(axis=0))


if __name__ == "__main__":
    main()