import pickle
from typing import Dict
import jax
import jax.numpy as jnp

from genepax.gp.cartesian_genetic_programming import CGP
from genepax.gp.sequential_genetic_programming import SequentialGP
from genepax.supervised_learning.dataset_utils import load_dataset

eps = 1e-6


def genome_size(conf: Dict, n_feats: int) -> int:
    try:
        file = open(f"../results/sensitivity/{conf['run_name']}.pickle", 'rb')
    except FileNotFoundError:
        print(f"../results/sensitivity/{conf['run_name']}.pickle")
        return -1
    repertoire = pickle.load(file)

    X_train, X_test, y_train, y_test = load_dataset(conf["problem"],
                                                    scale_x=conf.get("scale_x", False),
                                                    scale_y=conf.get("scale_y", False),
                                                    random_state=conf["seed"]
                                                    )
    n_features = n_feats
    if "seq" not in conf['run_name']:
        n_outputs = 1 if "feat" not in conf['run_name'] else n_features
        # Init the CGP policy graph with default values
        graph_structure = CGP(
            n_inputs=X_train.shape[1],
            n_outputs=n_outputs,
            n_nodes=conf["solver"]["n_nodes"],
            # n_input_constants=conf["solver"]["n_input_constants"],
            outputs_wrapper=lambda x: x,
        )
    else:
        first_cgp_structure = CGP(
            n_inputs=X_train.shape[1],
            n_outputs=n_features,
            n_nodes=conf["solver"]["n_nodes"],
            outputs_wrapper=lambda x: x,
        )
        second_cgp_structure = CGP(
            n_inputs=n_features,
            n_outputs=1,
            n_nodes=conf["solver"]["n_nodes"],
            outputs_wrapper=lambda x: x,
        )
        graph_structure = SequentialGP(first_cgp_structure, second_cgp_structure)

    best_idx = jnp.argmax(repertoire.fitnesses, axis=0)
    best_genotype = jax.tree.map(lambda x: x[best_idx][0], repertoire.genotypes)
    active_size = graph_structure.size(best_genotype)
    return active_size


if __name__ == '__main__':
    conf = {
        "solver": {"n_nodes": 50},
        "seed": 0,
        "tournament_size": 3,
        "problem": "chemical_2_competition",
        "scale_x": False,
        "scale_y": False,
    }

    problems = [
        "chemical_2_competition",
    ]

    with open("../results/genome_analysis_sensitivity.csv", "a") as f:
        f.write("problem,seed,algo,size,n_features\n")
        for seed in range(10):
            for extra in ['feats', 'featsls']:
                for n_feats in range(1, 57):
                    if n_feats % 2 != 0 or n_feats == 8:
                        continue
                    for problem in problems:
                        conf["problem"] = problem
                        conf["seed"] = seed
                        conf["run_name"] = (
                                f"CGP_{extra}_" + conf["problem"].replace("/", "_") + "_" + str(conf["seed"])
                                                                                                + "_" + str(n_feats)
                        )
                        # conf["repertoire_path"] = f"../results/{conf['run_name']}.pickle"
                        print(conf["run_name"])
                        active_size = genome_size(conf, n_feats)
                        if active_size >= 0:
                            f.write(f"{problem},{seed},{extra},{active_size},{n_feats}\n")
