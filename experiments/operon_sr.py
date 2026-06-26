from pyoperon.sklearn import SymbolicRegressor
import numpy as np
from sklearn.metrics import r2_score

from genepax.supervised_learning.dataset_utils import load_dataset

for seed in range(10):
    for problem in ["4544_GeographicalOriginalofMusic", "505_tecator", ]:
        X_train, X_test, y_train, y_test = load_dataset(
            problem,
            # scale_x=config.get("scale_x", False),
            # scale_y=config.get("scale_y", False),
            random_state=seed,
        )

        # NOTE: Operon takes the ravel of y
        reg = SymbolicRegressor(
            allowed_symbols="add,sub,mul,aq,sin,constant,variable",
            brood_size=10,
            comparison_factor=0,
            crossover_internal_probability=0.9,
            crossover_probability=1.0,
            epsilon=1e-05,
            female_selector="tournament",
            generations=1000,
            initialization_max_depth=5,
            initialization_max_length=10,
            initialization_method="btc",
            irregularity_bias=0.0,
            local_search_probability=1.0,
            lamarckian_probability=1.0,
            optimizer_iterations=1,
            optimizer='lm',
            male_selector="tournament",
            max_depth=10,
            max_evaluations=1000000,
            max_length=50,
            max_selection_pressure=100,
            model_selection_criterion="minimum_description_length",
            mutation_probability=0.25,
            n_threads=32,
            objectives=['r2', 'length'],
            offspring_generator="os",
            pool_size=1000,
            population_size=1000,
            random_state=seed,
            reinserter="keep-best",
            # max_time=900,
            tournament_size=3,
            # uncertainty= [sErr],
            add_model_intercept_term=True,
            add_model_scale_term=True
        )

        reg.fit(X_train, y_train.ravel())

        train_r2_scores = [s['objective_values'][0] for s in reg.pareto_front_]
        model_sizes = [s['objective_values'][1] for s in reg.pareto_front_]
        best_train_r2_score = min(train_r2_scores)
        best_train_r2_score_idx = np.argmin(train_r2_scores)
        best_model_size = model_sizes[best_train_r2_score_idx]
        test_r2_score = r2_score(y_test, reg.predict(X_test))
        print(problem, seed)

        with open(f"../results/operon_{problem}_{seed}.csv", "a") as f:
            f.write("max_fitness,test_accuracy,size\n")
            f.write(f"{-best_train_r2_score},{test_r2_score},{best_model_size}\n")
