import time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna, deap
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform, loguniform

if __name__ == "__main__":
    print("Optuna:", optuna.__version__, "| DEAP:", deap.__version__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25,
                                                        stratify=data.target, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    def cv_score(params):
        model = GradientBoostingClassifier(random_state=0, **params)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()

    def test_score(params):
        return GradientBoostingClassifier(random_state=0, **params).fit(X_train, y_train).score(X_test, y_test)

    summary = []
    default_cv = cv_score({})
    summary.append(("Default parameters", 1, default_cv, test_score({}), 0.0))
    print(f"Default GradientBoosting: CV accuracy {default_cv:.4f}")

    t = time.time()
    grid = GridSearchCV(GradientBoostingClassifier(random_state=0),
                        {"n_estimators": [50, 150, 300], "learning_rate": [0.01, 0.1, 0.3],
                         "max_depth": [2, 3, 5]},
                        cv=cv, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
    summary.append(("Grid search", len(grid.cv_results_["params"]), grid.best_score_,
                    grid.score(X_test, y_test), time.time() - t))
    print("Grid search best:", grid.best_params_, round(grid.best_score_, 4))

    t = time.time()
    rand = RandomizedSearchCV(GradientBoostingClassifier(random_state=0),
                              {"n_estimators": randint(50, 400), "learning_rate": loguniform(0.005, 0.5),
                               "max_depth": randint(1, 6), "subsample": uniform(0.5, 0.5),
                               "min_samples_leaf": randint(1, 20)},
                              n_iter=30, cv=cv, scoring="accuracy", n_jobs=-1,
                              random_state=0).fit(X_train, y_train)
    summary.append(("Random search", 30, rand.best_score_, rand.score(X_test, y_test), time.time() - t))
    print("Random search best:", {k: (round(float(v), 4) if isinstance(v, float) else int(v))
                                  for k, v in rand.best_params_.items()}, round(rand.best_score_, 4))

    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 50, 400),
                  "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
                  "max_depth": trial.suggest_int("max_depth", 1, 5),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)}
        return cv_score(params)

    t = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=30)
    summary.append(("Bayesian optimisation (Optuna TPE)", 30, study.best_value,
                    test_score(study.best_params), time.time() - t))
    print("Optuna best CV accuracy:", round(study.best_value, 4))
    print("Optuna best parameters :", {k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in study.best_params.items()})

    trials = study.trials_dataframe()
    plt.figure(figsize=(8, 3.5))
    plt.plot(trials["number"], trials["value"], "o", alpha=0.5, label="trial")
    plt.plot(trials["number"], trials["value"].cummax(), "r-", label="best so far")
    plt.xlabel("Trial"); plt.ylabel("CV accuracy"); plt.title("Optuna optimisation history")
    plt.legend(); plt.show()

    imp = optuna.importance.get_param_importances(study)
    print("Hyper-parameter importance:", {k: round(v, 3) for k, v in imp.items()})

    def decode(ind):
        g = [min(max(v, 0.0), 1.0) for v in ind]
        return {"n_estimators": int(50 + g[0] * 350),
                "learning_rate": float(10 ** (np.log10(0.005) + g[1] * 2)),
                "max_depth": int(1 + round(g[2] * 4)), "subsample": 0.5 + g[3] * 0.5,
                "min_samples_leaf": int(1 + g[4] * 19)}

    random.seed(0); np.random.seed(0)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("gene", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (cv_score(decode(ind)),))
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    t = time.time()
    pop = toolbox.population(n=8)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=4, stats=stats,
                                   halloffame=hof, verbose=True)
    evals = sum(rec["nevals"] for rec in log)
    ga_params = decode(hof[0])
    summary.append(("Genetic algorithm (DEAP)", evals, hof[0].fitness.values[0],
                    test_score(ga_params), time.time() - t))
    print("GA best CV accuracy:", round(hof[0].fitness.values[0], 4))
    print("GA best parameters :", {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in ga_params.items()})

    plt.figure(figsize=(7, 3))
    plt.plot(log.select("gen"), log.select("max"), "o-", label="best")
    plt.plot(log.select("gen"), log.select("avg"), "s--", label="average")
    plt.xlabel("Generation"); plt.ylabel("CV accuracy"); plt.title("Genetic algorithm progress")
    plt.legend(); plt.show()

    cmp = pd.DataFrame(summary, columns=["Method", "Models evaluated", "Best CV accuracy",
                                         "Test accuracy", "Time (s)"])
    print(cmp.round(4).to_string(index=False))
