import time
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np
import os.path
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.utils import Bunch
from sklearn import metrics
from sklearn.model_selection import train_test_split
my_metrics = {"f1": metrics.f1_score, "accuracy": metrics.accuracy_score}


class MyKColors(mlrose.MaxKColor):

    def __init__(self, edges):
        super().__init__(edges)

    def evaluate(self, state):
        return -super().evaluate(state)


problems = {
    "four_peaks": {
        "fitness_function": mlrose.FourPeaks(t_pct=0.05),
        "problem_max_val": 2
    },
    "flip_flop": {
        "fitness_function": mlrose.FlipFlop(),
        "problem_max_val": 2
    },
    "one_max": {
        "fitness_function": mlrose.OneMax(),
        "problem_max_val": 2
    }
}


def generate_problem(problem_name, problem_length):
    if problem_name == "max_3_colors":
        edges = generate_random_graph(problem_length)
        fitness_function = MyKColors(edges)
        return mlrose.DiscreteOpt(length=problem_length, fitness_fn=fitness_function,
                                  maximize=True, max_val=3)
    else:
        return mlrose.DiscreteOpt(length=problem_length, fitness_fn=problems[problem_name]["fitness_function"],
                                  maximize=True, max_val=problems[problem_name]["problem_max_val"])


def generate_random_graph(nb_edges):
    edges = []
    for i in range(nb_edges):
        e = np.random.randint(0, high=nb_edges, size=(2,))
        edges.append((e[0], e[1]))
    return edges


def run_algorithm(algorithm, to_append_metrics, problem_name, problem_length, plot_curve=False):
    problem = generate_problem(problem_name, problem_length)
    now = time.perf_counter()
    _, score, curve = algorithm(problem, random_state=42, curve=plot_curve)
    to_append_metrics.append([score, time.perf_counter() - now, problem.current_iteration, problem.fitness_evaluations])
    if plot_curve:
        return curve


def plot_scalability_curves(problem_name, problem_lengths, axes=None):
    sa_metrics = []
    rhc_metrics = []
    ga_metrics = []
    mimic_metrics = []
    for problem_length in problem_lengths:
        if (problem_length < 2) or (problem_length < 3 and problem_name == "Max3Colors"):
            raise Exception("The problem length has to be at least 3 for Max3Colors and 2 for the others")
        # Simulated annealing
        run_algorithm(mlrose.simulated_annealing, sa_metrics, problem_name, problem_length)
        # Randomized hill climbing
        run_algorithm(mlrose.random_hill_climb, rhc_metrics, problem_name, problem_length)
        # Genetic algorithms
        run_algorithm(mlrose.genetic_alg, ga_metrics, problem_name, problem_length)
        # MIMIC
        run_algorithm(mlrose.mimic, mimic_metrics, problem_name, problem_length)

    sa_metrics = np.array(sa_metrics)
    rhc_metrics = np.array(rhc_metrics)
    ga_metrics = np.array(ga_metrics)
    mimic_metrics = np.array(mimic_metrics)
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(20, 5))
    # Plot score got by each algorithm
    axes[0, 0].set_xlabel("Problem sizes")
    axes[0, 0].set_ylabel("Fitness function score")
    axes[0, 0].grid()
    axes[0, 0].plot(
        problem_lengths, sa_metrics[:, 0], "o-", color="r", label="SA"
    )
    axes[0, 0].plot(
        problem_lengths, rhc_metrics[:, 0], "o-", color="g", label="RHC"
    )
    axes[0, 0].plot(
        problem_lengths, ga_metrics[:, 0], "o-", color="b", label="GA"
    )
    axes[0, 0].plot(
        problem_lengths, mimic_metrics[:, 0], "o-", color="k", label="MIMIC"
    )
    axes[0, 0].legend(loc="best")
    # Plot wall clock time
    axes[0, 1].set_xlabel("Problem sizes")
    axes[0, 1].set_ylabel("Wall clock time log(ns)")
    axes[0, 1].grid()
    axes[0, 1].plot(
        problem_lengths, np.log(sa_metrics[:, 1]), "o-", color="r", label="SA"
    )
    axes[0, 1].plot(
        problem_lengths, np.log(rhc_metrics[:, 1]), "o-", color="g", label="RHC"
    )
    axes[0, 1].plot(
        problem_lengths, np.log(ga_metrics[:, 1]), "o-", color="b", label="GA"
    )
    axes[0, 1].plot(
        problem_lengths, np.log(mimic_metrics[:, 1]), "o-", color="k", label="MIMIC"
    )
    axes[0, 1].legend(loc="best")
    # Plot number of iterations
    axes[1, 0].set_xlabel("Problem sizes")
    axes[1, 0].set_ylabel("Log Number of iterations")
    axes[1, 0].grid()
    axes[1, 0].plot(
        problem_lengths, np.log(sa_metrics[:, 2]), "o-", color="r", label="SA"
    )
    axes[1, 0].plot(
        problem_lengths, np.log(rhc_metrics[:, 2]), "o-", color="g", label="RHC"
    )
    axes[1, 0].plot(
        problem_lengths, np.log(ga_metrics[:, 2]), "o-", color="b", label="GA"
    )
    axes[1, 0].plot(
        problem_lengths, np.log(mimic_metrics[:, 2]), "o-", color="k", label="MIMIC"
    )
    axes[1, 0].legend(loc="best")
    # Plot number of function evaluations
    axes[1, 1].set_xlabel("Problem sizes")
    axes[1, 1].set_ylabel("Log Number of function evaluations")
    axes[1, 1].grid()
    axes[1, 1].plot(
        problem_lengths, np.log(sa_metrics[:, 3]), "o-", color="r", label="SA"
    )
    axes[1, 1].plot(
        problem_lengths, np.log(rhc_metrics[:, 3]), "o-", color="g", label="RHC"
    )
    axes[1, 1].plot(
        problem_lengths, np.log(ga_metrics[:, 3]), "o-", color="b", label="GA"
    )
    axes[1, 1].plot(
        problem_lengths, np.log(mimic_metrics[:, 3]), "o-", color="k", label="MIMIC"
    )
    axes[1, 1].legend(loc="best")
    return plt


def add_loss_curve_to_axe(title, axe, curve, color):
    axe.set_title(title)
    axe.set_xlabel("Iteration")
    axe.set_ylabel("Fitness function score")
    axe.grid()
    axe.plot(curve[:, 1], curve[:, 0], color=color)


def plot_loss_curves(problem_name, problem_length, axes=None):

    if (problem_length < 2) or (problem_length < 3 and problem_name == "Max3Colors"):
        raise Exception("The problem length has to be at least 3 for Max3Colors and 2 for the others")
    # Simulated annealing
    sa_curve = run_algorithm(mlrose.simulated_annealing, [], problem_name, problem_length, plot_curve=True)
    # Randomized hill climbing
    rhc_curve = run_algorithm(mlrose.random_hill_climb, [], problem_name, problem_length, plot_curve=True)
    # Genetic algorithms
    ga_curve = run_algorithm(mlrose.genetic_alg, [], problem_name, problem_length, plot_curve=True)
    # MIMIC
    mimic_curve = run_algorithm(mlrose.mimic, [], problem_name, problem_length, plot_curve=True)

    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(20, 5))
    # Plot score got by each algorithm
    add_loss_curve_to_axe("Loss curve for SA", axes[0, 0], sa_curve, "r")
    add_loss_curve_to_axe("Loss curve for RHC", axes[0, 1], rhc_curve, "g")
    add_loss_curve_to_axe("Loss curve for GA", axes[1, 0], ga_curve, "b")
    add_loss_curve_to_axe("Loss curve for MIMIC", axes[1, 1], mimic_curve, "k")
    return plt


def plot_fitness_curves(curve, axe, title):
    axe.set_title(title)
    axe.set_xlabel("Iteration")
    axe.set_ylabel("Fitness function score")
    axe.grid()
    axe.plot(curve[:, 1], curve[:, 0], color="r")
    return plt


def one_hot_encoding(target):
    hot_targets = np.zeros((target.shape[0], 10))
    for i, t in enumerate(target):
        hot_targets[i, t - 1] = 1
    return hot_targets


def my_load_wine(path_to_files, return_X_y=False, hot_encode=False):
    path_red = os.path.join(path_to_files, "winequality-red.csv")
    path_white = os.path.join(path_to_files, "winequality-white.csv")
    df_red = pd.read_csv(path_red, delimiter=";")
    df_white = pd.read_csv(path_white, delimiter=";")
    red_type_list = [1] * len(df_red.index)  # We encode red wines by 1
    white_type_list = [0]*len(df_white.index)  # We encode white wines by 0
    df_red["type"] = red_type_list
    df_white["type"] = white_type_list
    df_all_wines = pd.concat([df_red, df_white])
    target = df_all_wines["quality"]
    df_all_wines.drop("quality", axis=1, inplace=True)
    target = target.to_numpy()
    data = df_all_wines.to_numpy()
    if hot_encode:
        target = one_hot_encoding(target)
    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=df_all_wines,
        feature_names=list(df_all_wines),
    )


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    detailed=False
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Borrowed from a scikit learn tutorial
    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    detailed : bool
        Dispaly detailed info about learning curves
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    if detailed:
        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

    return plt


def plot_loss_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))
    axes = [axes]

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Number of iterations")
    axes[0].set_ylabel("Log-loss value")
    estimator.fit(X, y)
    # Plot learning curve
    axes[0].grid()
    if estimator.algorithm == "gradient_descent":
        axes[0].plot(np.negative(estimator.fitness_curve), color="r", label="Training score")
    else:
        axes[0].plot(estimator.fitness_curve[:, 1], estimator.fitness_curve[:, 0], color="r", label="Training score")
    axes[0].legend(loc="best")
    return plt


def get_estimator_final_score(estimator, X_train, y_train, X_test, y_test, title, metric, print_result=True):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    score = my_metrics[metric](y_pred, y_test)
    if print_result:
        print(f"Test {metric} score for {title}: {score}")
    return score


if __name__ == '__main__':
    wine = my_load_wine("../Datasets/wine/", hot_encode=True)
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    algorithms = ["gradient_descent", "random_hill_climb", "simulated_annealing", "genetic_alg"]
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, alg in enumerate(algorithms):
        axe_to_plot = axes[i]
        estimator = mlrose.NeuralNetwork([10], algorithm=alg, max_iters=1000, early_stopping=True, max_attempts=10,
                                         random_state=42, curve=True)
        estimator.fit(X_train, y_train)
        title = f"Algorithm: {alg}"
        plot_loss_curve(estimator, title, X_train, y_train, axes=axe_to_plot)

