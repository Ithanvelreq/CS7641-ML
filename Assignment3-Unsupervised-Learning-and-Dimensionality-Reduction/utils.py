import time
from sklearn.feature_selection import mutual_info_classif
from sklearn.random_projection import GaussianRandomProjection
import scipy.stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import learning_curve
from sklearn.utils import Bunch
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA
my_metrics = {"f1": metrics.f1_score, "accuracy": metrics.accuracy_score}


def plot_ig_results(X, y, X_train, y_train, X_test, y_test, metric, axes=None, return_X=-1):
    nn_score_list = []
    now = time.perf_counter()
    ig_scores = mutual_info_classif(X, y)
    wall_clock = (time.perf_counter() - now)*1e3

    # Sort the features
    indices_sorted = []
    value_to_index = {}
    for i, key in enumerate(ig_scores):
        if value_to_index.get(key, None) is None:
            value_to_index[key] = [i]
        else:
            value_to_index[key].append(i)

    ig_scores.sort()
    ig_scores = np.flip(ig_scores)
    for score in ig_scores:
        indices_sorted.append(value_to_index[score][0])
        del value_to_index[score][0]

    if return_X != -1:
        return X[:, indices_sorted[:return_X+1]]

    for i in range(1, len(indices_sorted)+1):
        X_train_transofrmed = X_train[:, indices_sorted[:i]]
        X_test_transformed = X_test[:, indices_sorted[:i]]
        clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10, 10), random_state=42)
        clf.fit(X_train_transofrmed, y_train)
        y_pred = clf.predict(X_test_transformed)
        score = my_metrics[metric](y_pred, y_test)
        nn_score_list.append(score)

    if axes is None:
        _, axes = plt.subplots(1, 4)
    # Plot wall clock time
    x_axis = [i for i in range(1, X.shape[1]+1)]
    axes[0].set_title("Wall clock time (ms)")
    axes[0].set_xlabel("Number of components")
    axes[0].grid()
    axes[0].plot(x_axis, [wall_clock]*len(x_axis))
    # Plot variance ratio
    axes[1].set_title("Information Gain per component")
    axes[1].set_xlabel("Number of components")
    axes[1].grid()
    axes[1].plot(x_axis, ig_scores)
    # Plot Neural Network score
    axes[2].set_title(f"Neural Network {metric} score")
    axes[2].set_xlabel("Number of components")
    axes[2].grid()
    axes[2].plot(x_axis, nn_score_list)
    return plt


def plot_RP_results(seeds, X, X_train, y_train, X_test, y_test, metric, axes=None):
    wall_clock_time_list = []
    nn_score_list = []
    reconstruction_error_list = []
    variance = []
    plot_options = ["r", "g", "b"]
    if axes is None:
        _, axes = plt.subplots(1, 4)
    for i, seed in enumerate(seeds):
        for nb_c in range(1, X.shape[1] + 1):
            transformer = GaussianRandomProjection(random_state=seed, n_components=nb_c)
            now = time.perf_counter()
            transformer.fit(X)
            wall_clock = (time.perf_counter() - now) * 1e3
            wall_clock_time_list.append(wall_clock)

            X_train_transofrmed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)
            clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10, 10), random_state=42)
            clf.fit(X_train_transofrmed, y_train)
            y_pred = clf.predict(X_test_transformed)
            score = my_metrics[metric](y_pred, y_test)
            nn_score_list.append(score)

            X_transformed = transformer.transform(X)
            X_reconstructed = transformer.inverse_transform(X_transformed)
            reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1).mean()
            reconstruction_error_list.append(reconstruction_error)
            variance.append(np.var(X_transformed, axis=0).mean())

        # Plot wall clock time
        x_axis = [i for i in range(1, X.shape[1] + 1)]
        axes[0].set_title("Wall clock time (ms)")
        axes[0].set_xlabel("Number of principal components")
        axes[0].grid()
        axes[0].plot(x_axis, wall_clock_time_list, color=plot_options[i], label=f"random seed {seed}")
        axes[0].legend(loc="best")
        # Plot variance ratio
        axes[1].set_title("Mean variance per component")
        axes[1].set_xlabel("Number of principal components")
        axes[1].grid()
        axes[1].plot(x_axis, variance, color=plot_options[i], label=f"random seed {seed}")
        axes[1].legend(loc="best")
        # Plot reconstruction error
        axes[2].set_title("Reconstruction error mse")
        axes[2].set_xlabel("Number of principal components")
        axes[2].grid()
        axes[2].plot(x_axis, reconstruction_error_list, color=plot_options[i], label=f"random seed {seed}")
        axes[2].legend(loc="best")
        # Plot Neural Network score
        axes[3].set_title(f"Neural Network {metric} score")
        axes[3].set_xlabel("Number of principal components")
        axes[3].grid()
        axes[3].plot(x_axis, nn_score_list, color=plot_options[i], label=f"random seed {seed}")
        axes[3].legend(loc="best")

        wall_clock_time_list = []
        nn_score_list = []
        reconstruction_error_list = []
        variance = []
    return plt


def plot_transformer_results(algorithm, X, X_train, y_train, X_test, y_test, metric, axes=None):
    wall_clock_time_list = []
    nn_score_list = []
    reconstruction_error_list = []
    kurtosis = []
    for nb_c in range(1, X.shape[1]+1):
        if algorithm == "PCA":
            transformer = PCA(n_components=nb_c)
        elif algorithm == "ICA":
            transformer = FastICA(n_components=nb_c)
        else:
            raise Exception("Not Implemented")
        now = time.perf_counter()
        transformer.fit(X)
        wall_clock = (time.perf_counter() - now)*1e3
        wall_clock_time_list.append(wall_clock)

        X_train_transofrmed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)
        clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10, 10), random_state=42)
        clf.fit(X_train_transofrmed, y_train)
        y_pred = clf.predict(X_test_transformed)
        score = my_metrics[metric](y_pred, y_test)
        nn_score_list.append(score)


        X_transformed = transformer.transform(X)
        X_reconstructed = transformer.inverse_transform(X_transformed)
        reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1).mean()
        reconstruction_error_list.append(reconstruction_error)

        if algorithm == "ICA":
            kurtosis.append(np.abs(scipy.stats.kurtosis(X_transformed)).mean())

    if algorithm == "PCA":
        transformer = PCA(n_components=X.shape[1])
        transformer.fit(X)
        particular_title = "Variance ratio per principal component"
        particular_data = transformer.explained_variance_ratio_
    elif algorithm == "ICA":
        particular_title = "Absolute mean kurtosis per component"
        particular_data = kurtosis
    else:
        raise Exception("Not Implemented")

    if axes is None:
        _, axes = plt.subplots(1, 4)
    # Plot wall clock time
    x_axis = [i for i in range(1, X.shape[1]+1)]
    axes[0].set_title("Wall clock time (ms)")
    axes[0].set_xlabel("Number of principal components")
    axes[0].grid()
    axes[0].plot(x_axis, wall_clock_time_list)
    # Plot variance ratio
    axes[1].set_title(particular_title)
    axes[1].set_xlabel("Number of principal components")
    axes[1].grid()
    axes[1].plot(x_axis, particular_data)
    # Plot reconstruction error
    axes[2].set_title("Reconstruction error mse")
    axes[2].set_xlabel("Number of principal components")
    axes[2].grid()
    axes[2].plot(x_axis, reconstruction_error_list)
    # Plot Neural Network score
    axes[3].set_title(f"Neural Network {metric} score")
    axes[3].set_xlabel("Number of principal components")
    axes[3].grid()
    axes[3].plot(x_axis, nn_score_list)
    return plt


def plot_silhouette(clustering_algorithm, X, range_n_clusters, xlim=None):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        if xlim is None:
            xlim = [-0.5, 1]
        ax1.set_xlim(xlim)
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        if clustering_algorithm == "KMeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif clustering_algorithm == "GaussianMixture":
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            raise Exception("Not implemented")
        now = time.perf_counter()
        cluster_labels = clusterer.fit_predict(X)
        end = time.perf_counter() - now
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters} The average silhouette_score is :{silhouette_avg:.3f}"
              f". Took {end*1e3:.1f} ms")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"The silhouette plot for the various clusters. {clustering_algorithm} and {n_clusters} clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks(np.arange(xlim[0], xlim[1]+.2, .2))

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        if clustering_algorithm == "KMeans":
            centers = clusterer.cluster_centers_
        else:
            centers = clusterer.means_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for {clustering_algorithm} clustering on sample data with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

    return plt


def my_load_wine(path_to_files, return_X_y=False):
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
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    clusterer = KMeans(n_clusters=2, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    X = np.concatenate((X, np.expand_dims(cluster_labels, axis=1)), axis=1)
    clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10, 10), random_state=42)
    title = f"Added K-Means labels"
    plot_learning_curve(clf, title, X_train, y_train, axes=None, detailed=True)

