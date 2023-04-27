import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from datasets.generate_datasets import (
    AGENT_NAMES,
    DOPAMINE_DQN_ATARI_AGENT_ENVS,
    DOPAMINE_RAINBOW_ATARI_AGENT_ENVS,
    ENV_NAMES,
)

MARKER_SIZE = 100
COLOURS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:pink",
    "tab:cyan",
]


def plot_window_metrics(
    clustering_results,
    n_clusters: list[int],
    window_sizes: list[int],
    metric: str,
    y_label: str,
    optimal_index: tuple[int, int],
    hand_cluster_results: Optional[dict[str, float]],
    title: str,
    filename: str,
    figsize: tuple[int, int] = (8, 6),
):
    assert clustering_results.shape == (len(n_clusters), len(window_sizes))
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel("Number of Clusters", fontsize=13)
    ax.set_xticks(n_clusters)
    ax.set_xticklabels(n_clusters)

    for i in range(len(window_sizes)):
        ax.plot(
            n_clusters, clustering_results[:, i], label=f"Window size={window_sizes[i]}"
        )
    ax.scatter(
        n_clusters[optimal_index[0]],
        clustering_results[optimal_index],
        marker="*",
        label="Optimal Model",
        s=MARKER_SIZE,
    )
    if hand_cluster_results:
        ax.scatter(
            hand_cluster_results["n_clusters"],
            hand_cluster_results[metric],
            marker="X",
            label="Hand Clustered",
            s=MARKER_SIZE,
        )
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_window_all_metrics(
    clustering_results,
    metrics: list[str],
    y_labels: list[str],
    n_clusters: list[int],
    window_sizes: list[int],
    nrows: int,
    ncols: int,
    optimal_index: tuple[int, int],
    hand_cluster_results: Optional[dict[str, float]],
    title: str,
    filename: str,
    figsize: tuple[int, int] = (15, 10),
):
    assert len(clustering_results) == len(metrics)
    assert all(
        cluster_result.shape == (len(n_clusters), len(window_sizes))
        for cluster_result in clustering_results
    )
    assert nrows * ncols >= len(clustering_results)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.suptitle(title, fontsize=18)

    for i, ax in enumerate(axs.flatten()):
        metric = metrics[i].title().replace("_", " ")
        ax.set_title(metric, fontsize=16)
        ax.set_ylabel(y_labels[i], fontsize=13)
        ax.set_xlabel("Number of Clusters", fontsize=13)
        ax.set_xticks(n_clusters)
        ax.set_xticklabels(n_clusters)

        for j in range(len(window_sizes)):
            ax.plot(
                n_clusters,
                clustering_results[i][:, j],
                label=f"Window size={window_sizes[j]}",
            )
        ax.scatter(
            n_clusters[optimal_index[0]],
            clustering_results[i][optimal_index],
            marker="*",
            label="Optimal Model",
            s=MARKER_SIZE,
        )
        if hand_cluster_results is not None:
            ax.scatter(
                hand_cluster_results["n_clusters"],
                hand_cluster_results[metrics[i]],
                marker="X",
                label="Hand Clustered",
                s=MARKER_SIZE,
            )
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def plot_comparison_window_all_metrics(
    comparison_names: list[str],
    comparison_results: list,
    metrics: list[str],
    y_labels: list[str],
    line_styles: list[str],
    n_clusters: list[int],
    window_sizes: list[int],
    nrows: int,
    ncols: int,
    comparison_optimal_index: list[tuple[int, int]],
    comparison_hand_cluster_results: list[dict[str, float]],
    title: str,
    filename: str,
    figsize: tuple[int, int] = (15, 10),
):
    assert len(comparison_names) == len(comparison_results)
    assert all(
        len(clustering_results) == len(metrics)
        for clustering_results in comparison_results
    )
    assert all(
        cluster_result.shape == (len(n_clusters), len(window_sizes))
        for clustering_results in comparison_results
        for cluster_result in clustering_results
    )
    assert nrows * ncols >= len(comparison_results[0])

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.suptitle(title, fontsize=18)

    for i, ax in enumerate(axs.flatten()):
        metric = metrics[i].title().replace("_", " ")
        ax.set_title(metric, fontsize=16)
        ax.set_ylabel(y_labels[i], fontsize=13)
        ax.set_xlabel("Number of Clusters", fontsize=13)
        ax.set_xticks(n_clusters)
        ax.set_xticklabels(n_clusters)

        for k in range(len(comparison_names)):
            for j in range(len(window_sizes)):
                ax.plot(
                    n_clusters,
                    comparison_results[k][i][:, j],
                    label=comparison_names[k],
                    linestyle=line_styles[k],
                    c=COLOURS[j],
                )

            ax.scatter(
                n_clusters[comparison_optimal_index[k][0]],
                comparison_results[k][i][comparison_optimal_index[k]],
                marker="*",
                label=f"Optimal Model for {comparison_names[k]}",
                s=MARKER_SIZE,
            )

        # if hand_cluster_results is not None:
        #     ax.scatter(
        #         hand_cluster_results["n_clusters"],
        #         hand_cluster_results[metrics[i]],
        #         marker="X",
        #         label="Hand Clustered",
        #         s=MARKER_SIZE,
        #     )
        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        ax.legend(*zip(*unique))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


SAVE_ROOT_FOLDER = "figs/"
DATASET_ROOT_FOLDER = "../datasets/"
METRICS = [
    "plan_entropy",
    "skill_length_distribution",
    "skill_alignment",
    "skill_length",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "silhouette_score",
    "intensity_factor",
]
METRIC_Y_LABELS = [
    "Entropy",
    "p-values",
    "Normalised alignment",
    "Average length",
    "Ratio of cluster dispersion",
    "Cluster similarity",
    "Silhouette Coefficient",
    "Proportion of cluster transitions",
]
REDUCED_METRIC_INDEXES = [0, 1, 2, 3]
REDUCED_METRICS = [METRICS[i] for i in REDUCED_METRIC_INDEXES]
REDUCED_Y_LABELS = [METRIC_Y_LABELS[i] for i in REDUCED_METRIC_INDEXES]
assert len(METRICS) == len(METRIC_Y_LABELS)
LINE_STYLES = ["solid", "dashed"]


if __name__ == "__main__":
    # ===== Individual clustering =====
    for _agent_name, _env_name in tqdm(
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        dataset_folder = f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}"
        save_folder = f"{SAVE_ROOT_FOLDER}{_agent_name}-{_env_name}"

        agent_env_name = f'{_agent_name.replace("_", " ")} {_env_name}'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        for clustering_algo, algo_title in [
            ("kmeans-st", "Spatio-Temporal KMeans"),
            ("kmeans-st-state-value", "Spatio-Temporal KMeans with State-value"),
        ]:
            # Gathers the clustering metric results
            if os.path.exists(f"{dataset_folder}/{clustering_algo}/summary.npz"):
                print(dataset_folder, clustering_algo)
                with np.load(
                    f"{dataset_folder}/{clustering_algo}/summary.npz",
                    allow_pickle=True,
                ) as file:
                    clustering_metric_results = [file[metric] for metric in METRICS]
                    n_clusters = file["n_clusters"]
                    window_sizes = file["window_sizes"]

                    optimal_model = file["optimal_model"].item()
                    optimal_index = optimal_model["index"]
                    optimal_n_clusters = optimal_model["n_clusters"]
                    optimal_window_size = optimal_model["window_size"]

                # Gathers hand clustering results
                hand_cluster_results = None
                if os.path.exists(f"{dataset_folder}/hand-clustering.npz"):
                    with np.load(f"{dataset_folder}/hand-clustering.npz") as file:
                        if file["completed"].item():
                            hand_cluster_results = {"n_clusters": file["n_clusters"]}
                            for metric in METRICS:
                                hand_cluster_results[metric] = file[metric]
                        else:
                            print(
                                "Hand clustering exists but is not complete",
                                file=sys.stderr,
                            )

                # Plot the window results
                for i in range(len(METRICS)):
                    plot_window_metrics(
                        clustering_metric_results[i],
                        n_clusters,
                        window_sizes,
                        METRICS[i],
                        METRIC_Y_LABELS[i],
                        optimal_index,
                        hand_cluster_results,
                        title=f"{algo_title} {METRICS[i]} for {agent_env_name}",
                        filename=f"{save_folder}/{clustering_algo}-window-{METRICS[i]}.png",
                    )

                for i in REDUCED_METRIC_INDEXES:
                    plot_window_metrics(
                        clustering_metric_results[i],
                        n_clusters,
                        window_sizes,
                        METRICS[i],
                        METRIC_Y_LABELS[i],
                        optimal_index,
                        hand_cluster_results,
                        title=f"{algo_title} {METRICS[i]} for {agent_env_name}",
                        filename=f"{save_folder}/{clustering_algo}-reduced-window-{METRICS[i]}.png",
                    )

                plot_window_all_metrics(
                    clustering_metric_results,
                    METRICS,
                    METRIC_Y_LABELS,
                    n_clusters,
                    window_sizes,
                    2,
                    4,
                    optimal_index,
                    hand_cluster_results,
                    title=f"{algo_title} for {agent_env_name}",
                    filename=f"{save_folder}/{clustering_algo}-window-all.png",
                    figsize=(20, 10),
                )

                reduced_clustering_metric_results = [
                    clustering_metric_results[i] for i in REDUCED_METRIC_INDEXES
                ]
                plot_window_all_metrics(
                    reduced_clustering_metric_results,
                    REDUCED_METRICS,
                    REDUCED_Y_LABELS,
                    n_clusters,
                    window_sizes,
                    2,
                    2,
                    optimal_index,
                    hand_cluster_results,
                    title=f"{algo_title} for {agent_env_name}",
                    filename=f"{save_folder}/{clustering_algo}-reduced-window-all.png",
                    figsize=(12, 8),
                )

    # ===== Agent comparison =====

    clustering_algo = "kmeans-st"
    if not os.path.exists(f"{SAVE_ROOT_FOLDER}/agent-comparison"):
        os.mkdir(f"{SAVE_ROOT_FOLDER}/agent-comparison")
    for _env_name in tqdm(ENV_NAMES):
        if all(
            os.path.exists(
                f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/{clustering_algo}/summary.npz"
            )
            for _agent_name in AGENT_NAMES
        ):
            print(f"Generating agent comparison for {_env_name}")
            agent_clustering_metric_results = []
            agent_optimal_index = []
            agent_hand_cluster_results = []
            for _agent_name in AGENT_NAMES:
                with np.load(
                    f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/{clustering_algo}/summary.npz",
                    allow_pickle=True,
                ) as file:
                    clustering_metric_results = [file[metric] for metric in METRICS]
                    agent_clustering_metric_results.append(clustering_metric_results)
                    n_clusters = file["n_clusters"]
                    window_sizes = file["window_sizes"]
                    agent_optimal_index.append(file["optimal_model"].item()["index"])

            agents_title = " and ".join(AGENT_NAMES)
            plot_comparison_window_all_metrics(
                AGENT_NAMES,
                agent_clustering_metric_results,
                METRICS,
                METRIC_Y_LABELS,
                LINE_STYLES,
                n_clusters,
                window_sizes,
                2,
                4,
                agent_optimal_index,
                agent_hand_cluster_results,
                title=f"{_env_name} comparison for {agents_title}",
                filename=f"{SAVE_ROOT_FOLDER}/agent-comparison/{_env_name}-all-metrics.png",
                figsize=(20, 10),
            )
        else:
            print(
                f"Skipping: {_env_name} - ",
                [
                    os.path.exists(
                        f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/{clustering_algo}/summary.npz"
                    )
                    for _agent_name in AGENT_NAMES
                ],
            )

    # ===== Clustering comparison =====
    for _agent_name, _env_name in tqdm(
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        print(f"Generating clustering comparison of {_agent_name} and {_env_name}")
        dataset_folder = f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}"
        agent_env_name = f'{_agent_name.replace("_", " ")} {_env_name}'
        CLUSTERING_NAMES = ["ST KMeans", "ST KMeans + value"]
        clustering_title = "Spatio-Temporal KMeans with and without State-value"

        cluster_metric_results = []
        cluster_optimal_index = []
        hand_cluster_results = []
        for clustering_algo in ["kmeans-st", "kmeans-st-state-value"]:
            with np.load(
                f"{dataset_folder}/{clustering_algo}/summary.npz", allow_pickle=True
            ) as file:
                clustering_metric_results = [file[metric] for metric in METRICS]
                cluster_metric_results.append(clustering_metric_results)
                n_clusters = file["n_clusters"]
                window_sizes = file["window_sizes"]
                cluster_optimal_index.append(file["optimal_model"].item()["index"])

        plot_comparison_window_all_metrics(
            CLUSTERING_NAMES,
            cluster_metric_results,
            METRICS,
            METRIC_Y_LABELS,
            LINE_STYLES,
            n_clusters,
            window_sizes,
            2,
            4,
            cluster_optimal_index,
            hand_cluster_results,
            title=f"{agent_env_name} comparison of {clustering_title}",
            filename=f"{SAVE_ROOT_FOLDER}{_agent_name}-{_env_name}/clustering-comparison.png",
            figsize=(20, 10),
        )
