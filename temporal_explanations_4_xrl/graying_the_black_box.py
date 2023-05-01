"""
Implementation of kmeans using a spatio-temporal distances function
"""
from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp
from flax.core import FrozenDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from temporal_explanations_4_xrl.dataset import DatasetEpisode

try:
    from cuml import TSNE
except ImportError:
    from sklearn.manifold import TSNE


def feature_extraction(
    dataset: onp.ndarray,
    network_def: nn.Module,
    network_params: FrozenDict,
    network_args: dict[str, Any],
    network_feature: str,
    split_size: int = 1000,
) -> onp.ndarray:
    """Selects the features from the observation dataset using the network_features parameter

    Args:
        dataset: The dataset folder with all the trajectory files
        network_def: The network definition
        network_params: The network parameters
        network_args: The network apply arguments
        network_feature: The network feature to select
        split_size: The dataset split size

    Returns:
        A numpy array for the network features given an observation dataset
    """

    @jax.vmap
    def _vectorise_network(state: jnp.ndarray):
        """Using the mutable='intermediates' tag then intermediate variables saved for access.

        Args:
            state: the input to the networks

        Returns:
            the network output and the inner state of the network for saved variables
        """
        return network_def.apply(
            network_params, state, mutable="intermediates", **network_args
        )

    features_dataset = []
    for dataset_split in tqdm(onp.array_split(dataset, len(dataset) // split_size + 1)):
        network_output, network_state = _vectorise_network(dataset_split)
        features_dataset.append(network_state["intermediates"][network_feature][0])

    return onp.concatenate(features_dataset)


def save_network_features(network_features_dataset: onp.ndarray, filename: str):
    """Saves the network features

    Args:
        network_features_dataset: An array of network features for a dataset
        filename: The filename of the npz to save the network features
    """
    onp.savez_compressed(filename, network_features=network_features_dataset)


def load_network_features(filename: str) -> onp.ndarray:
    """Loads the network features

    Args:
        filename: The filename from which to load the network features

    Returns:
        A numpy array of network features
    """
    with onp.load(filename, allow_pickle=True) as file_data:
        return file_data["network_features"]


def pca_reduce_features(
    feature_dataset: onp.ndarray, components: int | float = 50
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Reduces the features using PCA to find the most 'important' features to the agent

    Args:
        feature_dataset: the feature dataset that has not been reduced yet
        components: The number of components to reduce the dataset to

    Returns:
        A tuple containing the reduced dataset and the singular values for each of the components
    """
    pca = PCA(n_components=components)
    reduced_features = pca.fit_transform(feature_dataset)

    return reduced_features, pca.explained_variance_, pca.singular_values_


def run_tsne(
    feature_dataset: onp.ndarray,
    save_path: str = None,
    verbose: int = 0,
    dimensions: int = 2,
    n_iter: int = 1000,
    perplexity: float = 30.0,
    **tsne_kwargs,
) -> tuple[onp.ndarray, float]:
    """Using t-SNE, we can plot the dataset in two-dimensions allowing easy visualisation of the aggregated states

    Args:
        feature_dataset: The dataset that is reduced to two-dimensions. As T-SNE is computationally demanding
            (N^2 or NlogN), the dataset should have reduced components, often done with PCA
        save_path: The location of where to save the result as to not have to recompute the result
        verbose: A T-SNE parameter of if to log the progress of the algorithm
        dimensions: The number of dimensions to project into
        n_iter: The number of iterations that the T-SNE algorithm will take to converge to a solution
        perplexity: Parameter of T-SNE
        **tsne_kwargs: Kwargs for t-SNE

    Returns:
        A tuple of the transformed dataset and the T-SNE KL divergence error value
    """
    if feature_dataset.shape[1] > 50:
        print(f"WARNING: more than 50 features, {feature_dataset.shape}")
    t_sne = TSNE(
        n_components=dimensions,
        verbose=verbose,
        n_iter=n_iter,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        **tsne_kwargs,
    )
    t_sne_dataset = t_sne.fit_transform(feature_dataset)

    if save_path:
        onp.savez_compressed(
            save_path,
            embedding=t_sne_dataset,
            metadata={
                "type": "t-sne",
                "n_components": dimensions,
                "verbose": verbose,
                "n_iter": n_iter,
                "perplexity": perplexity,
                "init": "pca",
                "learning_rate": "auto",
                "feature_dataset_shape": feature_dataset.shape,
                "kwargs": tsne_kwargs,
            },
        )

    return t_sne_dataset, t_sne.kl_divergence_


def compute_embedding_window_indexes(
    trajectories: list[DatasetEpisode], window_size: int
) -> onp.ndarray:
    """Computes the embedding window indexes for a list of trajectories and a window size."""
    return onp.vstack(
        [
            trajectory.start
            + onp.clip(
                onp.arange(-window_size, window_size + 1)[None, :]
                + onp.arange(trajectory.length)[:, None],
                0,
                trajectory.length - 1,
            )
            for trajectory in trajectories
        ]
    )


class SpatioTemporalKMeans(KMeans):
    """An alternative SpatioTemporal KMeans algorithm compared to Graying the Black Box that is compatible with euclidean distance."""

    def __init__(self, n_clusters: int, window_size: int, **kwargs):
        super().__init__(n_clusters, **kwargs)
        assert isinstance(window_size, int) and window_size >= 0, window_size
        self.window_size = window_size

    def fit(self, X, y=None, sample_weight=None):
        assert isinstance(X, tuple) and len(X) == 2, X
        embeddings, trajectories = X
        assert isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2
        length, features = embeddings.shape
        assert isinstance(trajectories, (list, tuple)) and all(
            isinstance(traj, DatasetEpisode) for traj in trajectories
        )
        assert len(embeddings) == sum(traj.length for traj in trajectories)

        # Compute the window indexes for the embeddings and trajectories
        #   For embeddings at the beginning and end of trajectories, we clip these values to 0 and the trajectory length - 1
        window_indexes = compute_embedding_window_indexes(
            trajectories, self.window_size
        )

        # Compute the temporal embeddings
        temporal_embeddings = embeddings[window_indexes]
        assert temporal_embeddings.shape == (length, self.window_size * 2 + 1, features)
        flattened_temporal_embeddings = temporal_embeddings.reshape(length, -1)

        # Run the standard KMeans algorithm with the new temporal embeddings
        return super().fit(flattened_temporal_embeddings)

    def predict(self, X, sample_weight=None):
        """Enable prediction for both individual embeddings and full trajectories."""
        if isinstance(X, tuple):
            embeddings, trajectories = X
            assert isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2
            assert isinstance(trajectories, list) and all(
                isinstance(traj, DatasetEpisode) for traj in trajectories
            )
            assert len(embeddings) == sum(traj.length for traj in trajectories)

            window_indexes = compute_embedding_window_indexes(
                trajectories, self.window_size
            )
            window_embeddings = embeddings[window_indexes]
            return super().predict(window_embeddings)
        else:
            embedding = X
            assert isinstance(embedding, np.ndarray) and len(embedding.shape) == 1

            window_embedding = np.stack(
                [embedding for _ in range(2 * self.window_size + 1)]
            )
            assert (
                window_embedding.shape == (2 * self.window_size + 1,) + embedding.shape
            )
            return super().predict(X, sample_weight)


class GrayingTheBlackBoxKMeans(KMeans):
    """Not recommended to use, the clusters found are weird, unclear why."""

    def __init__(self, n_clusters: int, window_size: int):
        super().__init__(n_clusters)
        assert isinstance(window_size, int) and 0 <= window_size
        self.window_size = window_size
        self.cluster_centers_ = np.zeros(self.n_clusters)

        self.initial_cluster_centers = 0
        self.steps_taken = 0

    def fit_predict(self, X, y=None, sample_weight=None):
        assert isinstance(X, tuple) and len(X) == 2
        embeddings, trajectories = X
        assert isinstance(embeddings, np.ndarray) and embeddings.ndim == 2
        length, n_features = embeddings.shape
        assert isinstance(trajectories, list) and all(
            isinstance(traj, DatasetEpisode) for traj in trajectories
        )
        assert sum(traj.length for traj in trajectories) == len(embeddings)

        window_indexes = compute_embedding_window_indexes(
            trajectories, self.window_size
        )
        embedding_window = embeddings[window_indexes]
        padded_embedding_window = embedding_window[:, :, None, :]

        del window_indexes, embedding_window

        # initialise the cluster centers
        if self.init == "random":
            self.cluster_centers_ = embeddings[
                np.random.choice(len(embeddings), self.n_clusters)
            ]
        elif self.init == "k-means++":
            self.cluster_centers_ = np.expand_dims(
                embeddings[np.random.choice(len(embeddings))], axis=0
            )
            for i in range(self.n_clusters - 1):
                embedding_cluster_dist = np.linalg.norm(
                    padded_embedding_window - self.cluster_centers_, axis=(1, 3)
                )
                assert embedding_cluster_dist.shape == (
                    length,
                    len(self.cluster_centers_),
                ), embedding_cluster_dist.shape
                squared_cluster_dist = np.square(np.min(embedding_cluster_dist, axis=1))
                assert squared_cluster_dist.shape == (length,)

                new_cluster = embeddings[
                    np.random.choice(
                        len(embeddings),
                        p=squared_cluster_dist / np.sum(squared_cluster_dist),
                    )
                ]
                self.cluster_centers_ = np.vstack([self.cluster_centers_, new_cluster])
        else:
            raise AttributeError(f"unknown cluster initialisation: {self.init}")
        assert self.cluster_centers_.shape == (self.n_clusters, n_features)
        self.initial_cluster_centers = self.cluster_centers_.copy()

        # find the cluster centers
        cluster_labels = np.zeros(length)
        for steps_taken in range(self.max_iter):
            embedding_cluster_dist = np.linalg.norm(
                padded_embedding_window - self.cluster_centers_, axis=(1, 3)
            )
            assert embedding_cluster_dist.shape == (
                length,
                self.n_clusters,
            ), embedding_cluster_dist.shape

            cluster_labels = np.argmin(embedding_cluster_dist, axis=1)
            assert cluster_labels.shape == (length,)

            # Update cluster centers
            new_cluster_centers = np.empty_like(self.cluster_centers_)
            for i in range(self.n_clusters):
                cluster_embeddings = embeddings[cluster_labels == i]
                if len(cluster_embeddings) == 0:
                    min_dist = np.min(embedding_cluster_dist, axis=1)
                    assert min_dist.shape == (length,)
                    squared_cluster_dist = np.square(min_dist)
                    new_cluster_centers[i] = embeddings[
                        np.random.choice(
                            length,
                            p=squared_cluster_dist / np.sum(squared_cluster_dist),
                        )
                    ]
                else:
                    new_cluster_centers[i] = np.mean(cluster_embeddings)
            assert new_cluster_centers.shape == (self.n_clusters, n_features)

            # Check if the new cluster centers are close to the old cluster centers
            if np.allclose(self.cluster_centers_, new_cluster_centers, atol=self.tol):
                self.cluster_centers_ = new_cluster_centers
                self.steps_taken = steps_taken
                return cluster_labels
            else:
                self.cluster_centers_ = new_cluster_centers
        self.steps_taken = self.max_iter

        # return the embedding labels
        return cluster_labels


def find_optimal_model(*metrics: np.ndarray) -> int:
    """Find the optimal model given a list of metrics.

    These will be ordered in ascending order, lowest to highest.
    Therefore, models will be the lowest values will be viewed as best
    """
    assert all(
        isinstance(metric, np.ndarray) and metric.ndim == 1 for metric in metrics
    )

    ordered_metrics = [np.argsort(metric) for metric in metrics]
    index = 1
    while (
        optimal_index := set.intersection(
            *[set(metric[:index]) for metric in ordered_metrics]
        )
    ) == set():
        index += 1
    return optimal_index.pop()
