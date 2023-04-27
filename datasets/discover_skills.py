import os
from typing import Any

import joblib
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from datasets.generate_datasets import (
    DOPAMINE_DQN_ATARI_AGENT_ENVS,
    DOPAMINE_RAINBOW_ATARI_AGENT_ENVS,
)
from temporal_explanations_4_xrl.agent_networks import (
    load_dopamine_dqn_flax_model,
    load_dopamine_rainbow_flax_model,
)
from temporal_explanations_4_xrl.dataset import (
    load_atari_obs,
    load_discrete_actions,
    load_state_values,
    load_trajectories,
)
from temporal_explanations_4_xrl.graying_the_black_box import (
    GrayingTheBlackBoxKMeans,
    SpatioTemporalKMeans,
    feature_extraction,
    find_optimal_model,
    load_network_features,
    pca_reduce_features,
    run_tsne,
    save_network_features,
)
from temporal_explanations_4_xrl.plan import Plan
from temporal_explanations_4_xrl.skill import (
    discrete_policy_distance,
    skill_labels_to_trajectory_skills,
)
from temporal_explanations_4_xrl.utils import load_embedding


def run_graying_the_black_box_tsne(
    agent_name: str,
    env_name: str,
    dataset_root_folder: str = "",
    model_root_folder: str = "../models",
):
    """Train Graying the Black Box t-SNE for agent_name and env_name."""
    print(
        f"Computing Graying the Black Box t-SNE embeddings for {agent_name=}, {env_name=}"
    )
    if not os.path.exists(
        f"{dataset_root_folder}{agent_name}-{env_name}/embedding/tsne-dense-50.npz"
    ):
        if not os.path.exists(
            f"{dataset_root_folder}{agent_name}-{env_name}/embedding"
        ):
            os.mkdir(f"{dataset_root_folder}{agent_name}-{env_name}/embedding")

        if not os.path.exists(
            f"{dataset_root_folder}{agent_name}-{env_name}/dense-features.npz"
        ):
            print("\tComputing dense features")
            obs = load_atari_obs(
                f"{dataset_root_folder}{agent_name}-{env_name}/trajectories"
            )
            if agent_name == "dqn_adam_mse":
                model_def, model_params = load_dopamine_dqn_flax_model(
                    env_name, f"{model_root_folder}/dopamine/jax"
                )
            elif agent_name == "rainbow":
                model_def, model_params = load_dopamine_rainbow_flax_model(
                    env_name, f"{model_root_folder}/dopamine/jax"
                )
            else:
                raise Exception(f"Unknown agent name: {agent_name}")

            dense_features = feature_extraction(
                obs,
                model_def,
                model_params,
                {},
                "dense",
            )
            save_network_features(
                dense_features,
                f"{dataset_root_folder}{agent_name}-{env_name}/embedding/dense-features.npz",
            )
        else:
            dense_features = load_network_features(
                f"{dataset_root_folder}{agent_name}-{env_name}/embedding/dense-features.npz"
            )

        print("\tRunning t-sne")
        dense_50, _, _ = pca_reduce_features(dense_features)
        run_tsne(
            dense_50,
            f"{dataset_root_folder}{agent_name}-{env_name}/embedding/tsne-dense-50.npz",
        )


def run_spatio_temporal_kmeans(
    embedding: np.ndarray,
    n_clusters: list[int],
    window_sizes: list[int],
    dataset_folder: str,
    save_root_folder: str,
    metadata: dict[str, Any],
    clustering_algorithm=SpatioTemporalKMeans,
):
    """Runs the spatio-temporal kmeans."""
    if not os.path.exists(save_root_folder):
        os.mkdir(save_root_folder)

    actions = load_discrete_actions(dataset_folder)
    trajectories = load_trajectories(dataset_folder)

    plan_entropy = np.zeros((len(n_clusters), len(window_sizes)))
    skill_length_distribution = np.zeros((len(n_clusters), len(window_sizes)))
    skill_alignment = np.zeros((len(n_clusters), len(window_sizes)))
    skill_length = np.zeros((len(n_clusters), len(window_sizes)))
    calinski_harabasz_score = np.zeros((len(n_clusters), len(window_sizes)))
    davies_bouldin_score = np.zeros((len(n_clusters), len(window_sizes)))
    silhouette_score = np.zeros((len(n_clusters), len(window_sizes)))
    intensity_factor = np.zeros((len(n_clusters), len(window_sizes)))

    with tqdm(total=len(n_clusters) * len(window_sizes), desc=save_root_folder) as pbar:
        for i, n_cluster in enumerate(n_clusters):
            for j, window_size in enumerate(window_sizes):
                # Run the clustering
                cluster_model = clustering_algorithm(
                    n_clusters=n_cluster, window_size=window_size
                )
                skill_labels = cluster_model.fit_predict((embedding, trajectories))

                # Make the plan
                trajectory_skills = skill_labels_to_trajectory_skills(
                    skill_labels, actions, trajectories
                )
                assert len(trajectory_skills) == len(trajectories)
                plan = Plan(trajectory_skills)

                # Metrics
                plan_entropy[i, j] = plan.entropy()
                skill_length_distribution[i, j] = plan.length_distribution()
                skill_alignment[i, j] = plan.alignment(
                    penalty=1, policy_distance=discrete_policy_distance
                )
                skill_length[i, j] = plan.lengths()
                calinski_harabasz_score[i, j] = metrics.calinski_harabasz_score(
                    embedding, skill_labels
                )
                davies_bouldin_score[i, j] = metrics.davies_bouldin_score(
                    embedding, skill_labels
                )
                silhouette_score[i, j] = metrics.silhouette_score(
                    embedding, skill_labels
                )
                intensity_factor[i, j] = np.mean(
                    [
                        len(skills) / trajectory.length
                        for skills, trajectory in zip(trajectory_skills, trajectories)
                    ]
                )

                # Save joblib
                joblib.dump(
                    {
                        "clustering_model": cluster_model,
                        "clustering_type": clustering_algorithm.__name__,
                        "n_clusters": n_cluster,
                        "window_size": window_size,
                        "embedding shape": embedding.shape,
                        "skill_labels": skill_labels,
                        "plan": plan,
                        "metadata": metadata,
                        # ====== Metrics =======
                        "plan_entropy": plan_entropy[i, j],
                        "skill_length_distribution": skill_length_distribution[i, j],
                        "skill_alignment": skill_alignment[i, j],
                        "skill_length": skill_length[i, j],
                        "calinski_harabasz_score": calinski_harabasz_score[i, j],
                        "davies_bouldin_score": davies_bouldin_score[i, j],
                        "silhouette_score": silhouette_score[i, j],
                        "intensity_factor": intensity_factor[i, j],
                    },
                    f"{save_root_folder}/n_clusters-{n_cluster}-window_size-{window_size}.joblib",
                )

                pbar.update()

    optimal_index = np.unravel_index(
        find_optimal_model(
            plan_entropy.flatten(),
            # -skill_length_distribution.flatten(),
            -skill_alignment.flatten(),
            -calinski_harabasz_score.flatten(),
            davies_bouldin_score.flatten(),
            -silhouette_score.flatten(),
            # intensity_factor.flatten(),
        ),
        (len(n_clusters), len(window_sizes)),
    )
    np.savez_compressed(
        f"{save_root_folder}/summary.npz",
        n_clusters=n_clusters,
        window_sizes=window_sizes,
        embedding=embedding.shape,
        metadata=metadata,
        # Metric summary
        plan_entropy=plan_entropy,
        skill_length_distribution=skill_length_distribution,
        skill_alignment=skill_alignment,
        skill_length=skill_length,
        calinski_harabasz_score=calinski_harabasz_score,
        davies_bouldin_score=davies_bouldin_score,
        silhouette_score=silhouette_score,
        intensity_factor=intensity_factor,
        # Optimal model based on the metrics
        optimal_model={
            "index": optimal_index,
            "n_clusters": n_clusters[optimal_index[0]],
            "window_size": window_sizes[optimal_index[1]],
        },
    )


DATASET_ROOT_FOLDER = ""

if __name__ == "__main__":
    # Run Graying the black box algorithm for atari
    for _agent_name, _env_name in (
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        print(f"{_agent_name=}, {_env_name=}")
        if not os.path.exists(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/embedding/tsne-dense-50.npz"
        ):
            print("\tComputing t-SNE embedding")
            tsne_embeddings = run_graying_the_black_box_tsne(
                agent_name=_agent_name,
                env_name=_env_name,
                dataset_root_folder=DATASET_ROOT_FOLDER,
            )

        tsne_embeddings = load_embedding(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/embedding/tsne-dense-50.npz"
        )
        print("\tComputing Spatio-Temporal KMeans clusters")
        if not os.path.exists(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/kmeans-st/summary.npz"
        ):
            run_spatio_temporal_kmeans(
                embedding=tsne_embeddings,
                n_clusters=[3, 5, 8, 10, 14, 17],
                window_sizes=[0, 1, 2, 3, 5],
                dataset_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/trajectories",
                save_root_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/kmeans-st/",
                metadata={"agent-name": _agent_name, "env-name": _env_name},
            )

        state_value = load_state_values(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/trajectories"
        )
        state_value_tsne_embedding = np.hstack(
            [tsne_embeddings, state_value.reshape(-1, 1)]
        )
        if not os.path.exists(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/kmeans-st-state-value/summary.npz"
        ):
            run_spatio_temporal_kmeans(
                embedding=tsne_embeddings,
                n_clusters=[3, 5, 8, 10, 14, 17],
                window_sizes=[0, 1, 2, 3, 5],
                dataset_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/trajectories",
                save_root_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/kmeans-st-state-value",
                metadata={"agent-name": _agent_name, "env-name": _env_name},
            )

        if not os.path.exists(
            f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/graying-the-black-box-kmeans/summary.npz"
        ):
            run_spatio_temporal_kmeans(
                embedding=tsne_embeddings,
                n_clusters=[3, 5, 8, 10, 14, 17],
                window_sizes=[0, 1, 2, 3, 5],
                dataset_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/trajectories",
                save_root_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}/graying-the-black-box-kmeans",
                metadata={"agent-name": _agent_name, "env-name": _env_name},
                clustering_algorithm=GrayingTheBlackBoxKMeans,
            )
