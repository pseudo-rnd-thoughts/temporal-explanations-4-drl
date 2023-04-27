import json
import os

import numpy as np
from tqdm import tqdm

from datasets.generate_datasets import (
    DOPAMINE_DQN_ATARI_AGENT_ENVS,
    DOPAMINE_RAINBOW_ATARI_AGENT_ENVS,
)
from temporal_explanations_4_xrl.agent_networks import (
    load_dopamine_dqn_flax_model,
    load_dopamine_rainbow_flax_model,
)
from temporal_explanations_4_xrl.autoencoder import (
    AtariAutoencoder,
    AtariVariationalAutoencoder,
)
from temporal_explanations_4_xrl.dataset import load_atari_obs
from temporal_explanations_4_xrl.graying_the_black_box import (
    feature_extraction,
    load_network_features,
    pca_reduce_features,
    run_tsne,
    save_network_features,
)
from temporal_explanations_4_xrl.utils import load_embedding


def train_graying_the_black_box_tsne(
    agent_name: str,
    env_name: str,
    dataset_root_folder: str = "../datasets/",
    model_root_folder: str = "../models",
):
    """Train Graying the Black Box t-SNE for agent_name and env_name."""
    print(
        f"Computing Graying the Black Box t-SNE embeddings for {agent_name=}, {env_name=}"
    )
    if not os.path.exists(
        f"{dataset_root_folder}/training/{agent_name}-{env_name}/embedding/tsne-dense-50.npz"
    ):
        if not os.path.exists(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding"
        ):
            os.mkdir(f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding")

        if not os.path.exists(
            f"{dataset_root_folder}/training/{agent_name}-{env_name}/dense-features.npz"
        ):
            print("\tComputing dense features")
            obs = load_atari_obs(
                f"{dataset_root_folder}/training/{agent_name}-{env_name}/trajectories"
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
                f"{dataset_root_folder}/training/{agent_name}-{env_name}/embedding/dense-features.npz",
            )
        else:
            dense_features = load_network_features(
                f"{dataset_root_folder}/training/{agent_name}-{env_name}/embedding/dense-features.npz"
            )

        print("\tRunning t-sne")
        dense_50, _, _ = pca_reduce_features(dense_features)
        run_tsne(
            dense_50,
            f"{dataset_root_folder}/training/{agent_name}-{env_name}/embedding/tsne-dense-50.npz",
        )


def train_atari_autoencoder(
    agent_name: str,
    env_name: str,
    latent_dims: int,
    training_epochs: int,
    model_name: str,
    dataset_root_folder: str = "../datasets/",
):
    print(f"Training {agent_name=}, {env_name=} atari autoencoder - {model_name}")
    if not os.path.exists(
        f"{dataset_root_folder}training/{agent_name}-{env_name}/models/{model_name}.index"
    ):
        if not os.path.exists(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/models"
        ):
            os.mkdir(f"{dataset_root_folder}training/{agent_name}-{env_name}/models")

        training_obs = (
            load_atari_obs(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories",
                num_files=10,
            )
            / 255.0
        )
        testing_obs = (
            load_atari_obs(
                f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories",
                num_files=10,
            )
            / 255.0
        )

        model = AtariAutoencoder(latent_dims)
        model.compile(optimizer="adam", loss="mse")

        history = model.fit(
            x=training_obs,
            y=training_obs,
            shuffle=True,
            epochs=training_epochs,
            batch_size=32,
            validation_data=(testing_obs, testing_obs),
        )
        model.save_weights(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/models/{model_name}"
        )

        with open(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories/metadata.json"
        ) as file:
            data = json.load(file)
        training_embedding = np.zeros((data["steps-taken"], model.latent_dims))
        pos = 0
        for file_num in tqdm(
            range(
                len(
                    os.listdir(
                        f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories"
                    )
                )
                - 1
            )
        ):
            with np.load(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories/trajectory-{file_num}.npz"
            ) as file:
                agent_obs = file["agent_obs"]

            for obs in np.array_split(agent_obs, 5):
                training_embedding[pos : pos + len(obs)] = model.encode(obs)
                pos += len(obs)

        with open(
            f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories/metadata.json"
        ) as file:
            data = json.load(file)
        testing_embedding = np.zeros((data["steps-taken"], model.latent_dims))
        pos = 0
        for file_num in tqdm(range(data["dataset-episodes"])):
            with np.load(
                f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories/trajectory-{file_num}.npz"
            ) as file:
                agent_obs = file["agent_obs"] / 255.0

            for obs in np.array_split(agent_obs, 5):
                testing_embedding[pos : pos + len(obs)] = model.encode(obs)
                pos += len(obs)

        np.savez_compressed(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-embedding.npz",
            embedding=training_embedding,
            testing_embedding=testing_embedding,
            fit_history=history.history,
            fit_params=history.params,
            latent_dims=model.latent_dims,
        )
        del model
        del training_obs
        del testing_obs

        if not os.path.exists(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-tsne-embedding.npz"
        ):
            print(
                f"\tComputing t-SNE for {agent_name=} {env_name=} autoencoder embeddings"
            )
            autoencoder_embedding = load_embedding(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-embedding.npz"
            )
            run_tsne(
                autoencoder_embedding,
                f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-tsne-embedding.npz",
            )


def train_atari_variational_autoencoder(
    agent_name: str,
    env_name: str,
    latent_dims: int,
    training_epochs: int,
    model_name: str,
    dataset_root_folder: str = "../datasets/",
):
    print(f"Training {agent_name=}, {env_name=} atari variational autoencoder")
    if not os.path.exists(
        f"{dataset_root_folder}training/{agent_name}-{env_name}/models/{model_name}.index"
    ):
        if not os.path.exists(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/models"
        ):
            os.mkdir(f"{dataset_root_folder}training/{agent_name}-{env_name}/models")

        training_obs = (
            load_atari_obs(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories",
                num_files=15,
            )
            / 255.0
        )
        testing_obs = (
            load_atari_obs(
                f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories",
                num_files=15,
            )
            / 255.0
        )

        model = AtariVariationalAutoencoder(latent_dims)
        model.compile(optimizer="adam")

        history = model.fit(
            training_obs,
            shuffle=True,
            epochs=training_epochs,
            batch_size=32,
            validation_data=(testing_obs,),
        )
        model.save_weights(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/models/{model_name}"
        )

        with open(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories/metadata.json"
        ) as file:
            data = json.load(file)
        training_embedding = np.zeros((data["saved-steps"], model.latent_dims))
        pos = 0
        for file_num in tqdm(
            range(
                len(
                    os.listdir(
                        f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories"
                    )
                )
                - 1
            )
        ):
            with np.load(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/trajectories/trajectory-{file_num}.npz"
            ) as file:
                agent_obs = file["agent_obs"]

            for obs in np.array_split(agent_obs, 5):
                training_embedding[pos : pos + len(obs)] = model.encode(obs)
                pos += len(obs)

        with open(
            f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories/metadata.json"
        ) as file:
            data = json.load(file)
        testing_embedding = np.zeros((data["saved-steps"], model.latent_dims))
        pos = 0
        for file_num in tqdm(range(data["dataset-episodes"])):
            with np.load(
                f"{dataset_root_folder}testing/{agent_name}-{env_name}/trajectories/trajectory-{file_num}.npz"
            ) as file:
                agent_obs = file["agent_obs"] / 255.0

            for obs in np.array_split(agent_obs, 5):
                testing_embedding[pos : pos + len(obs)] = model.encode(obs)
                pos += len(obs)

        np.savez_compressed(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-embedding.npz",
            embedding=training_embedding,
            testing_embedding=testing_embedding,
            fit_history=history.history,
            fit_params=history.params,
            latent_dims=model.latent_dims,
        )
        del model
        del training_obs
        del testing_obs

        if not os.path.exists(
            f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-tsne-embedding.npz"
        ):
            print(
                f"\tComputing t-SNE for {agent_name=} {env_name=} variational autoencoder embeddings"
            )
            autoencoder_embedding = load_embedding(
                f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-embedding.npz"
            )
            run_tsne(
                autoencoder_embedding,
                f"{dataset_root_folder}training/{agent_name}-{env_name}/embedding/{model_name}-tsne-embedding.npz",
            )


if __name__ == "__main__":
    # Graying the black box
    for _agent_name, _env_name in (
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        train_graying_the_black_box_tsne(_agent_name, _env_name)

    # Baseline temporal explanation
    for _agent_name, _env_name in (
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        train_atari_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=16,
            training_epochs=20,
            model_name="autoencoder-model",
        )

    # Exploring different latent dims
    print("Latent Dims")
    for _agent_name, _env_name in DOPAMINE_DQN_ATARI_AGENT_ENVS:
        train_atari_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=8,
            training_epochs=20,
            model_name="autoencoder-8-model",
        )
        train_atari_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=64,
            training_epochs=20,
            model_name="autoencoder-64-model",
        )

    # Baseline temporal explanation
    for _agent_name, _env_name in (
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        train_atari_variational_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=16,
            training_epochs=20,
            model_name="vae-model",
        )

    # Exploring different latent dims
    for _agent_name, _env_name in DOPAMINE_DQN_ATARI_AGENT_ENVS:
        train_atari_variational_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=8,
            training_epochs=20,
            model_name="vae-8-model",
        )
        train_atari_variational_autoencoder(
            agent_name=_agent_name,
            env_name=_env_name,
            latent_dims=64,
            training_epochs=20,
            model_name="vae-64-model",
        )
