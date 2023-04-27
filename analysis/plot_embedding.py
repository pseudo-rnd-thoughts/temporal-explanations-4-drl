import os

import matplotlib.pyplot as plt
import numpy as np

envs = ["Breakout", "MsPacman", "Pong", "Riverraid", "Seaquest", "Qbert"]

# Autoencoder loss figure
print("Autoencoder loss")
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
for env, ax in zip(envs, axs.flatten()):
    ax.set_title(env)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error")
    for agent in ["dqn_adam_mse", "rainbow"]:
        if os.path.exists(
            f"../datasets/training/{agent}-{env}/embedding/autoencoder-model-embedding.npz"
        ):
            with np.load(
                f"../datasets/training/{agent}-{env}/embedding/autoencoder-model-embedding.npz",
                allow_pickle=True,
            ) as file:
                training = file["fit_history"].item()["loss"]
                testing = file["fit_history"].item()["val_loss"]

                epochs = int(file["fit_params"].item()["epochs"])

            ax.plot(np.arange(epochs), training, linestyle="solid", label=agent)
            ax.plot(np.arange(epochs), testing, linestyle="dashed", label=agent)
            ax.legend()
plt.tight_layout()
plt.savefig("figs/autoencoder-training.png")
plt.close()

print("Variational Autoencoder")
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
for env, ax in zip(envs, axs.flatten()):
    ax.set_title(env)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error")
    for agent in ["dqn_adam_mse"]:  # + rainbow
        if os.path.exists(
            f"../datasets/training/{agent}-{env}/embedding/vae-model-embedding.npz"
        ):
            with np.load(
                f"../datasets/training/{agent}-{env}/embedding/vae-model-embedding.npz",
                allow_pickle=True,
            ) as file:
                history = file["fit_history"].item()

                training_loss = history["loss"]
                training_reconstruction = history["reconstruction_loss"]
                training_kl = history["kl_loss"]

                testing_loss = history["val_loss"]
                testing_reconstruction = history["val_reconstruction_loss"]
                testing_kl = history["val_kl_loss"]

                epochs = int(file["fit_params"].item()["epochs"])

            ax.plot(np.arange(epochs), training_loss, linestyle="solid", label="Total")
            ax.plot(
                np.arange(epochs),
                training_reconstruction,
                linestyle="solid",
                label="Reconstruction",
            )
            ax.plot(np.arange(epochs), training_kl, linestyle="solid", label="KL")

            ax.plot(np.arange(epochs), testing_loss, linestyle="dashed", label="Total")
            ax.plot(
                np.arange(epochs),
                testing_reconstruction,
                linestyle="dashed",
                label="Reconstruction",
            )
            ax.plot(np.arange(epochs), testing_kl, linestyle="dashed", label="KL")
            ax.legend()
plt.tight_layout()
plt.savefig("figs/vae-training.png")
plt.close()

# Cluster Algorithm
for folder in os.listdir("../datasets/v1/training"):
    if not os.path.exists(f"figs/{folder}"):
        os.mkdir(f"figs/{folder}")
    if os.path.exists(
        f"../datasets/training/{folder}/graying-the-black-box/st-kmeans-results.npz"
    ):
        print(folder)
        for metric in ["entropy", "distribution", "alignment"]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(folder)
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel(metric)
            for algo, name in [
                [
                    "graying-the-black-box/st-kmeans-results.npz",
                    "Graying the black Box",
                ],
                [
                    "baseline-clustering/st-kmeans-results.npz",
                    "Spatio-Temporal KMeans",
                ],
                ["baseline-clustering/kmeans-results.npz", "KMeans"],
            ]:
                with np.load(f"../datasets/v1/training/{folder}/{algo}") as file:
                    data = file[metric]
                    num_clusters = file["num_clusters"].astype(np.int32)
                    if "window_sizes" in file:
                        data = np.min(data, axis=1)
                    assert data.shape == (
                        len(num_clusters),
                    ), f"{data.shape}, {num_clusters}"

                ax.plot(num_clusters, data, label=name)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"figs/{folder}/cluster-{metric}.png")
            plt.close()

# ST Cluster Algorithm
print("Spatio-Temporal Clustering Algorithm")
for folder in os.listdir("../datasets/v1/training"):
    if not os.path.exists(f"figs/{folder}"):
        os.mkdir(f"figs/{folder}")
    if os.path.exists(
        f"../datasets/v1/training/{folder}/graying-the-black-box/st-kmeans-results.npz"
    ):
        print(folder)
        for algo, name in [
            ["graying-the-black-box/st-kmeans-results.npz", "Graying the black Box"],
            [
                "baseline-clustering/st-kmeans-results.npz",
                "Baseline Temporal Explanation",
            ],
        ]:
            for metric in ["entropy", "distribution"]:  # alignment
                with np.load(f"../datasets/v1/training/{folder}/{algo}") as file:
                    data = file[metric]
                    num_clusters = file["num_clusters"]
                    window_sizes = file["window_sizes"]

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_title(f"{name} - Spatio Temporal KMeans")
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel(metric)
                for pos in range(len(window_sizes)):
                    ax.plot(
                        num_clusters,
                        data[:, pos],
                        label=f"Window size={window_sizes[pos]}",
                    )
                ax.legend()

                plt.tight_layout()
                plt.savefig(
                    f"figs/{folder}/st-kmeans-{name.replace(' ', '-').lower()}-{metric}.png"
                )
                plt.close()
