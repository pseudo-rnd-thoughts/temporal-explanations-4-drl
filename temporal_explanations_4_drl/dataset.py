from __future__ import annotations

import functools
import json
import os
from collections import namedtuple
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.utils import seeding
from tqdm import tqdm

from temporal_explanations_4_xrl.utils import create_directory

DatasetEpisode = namedtuple("DatasetEpisode", ["length", "start", "end"])


def generate_atari_dataset(
    q_value_policy: Callable[[ObsType], np.ndarray],
    agent_name: str,
    env_name: str,
    save_folder: str,
    dataset_size: int = None,
    dataset_episodes: int = None,
    num_vector_envs: int = 20,
    epsilon: float = 0.01,
    seed: int | None = None,
    env_hyperparameters: dict[str, Any] = None,
) -> tuple[int, int, int]:
    """Generates a training dataset using an agent for an environment, actions are selected randomly
        (but not saved to the dataset) epsilon percent of the time

    Args:
        q_value_policy: The agent policy that uses the observation, returning the q-values
        agent_name: The agent name
        env_name: The atari environment name
        save_folder: The folder where the trajectory is saved to
        dataset_size: The size of the dataset generate
        dataset_episodes: The number of dataset episodes to generate
        num_vector_envs: The number of environments to run in parallel
        epsilon: The probability of a random action being taken, default is 1%
        seed: The seed for the vector environments
        env_hyperparameters: Atari environment parameters

    Returns:
        The number of episodes run, the total number of steps taken and the total number of steps saved
    """
    create_directory(save_folder)
    if env_hyperparameters is None:
        env_hyperparameters = {}

    assert (dataset_size is not None) != (
        dataset_episodes is not None
    ), f"Expects that either dataset size ({dataset_size}) or dataset episodes ({dataset_size}) are not None"

    # Initial the environments and the initial observations
    env = gym.vector.make(
        f"{env_name}NoFrameskip-v4",
        num_envs=num_vector_envs,
        asynchronous=False,
        wrappers=lambda x: gym.wrappers.FrameStack(
            gym.wrappers.AtariPreprocessing(x, noop_max=0), 4
        ),
        render_mode="rgb_array",
        **env_hyperparameters,
    )
    # Pytorch is NCHW and Jax is NHWC, so we need to transform the observation
    env = gym.wrappers.TransformObservation(
        env, lambda observe: np.moveaxis(observe, 1, 3)
    )
    agent_obs, info = env.reset(seed=seed)
    rng, _ = seeding.np_random(seed=seed)

    # Agent obs, q-values (actions), reward, ram obs and human obs for each environment to track transitions
    env_datasets = [([], [], [], [], [], []) for _ in range(num_vector_envs)]

    # Generate a progress bar as the process can take a while depending on the dataset_size
    progress_bar = tqdm(mininterval=5, desc=f"{agent_name}, {env_name}")
    # Save the total steps taken, the number of steps saved so far and
    #   the total number of trajectories / episodes finished
    steps_taken, saved_steps, episodes_run = 0, 0, 0
    while (dataset_size is not None and saved_steps < dataset_size) or (
        dataset_episodes is not None and episodes_run < dataset_episodes
    ):
        # Select the actions for the observations
        q_values = q_value_policy(agent_obs)
        max_actions = np.argmax(q_values, axis=1)
        random_actions = env.action_space.sample()
        actions = np.where(
            rng.random(num_vector_envs) > epsilon, max_actions, random_actions
        )

        assert q_values.shape == (
            num_vector_envs,
            env.single_action_space.n,
        ), f"{q_values.shape} != ({num_vector_envs}, {env.action_space.shape})"
        assert actions.shape == (env.num_envs,), actions.shape

        human_obs = env.call("render")
        ram_obs = [ale.getRAM() for ale in env.call("ale")]
        next_agent_obs, reward, terminated, truncated, info = env.step(actions)
        for env_num in range(num_vector_envs):
            steps_taken += 1

            env_datasets[env_num][0].append(
                np.moveaxis(info["final_observation"][env_num], 0, 2)
                if terminated[env_num] or truncated[env_num]
                else agent_obs[env_num]
            )
            env_datasets[env_num][1].append(q_values[env_num])
            env_datasets[env_num][2].append(actions[env_num])
            env_datasets[env_num][3].append(reward[env_num])
            # For the final observations, these render obs will not be correct.
            env_datasets[env_num][4].append(ram_obs[env_num])
            env_datasets[env_num][5].append(human_obs[env_num])

            # If the environment terminates then save the trajectory
            if terminated[env_num] or truncated[env_num]:
                with open(f"{save_folder}/trajectory-{episodes_run}.npz", "wb") as file:
                    (
                        env_obs,
                        env_q_values,
                        env_actions,
                        env_rewards,
                        env_ram_obs,
                        env_human_obs,
                    ) = env_datasets[env_num]

                    np.savez_compressed(
                        file,
                        agent_obs=np.array(env_obs),
                        q_values=np.array(env_q_values),
                        actions=np.array(env_actions),
                        rewards=np.array(env_rewards),
                        ram_obs=np.array(env_ram_obs),
                        human_obs=np.array(env_human_obs),
                        length=len(env_obs),
                        termination=terminated[env_num],
                        truncation=truncated[env_num],
                        lives=info["lives"][env_num],
                    )

                # update the saved_steps and episodes_runs then reset the environment dataset
                saved_steps += len(env_obs)
                episodes_run += 1
                env_datasets[env_num] = ([], [], [], [], [], [])

        # Update the observations with the new observation
        agent_obs = next_agent_obs

        progress_bar.update(num_vector_envs)
    # Close the progress bar
    progress_bar.close()

    # When the dataset is generated then save the metadata
    with open(f"{save_folder}/metadata.json", "w") as file:
        json.dump(
            {
                "agent-name": agent_name,
                "env-name": env_name,
                "dataset-size": dataset_size,
                "dataset-episodes": dataset_episodes,
                "num-envs": num_vector_envs,
                "env-parameters": env_hyperparameters,
                "episodes-run": episodes_run,
                "steps-taken": steps_taken,
                "saved-steps": saved_steps,
                "seed": seed,
            },
            file,
        )

    # Return the metadata
    return episodes_run, steps_taken, saved_steps


def _load_dataset_prop(
    dataset_folder: str,
    prop: str,
    dtype: str = None,
    num_files: int | None = None,
    size: int | None = None,
) -> np.ndarray:
    """Loads a dataset from dataset folder with property, dtype, num_files to load and dataset size

    :param dataset_folder: The folder to load the dataset from
    :param prop: The property to load
    :param dtype: The dtype from the dataset
    :param num_files: The number of files to load
    :param size: The size of the dataset
    :return: The loaded dataset
    """
    assert not (
        (num_files is not None) and (size is not None)
    ), f"Both {num_files} and {size} are not None, only allow one of the parameters to be not None"
    assert (
        num_files is None or 0 < num_files
    ), f"Number of files ({num_files}) must be None or greater than zero"
    assert size is None or 0 < size, f"Size ({size}) must be None or greater than zero"

    filenames = sorted(
        (
            filename
            for filename in os.listdir(dataset_folder)
            if "trajectory" in filename and ".npz" in filename
        ),
        key=lambda filename: int(
            filename.replace("trajectory-", "").replace(".npz", "")
        ),
    )
    if len(filenames) == 0:
        return np.array([])

    if size is not None:
        dataset_size = 0
        pos = 0
        while dataset_size < size:
            with np.load(
                f"{dataset_folder}/{filenames[pos]}", allow_pickle=True
            ) as file:
                dataset_size += file["length"]
                pos += 1
        filenames = filenames[:pos]
    else:
        if num_files is not None:
            filenames = filenames[slice(num_files)]
        dataset_size = 0
        for filename in filenames:
            with np.load(f"{dataset_folder}/{filename}", allow_pickle=True) as file:
                dataset_size += file["length"]

    with np.load(f"{dataset_folder}/{filenames[0]}", allow_pickle=True) as file:
        data_shape = file[prop].shape
    dataset = np.empty(shape=(dataset_size,) + data_shape[1:], dtype=dtype)

    pos = 0
    for filename in filenames:
        with np.load(f"{dataset_folder}/{filename}") as file:
            length = file["length"]
            dataset[pos : pos + length] = file[prop]
            pos += length

    return dataset


load_obs = functools.partial(_load_dataset_prop, prop="obs")
load_atari_obs = functools.partial(_load_dataset_prop, prop="agent_obs", dtype=np.uint8)
load_atari_ram_obs = functools.partial(
    _load_dataset_prop, prop="ram_obs", dtype=np.uint8
)
load_atari_human_obs = functools.partial(
    _load_dataset_prop, prop="human_obs", dtype=np.uint8
)
load_discrete_actions = functools.partial(
    _load_dataset_prop, prop="actions", dtype=np.int32
)
load_continuous_actions = functools.partial(
    _load_dataset_prop, prop="actions", dtype=np.float32
)
load_q_values = functools.partial(_load_dataset_prop, prop="q_values")
load_rewards = functools.partial(_load_dataset_prop, prop="rewards")


def load_state_values(dataset_folder: str, **kwargs) -> np.ndarray:
    """
    Loads the q-values from the dataset and then normalises the values

    :param dataset_folder: the dataset folders
    :return: normalised q-values for the dataset
    """
    q_value_dataset = load_q_values(dataset_folder, **kwargs)
    return np.max(q_value_dataset, axis=1)


def load_trajectories(dataset_folder: str) -> list[DatasetEpisode]:
    """
    Loads the trajectories with the trajectory size, start and end from the dataset

    :param dataset_folder: the dataset folder
    :return: List of trajectory sizes, the start and end position in the trajectory
    """
    trajectories, pos = [], 0
    filenames = sorted(
        (
            filename
            for filename in os.listdir(dataset_folder)
            if "trajectory" in filename and ".npz" in filename
        ),
        key=lambda filename: int(
            filename.replace("trajectory-", "").replace(".npz", "")
        ),
    )
    for filename in filenames:
        with np.load(f"{dataset_folder}/{filename}", allow_pickle=True) as file:
            length = file["length"]

            trajectories.append(
                DatasetEpisode(length=length, start=pos, end=pos + length)
            )
            pos += length

    return trajectories


def dataset_to_trajectories(
    dataset: np.ndarray, trajectories: list[DatasetEpisode]
) -> list[np.ndarray]:
    """
    Converts the dataset to a list of trajectories

    :param dataset: the dataset with the trajectories concatenated
    :param trajectories: the trajectories from the dataset
    :return: List of trajectories
    """
    return [dataset[trajectory.start : trajectory.end] for trajectory in trajectories]
