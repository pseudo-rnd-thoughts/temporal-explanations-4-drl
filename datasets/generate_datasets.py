import os
from typing import Optional

from jax import vmap

from temporal_explanations_4_drl.agent_networks import (
    load_dopamine_dqn_flax_model,
    load_dopamine_rainbow_flax_model,
)
from temporal_explanations_4_drl.dataset import generate_atari_dataset


def make_atari_dopamine_dqn_dataset(
    agent_name: str,
    env_name: str,
    folder: str,
    dataset_sizes: Optional[int],
    dataset_episodes: Optional[int],
    num_vector_envs: int,
    seed: Optional[int] = None,
    epsilon: float = 0.01,
):
    """Makes atari dopamine dqn dataset for agent name on environment name."""
    print(f"Generating {agent_name=}, {env_name=} dataset")
    if not os.path.exists(
        f"{folder}/{agent_name}-{env_name}/trajectories/trajectory-0.npz"
    ):
        model_def, model_params = load_dopamine_dqn_flax_model(
            env_name, "../models/dopamine/jax"
        )

        @vmap
        def policy(obs):
            return model_def.apply(model_params, obs)

        generate_atari_dataset(
            q_value_policy=policy,
            agent_name=agent_name,
            env_name=env_name,
            save_folder=f"{folder}/{agent_name}-{env_name}/trajectories",
            dataset_size=dataset_sizes,
            dataset_episodes=dataset_episodes,
            num_vector_envs=num_vector_envs,
            seed=seed,
            epsilon=epsilon,
        )


def make_atari_dopamine_rainbow_dataset(
    agent_name: str,
    env_name: str,
    folder: str,
    dataset_sizes: Optional[int],
    dataset_episodes: Optional[int],
    num_vector_envs: int,
    seed: Optional[int] = None,
    epsilon: float = 0.01,
):
    print(f"{agent_name=}, {env_name=}")
    if not os.path.exists(
        f"{folder}/{agent_name}-{env_name}/trajectories/trajectory-0.npz"
    ):
        model_def, model_params = load_dopamine_rainbow_flax_model(
            env_name, "../models/dopamine/jax"
        )

        @vmap
        def policy(obs):
            return model_def.apply(model_params, obs)

        generate_atari_dataset(
            q_value_policy=policy,
            agent_name=agent_name,
            env_name=env_name,
            save_folder=f"{folder}/{agent_name}-{env_name}/trajectories",
            dataset_size=dataset_sizes,
            dataset_episodes=dataset_episodes,
            num_vector_envs=num_vector_envs,
            seed=seed,
            epsilon=epsilon,
        )


SEED = 123
EPSILON = 0.05  # 5% probability of random actions
ENV_NAMES = [
    "Asterix",
    "Breakout",
    "Pong",
    "Qbert",
    "Seaquest",
    "SpaceInvaders",
]
AGENT_NAMES = ["dqn_adam_mse", "rainbow"]
DOPAMINE_DQN_ATARI_AGENT_ENVS = [("dqn_adam_mse", env_name) for env_name in ENV_NAMES]
DOPAMINE_RAINBOW_ATARI_AGENT_ENVS = [("rainbow", env_name) for env_name in ENV_NAMES]


if __name__ == "__main__":
    # Atari Dqn
    for _agent_name, _env_name in DOPAMINE_DQN_ATARI_AGENT_ENVS:
        make_atari_dopamine_dqn_dataset(
            _agent_name,
            _env_name,
            folder="",
            dataset_sizes=60_000,
            dataset_episodes=None,
            num_vector_envs=16,
            seed=SEED,
            epsilon=EPSILON,
        )

    # Atari Rainbow
    for _agent_name, _env_name in DOPAMINE_RAINBOW_ATARI_AGENT_ENVS:
        make_atari_dopamine_rainbow_dataset(
            _agent_name,
            _env_name,
            folder="",
            dataset_sizes=60_000,
            dataset_episodes=None,
            num_vector_envs=16,
            seed=SEED,
            epsilon=EPSILON,
        )

    # Box2D
    # for _agent_name, _env_name in sb3_box2d_agent_envs:
    #     make_box2d_sb3_dataset(
    #         _agent_name, _env_name,
    #         folder="",
    #         dataset_size=20_000,
    #         dataset_episodes=None,
    #         num_vector_envs=8,
    #         seed=SEED
    #     )
    #
    # for _agent_name, _env_name in sb3_mujoco_agent_envs:
    #     make_mujoco_sb3_dataset(
    #         _agent_name, _env_name,
    #         folder="",
    #         dataset_size=40_000,
    #         dataset_episodes=None,
    #         num_vector_envs=12,
    #         seed=SEED
    #     )
