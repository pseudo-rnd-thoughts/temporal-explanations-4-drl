"""
Implementation of several network architectures
"""
from __future__ import annotations

import pickle

import flax.linen as nn
import flax.training.checkpoints
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import numpy as onp
from flax.core import FrozenDict
from flax.training import checkpoints


class AtariDqnFlaxNetwork(nn.Module):
    """Implementation of a Deep Q Network for Atari Learning environment"""

    num_actions: int
    preprocess_obs: bool = True

    def setup(self):
        initialiser = nn.initializers.xavier_uniform()

        self.Conv_0 = nn.Conv(
            features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initialiser
        )
        self.Conv_1 = nn.Conv(
            features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initialiser
        )
        self.Conv_2 = nn.Conv(
            features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initialiser
        )
        self.Dense_0 = nn.Dense(features=512, kernel_init=initialiser)
        self.Dense_1 = nn.Dense(features=self.num_actions, kernel_init=initialiser)

    @nn.compact
    def __call__(self, obs: onp.ndarray):
        features_obs = self.features(obs)
        # self.sow("intermediates", "conv", features_obs)

        return self.q_network(features_obs)

    def features(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        for conv in (self.Conv_0, self.Conv_1, self.Conv_2):
            obs = nn.relu(conv(obs))
        return obs

    def q_network(self, features):
        features = features.reshape(-1)
        hidden = nn.relu(self.Dense_0(features))
        self.sow("intermediates", "dense", hidden)

        q_values = self.Dense_1(hidden)
        return q_values

    def feature_0(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        return nn.relu(self.Conv_0(obs))

    def q_network_0(self, features):
        for conv in (self.Conv_1, self.Conv_2):
            features = nn.relu(conv(features))

        features = features.reshape(-1)
        hidden = nn.relu(self.Dense_0(features))
        return self.Dense_1(hidden)

    def feature_1(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        for conv in (self.Conv_0, self.Conv_1):
            obs = nn.relu(conv(obs))
        return obs

    def q_network_1(self, features):
        features = nn.relu(self.Conv_2(features))

        features = features.reshape(-1)
        hidden = nn.relu(self.Dense_0(features))
        return self.Dense_1(hidden)

    def feature_2(self, obs):
        return self.features(obs)

    def q_network_2(self, features):
        return self.q_network(features)


class AtariRainbowFlaxNetwork(nn.Module):
    """Implementation of a Rainbow Deep Q Network for Atari Learning environment"""

    num_actions: int
    num_atoms: int
    supports: jnp.ndarray
    preprocess_obs: bool = True

    def setup(self):
        initialiser = nn.initializers.xavier_uniform()

        self.Conv_0 = nn.Conv(
            features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initialiser
        )
        self.Conv_1 = nn.Conv(
            features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initialiser
        )
        self.Conv_2 = nn.Conv(
            features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initialiser
        )
        self.Dense_0 = nn.Dense(features=512, kernel_init=initialiser)
        self.Dense_1 = nn.Dense(
            self.num_actions * self.num_atoms, kernel_init=initialiser
        )

    @nn.compact
    def __call__(self, obs: onp.ndarray):
        features_obs = self.features(obs)
        # self.sow("intermediates", "conv", features_obs)

        return self.q_network(features_obs)

    def features(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        for conv in (self.Conv_0, self.Conv_1, self.Conv_2):
            obs = conv(obs)
            obs = nn.relu(obs)
        return obs

    def q_network(self, features):
        features = features.reshape(-1)
        hidden = self.Dense_0(features)
        hidden = nn.relu(hidden)
        self.sow("intermediates", "dense", hidden)

        logits = self.Dense_1(hidden).reshape((self.num_actions, self.num_atoms))
        probabilities = nn.softmax(logits)
        q_values = jnp.sum(self.supports * probabilities, axis=1)

        return q_values

    def feature_0(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        return nn.relu(self.Conv_0(obs))

    def q_network_0(self, features):
        for conv in (self.Conv_1, self.Conv_2):
            features = nn.relu(conv(features))

        features = features.reshape(-1)

        hidden = nn.relu(self.Dense_0(features))
        logits = self.Dense_1(hidden).reshape((self.num_actions, self.num_atoms))
        return jnp.sum(self.supports * nn.softmax(logits), axis=1)

    def feature_1(self, obs):
        if self.preprocess_obs:
            obs = obs.astype(jnp.float32) / 255.0

        for conv in (self.Conv_0, self.Conv_1):
            obs = nn.relu(conv(obs))
        return obs

    def q_network_1(self, features):
        features = nn.relu(self.Conv_2(features))
        features = features.reshape(-1)

        hidden = nn.relu(self.Dense_0(features))
        logits = self.Dense_1(hidden).reshape((self.num_actions, self.num_atoms))
        return jnp.sum(self.supports * nn.softmax(logits), axis=1)

    def feature_2(self, obs):
        return self.features(obs)

    def q_network_2(self, features):
        return self.q_network(features)


def load_dopamine_dqn_flax_model(
    env_name: str, model_root_folder: str, model_name: str = "dqn_adam_mse"
) -> tuple[AtariDqnFlaxNetwork, FrozenDict]:
    """Loads the dopamine dqn flax model and returns the model definition and params."""
    env = gym.make(f"ALE/{env_name}-v5")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    model_def = AtariDqnFlaxNetwork(num_actions=env.action_space.n)

    with open(f"{model_root_folder}/{model_name}/{env_name}/1/ckpt.199", "rb") as file:
        param_data = pickle.load(file)
        model_params = flax.core.FrozenDict(
            {"params": checkpoints.convert_pre_linen(param_data["online_params"])}
        )
    return model_def, model_params


def load_dopamine_rainbow_flax_model(
    env_name: str, model_root_folder: str, model_name: str = "rainbow"
) -> tuple[AtariRainbowFlaxNetwork, FrozenDict]:
    """Loads the dopamine rainbow flax model and returns the model definition and params."""
    env = gym.make(f"ALE/{env_name}-v5")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    model_def = AtariRainbowFlaxNetwork(
        num_actions=env.action_space.n, num_atoms=51, supports=np.linspace(-10, 10, 51)
    )
    with open(f"{model_root_folder}/{model_name}/{env_name}/1/ckpt.199", "rb") as file:
        param_data = pickle.load(file)
        model_params = flax.core.FrozenDict(
            {"params": checkpoints.convert_pre_linen(param_data["online_params"])}
        )
    return model_def, model_params
