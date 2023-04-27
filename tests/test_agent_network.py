import gymnasium as gym
import numpy as np
import pytest

from temporal_explanations_4_xrl.agent_networks import (
    AtariDqnTfNetwork,
    AtariRainbowTfNetwork,
    flax_to_tf_weights,
    load_dopamine_dqn_flax_model,
    load_dopamine_rainbow_flax_model,
)


@pytest.mark.parametrize("env_name", ["Pong"])
def test_flax_tf_compatibility(
    env_name: str, root_network_folder="../models/dopamine/jax"
):
    num_actions = gym.make(f"ALE/{env_name}-v5").action_space.n

    obs = np.random.randint(0, 255, (1, 84, 84, 4), dtype=np.uint8).astype(np.float32)

    model_def, model_params = load_dopamine_dqn_flax_model(
        env_name, root_network_folder
    )
    flax_output = np.array(model_def.apply(model_params, obs))

    tf_model = AtariDqnTfNetwork(num_actions)
    flax_to_tf_weights(model_params, tf_model)
    tf_output = np.array(tf_model(obs))
    assert np.allclose(flax_output, tf_output)

    model_def, model_params = load_dopamine_rainbow_flax_model(
        env_name, root_network_folder
    )
    flax_output = np.array(model_def.apply(model_params, obs))

    tf_model = AtariRainbowTfNetwork(num_actions, 51, np.linspace(-10, 10, 51))
    flax_to_tf_weights(model_params, tf_model)
    tf_output = np.array(tf_model(obs))
    assert np.allclose(flax_output, tf_output)
