"""
Functions for explanation mechanisms
- Dataset explanation: Implementation of the dataset similarity explanation from our work
- Skill explanation: Implementation of the skill based explanation from our work
- Plan explanation: Implementation of the plan + skill-based explanation from our work
- Grad-CAM: Implementation of Selvaraju et al., 2016 "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Perturbation-based Saliency Map: Implementation of Greyanus et al., 2018 "Visualizing and Understanding Atari Agents"
"""
from __future__ import annotations

import textwrap

import cv2
import flax.core
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy as onp
from flax import linen as nn
from flax.core import FrozenDict
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from temporal_explanations_4_xrl.plan import Plan
from temporal_explanations_4_xrl.skill import SkillInstance

FONT_SIZE = 18


__all__ = [
    "generate_obs_to_explain",
    "generate_dataset_explanation",
    "generate_skill_explanation",
    "generate_plan_explanation",
    "generate_atari_perturbation_saliency_explanation",
    "generate_atari_grad_cam_explanation",
    "save_explanation",
    "save_observation_with_explanation",
    "save_observation_with_two_explanations",
]


def generate_obs_to_explain(
    q_values: onp.ndarray,
    num_explanations: int,
    window: int = 10,
) -> onp.ndarray:
    """Explains the observation to explain through the q-value deviation.

    Args:
        q_values: The q-values for the dataset
        num_explanations: The number of explanations to make
        window: To prevent generating observation that are temporally very close, we have a blocking window around each index found.

    Returns:
        The dataset indexes of the observations to explain
    """
    length, n_actions = q_values.shape
    valid_indexes = onp.ones(length, dtype=onp.bool_)

    q_value_variance = onp.std(q_values, axis=1)
    assert q_value_variance.shape == (length,)
    explain_indexes = onp.zeros(num_explanations, dtype=onp.int32)
    for i in range(num_explanations):
        index = onp.argmax(onp.where(valid_indexes, q_value_variance, 0.0))
        explain_indexes[i] = index

        valid_indexes[
            onp.clip(onp.arange(index - window, index + window + 1), 0, length)
        ] = False

    return explain_indexes


def save_explanation(
    explanation: onp.ndarray | tuple[onp.ndarray, str],
    filename: str,
    fps: int = 10,
):
    """Saves an explanation using a filename and frames per second

    Args:
        explanation: The explanation for an observation
        filename: The save filename
        fps: The frames per second
    """
    if isinstance(explanation, tuple):
        explanation, expert_knowledge = explanation
        assert isinstance(expert_knowledge, str)
    else:
        expert_knowledge = None
    assert (
            isinstance(explanation, onp.ndarray)
            and explanation.ndim == 4
            and explanation.shape[-1] == 3
    )

    fig, axs = plt.subplots(figsize=(5, 5))
    axs.set_title("Explanation", fontsize=FONT_SIZE)
    explanation_plot = axs.imshow(explanation[0])
    if expert_knowledge is not None:
        axs.text(
            79,
            215,
            "\n".join(textwrap.wrap(expert_knowledge, width=19)),
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="top",
        )
    axs.axis("off")
    plt.tight_layout()

    if explanation.shape[0] == 1:
        plt.savefig(f"{filename}.png")
    else:

        def _update_plot(time_step):
            explanation_plot.set_data(explanation[time_step])
            return (explanation_plot,)

        animation = FuncAnimation(fig, _update_plot, frames=len(explanation), blit=True)
        animation.save(f"{filename}.mp4", fps=fps)

    plt.close()


def save_observation_with_explanation(
    obs: onp.ndarray,
    explanation: onp.ndarray | tuple[onp.ndarray, str],
    filename: str,
    fps: int = 10,
):
    """Saves an individual explanation using the obs, explanation and filename

    Args:
        obs: The agent observation to explain
        explanation: The explanation for the observation
        filename: The save filename
        fps: The frames per second
    """
    assert isinstance(obs, onp.ndarray) and obs.ndim == 3 and obs.shape[-1] == 3
    if isinstance(explanation, tuple):
        explanation, expert_knowledge = explanation
        assert isinstance(expert_knowledge, str)
    else:
        expert_knowledge = None
    assert (
        isinstance(explanation, onp.ndarray)
        and explanation.ndim == 4
        and explanation.shape[-1] == 3
    )

    fig, axs = plt.subplots(ncols=2, figsize=(5.5, 4))
    axs[0].set_title("Observation", fontsize=FONT_SIZE)
    axs[1].set_title("Explanation", fontsize=FONT_SIZE)
    axs[0].imshow(obs)
    explanation_plot = axs[1].imshow(explanation[0])
    if expert_knowledge is not None:
        axs[1].text(
            79,
            215,
            "\n".join(textwrap.wrap(expert_knowledge, width=19)),
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="top",
        )
    axs[0].axis("off")
    axs[1].axis("off")

    if explanation.shape[0] == 1:
        plt.savefig(f"{filename}.png")
    else:

        def _update_plot(time_step):
            explanation_plot.set_data(explanation[time_step])
            return (explanation_plot,)

        animation = FuncAnimation(fig, _update_plot, frames=len(explanation), blit=True)
        animation.save(f"{filename}.mp4", fps=fps)

    plt.close()


def save_observation_with_two_explanations(
    obs: onp.ndarray,
    explanation_1: onp.ndarray | tuple[onp.ndarray, str],
    explanation_2: onp.ndarray | tuple[onp.ndarray, str],
    filename: str,
    fps: int = 10,
):
    """Save contrastive explanation for the obs, explanation 1 and 2 with a filename

    Args:
        obs: The observation to explain
        explanation_1: The first anonymous explanation for the observation
        explanation_2: The second anonymous explanation for the observation
        filename: The save filename
        fps: The frames per second
    """
    expert_knowledge_1, expert_knowledge_2 = None, None
    assert isinstance(obs, onp.ndarray) and obs.ndim == 3 and obs.shape[-1] == 3
    if isinstance(explanation_1, tuple):
        explanation_1, expert_knowledge_1 = explanation_1
        assert isinstance(expert_knowledge_1, str)
    assert (
        isinstance(explanation_1, onp.ndarray)
        and explanation_1.ndim == 4
        and explanation_1.shape[-1] == 3
    ), f"{type(explanation_1)=}, {explanation_1.shape=}, {filename=}"
    if isinstance(explanation_2, tuple):
        explanation_2, expert_knowledge_2 = explanation_2
        assert isinstance(expert_knowledge_2, str)
    assert (
        isinstance(explanation_2, onp.ndarray)
        and explanation_2.ndim == 4
        and explanation_2.shape[-1] == 3
    ), f"{type(explanation_2)=}, {explanation_2.shape=}, {filename=}"

    fig, axs = plt.subplots(ncols=3, figsize=(7, 4))

    axs[0].set_title("Observation", fontsize=FONT_SIZE)
    axs[0].imshow(obs)
    axs[0].axis("off")

    axs[1].set_title("Explanation 1", fontsize=FONT_SIZE)
    explanation_1_plot = axs[1].imshow(explanation_1[0])
    if expert_knowledge_1 is not None:
        axs[1].text(
            79,
            215,
            "\n".join(textwrap.wrap(expert_knowledge_1, width=19)),
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="top",
        )
    axs[1].axis("off")

    axs[2].set_title("Explanation 2", fontsize=FONT_SIZE)
    explanation_2_plot = axs[2].imshow(explanation_2[0])
    if expert_knowledge_2 is not None:
        axs[2].text(
            79,
            215,
            "\n".join(textwrap.wrap(expert_knowledge_2, width=19)),
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="top",
        )
    axs[2].axis("off")

    if explanation_1.shape[0] == 1 and explanation_2.shape[0] == 1:
        plt.savefig(f"{filename}.png")
    else:

        def _update_plot(time_step):
            if time_step < len(explanation_1):
                explanation_1_plot.set_data(explanation_1[time_step])
            if time_step < len(explanation_2):
                explanation_2_plot.set_data(explanation_2[time_step])
            return explanation_1_plot, explanation_2_plot

        animation = FuncAnimation(
            fig,
            _update_plot,
            frames=max(len(explanation_1), len(explanation_2)),
            blit=True,
        )
        animation.save(f"{filename}.mp4", fps=fps)

    plt.close()


def generate_dataset_explanation(
    explanation_obs_embedding: onp.ndarray,
    embedded_dataset: onp.ndarray,
    visualise_dataset_obs: onp.ndarray,
    num_explanations: int = 2,
    explanation_length: int = 50,
) -> onp.ndarray:
    """Implementation of the dataset explanation with a provided explanation obs embedding
        and a dataset of embeddings, along with a dataset of observations to visualise the agent.

    Args:
        explanation_obs_embedding: The explanation observation embedding
        embedded_dataset: The embedded dataset
        visualise_dataset_obs: The visualisation dataset observations
        num_explanations: The number of explanations generated
        explanation_length: The length of the explanations provided

    Returns:
        Numpy array for each explanation
    """
    assert isinstance(embedded_dataset, onp.ndarray) and isinstance(
        explanation_obs_embedding, onp.ndarray
    )
    assert embedded_dataset.ndim == 2 and explanation_obs_embedding.ndim == 1
    assert embedded_dataset.shape[1] == explanation_obs_embedding.shape[0]

    # Compute the explanation obs embedding then distance between the embedded dataset and the embedded explanation obs
    explanation_dataset_distance = onp.linalg.norm(
        embedded_dataset - explanation_obs_embedding, axis=-1
    )
    assert explanation_dataset_distance.shape == (len(embedded_dataset),)

    # We wish to generate num_explanations that do not contain the close by points, i.e., within explanation_length
    #   We find the minimum index then maximise the following explanation_length points
    #   To note, np.argpartition exists to efficiently find the minimum K indexes however
    #   as we wish to ignore the following indexes this doesn't work
    minimum_dataset_indexes = onp.zeros(num_explanations, dtype=onp.int32)
    max_distance = onp.max(explanation_dataset_distance) + 1
    for num in range(num_explanations):
        minimum_dataset_indexes[num] = onp.argmin(explanation_dataset_distance)
        following_points = min(
            minimum_dataset_indexes[num] + explanation_length, len(embedded_dataset)
        )
        explanation_dataset_distance[
            minimum_dataset_indexes[num] : following_points
        ] = max_distance

    # Using the minimum dataset indexes then we find the next explanation_length observations,
    #   this assumes that we have not reached the end of a trajectory which may not be true
    dataset_explanation_indexes = onp.clip(
        minimum_dataset_indexes[:, None] + onp.arange(explanation_length)[None, :],
        0,
        len(embedded_dataset) - 1,
    )
    assert dataset_explanation_indexes.shape == (num_explanations, explanation_length)
    explanation_obs = visualise_dataset_obs[dataset_explanation_indexes]
    assert (
        explanation_obs.shape
        == (num_explanations, explanation_length) + visualise_dataset_obs.shape[1:]
    )
    return explanation_obs


def generate_skill_explanation(
    explain_obs_embedding: onp.ndarray,
    explain_obs_skill: int,
    dataset_embedding: onp.ndarray,
    plan: Plan,
    visualise_dataset_obs: onp.ndarray,
    num_explanations: int = 2,
    max_length: int = 50,
) -> list[onp.ndarray]:
    """Generates a skill explanation from the explanation obs skill and using the dataset skill transitions.

    Args:
        explain_obs_embedding: The embedding of the observation to explain
        explain_obs_skill: The skill of the observation to explain
        dataset_embedding: Embedding of the training dataset
        plan: The plan from which to use the skills
        visualise_dataset_obs: The visualisation of dataset observations
        num_explanations: The number of explanations generated
        max_length: The maximum length of skill explanations

    Returns:
        The explanations from the skills
    """
    possible_explanation_skills: list[SkillInstance] = [
        skill
        for skill_list in plan.transitions[explain_obs_skill]
        for skill in skill_list
    ]
    assert all(
        isinstance(skill, SkillInstance) for skill in possible_explanation_skills
    )
    assert len(possible_explanation_skills) > 0

    possible_explanation_skills = sorted(
        possible_explanation_skills, key=lambda skill: skill.length, reverse=True
    )[: len(possible_explanation_skills) // 2]

    assert all(skill.length > 0 for skill in possible_explanation_skills)
    # Find the minimum distance to a skill embedding to the explain_obs_embedding
    skill_min_distance = onp.zeros(len(possible_explanation_skills), dtype=onp.float32)
    skill_arg_distance = onp.zeros(len(possible_explanation_skills), dtype=onp.int32)
    for i, skill in enumerate(possible_explanation_skills):
        embedding_distances = onp.linalg.norm(
            dataset_embedding[skill.dataset_start_index : skill.dataset_end_index]
            - explain_obs_embedding,
            axis=1,
        )
        skill_min_distance[i] = onp.min(embedding_distances)
        skill_arg_distance[i] = onp.argmin(embedding_distances)

    explanations = []
    for pos in onp.argsort(skill_min_distance)[:num_explanations]:
        skill = possible_explanation_skills[pos]
        arg = skill_arg_distance[pos]

        start_index = skill.dataset_start_index + arg - max_length // 4
        end_index = skill.dataset_start_index + arg + 3 * max_length // 4

        assert end_index - start_index <= max_length
        explanations.append(visualise_dataset_obs[start_index:end_index])

    return explanations


def generate_plan_explanation(
    explain_obs_embedding: onp.ndarray,
    explain_obs_skill: int,
    dataset_embedding: onp.ndarray,
    plan: Plan,
    visualise_dataset_obs: onp.ndarray,
    num_explanations: int = 2,
    max_length: int = 50,
) -> tuple[list[onp.ndarray], str]:
    """Generates a skill explanation from the explanation obs skill and using the dataset skill transitions.

    Args:
        explain_obs_embedding: The embedding of the observation to explain
        explain_obs_skill: The skill of the observation to explain
        dataset_embedding: Embedding of the training dataset
        plan: The plan from which to use the skills (must include `skill_knowledge` data for each skill)
        visualise_dataset_obs: The visualisation of dataset observations
        num_explanations: The number of explanations generated
        max_length: The maximum length of skill explanations

    Returns:
        The visual explanation along with expert knowledge text
    """
    assert isinstance(plan.skill_knowledge, onp.ndarray)
    assert plan.skill_knowledge.dtype == np.object_, type(plan.skill_knowledge.dtype)
    assert plan.skill_knowledge.ndim == 1
    assert plan.skill_knowledge.shape == (plan.unique_skill_types,)

    visualise_explanation = generate_skill_explanation(
        explain_obs_embedding,
        explain_obs_skill,
        dataset_embedding,
        plan,
        visualise_dataset_obs,
        num_explanations=num_explanations,
        max_length=max_length,
    )
    expert_knowledge_explanation: str = plan.skill_knowledge[explain_obs_skill]

    return visualise_explanation, expert_knowledge_explanation


def generate_atari_grad_cam_explanation(
    network_def: nn.Module,
    network_params: FrozenDict,
    obs: onp.ndarray,
    action: int,
    feature_method: str = "features",
    q_network_method: str = "q_network",
) -> tuple[onp.ndarray, onp.ndarray]:
    """Runs grad cam for atari

    Args:
        network_def: The network definition
        network_params: The network parameters
        obs: The observations to run
        action: The action to use for computing grad cam
        feature_method: The method for computing the features
        q_network_method: The method for computing the Q-network

    Returns:
        The saliency map and true saliency values
    """

    def feature_q_value(features):
        q_values = network_def.apply(
            network_params, features, method=getattr(network_def, q_network_method)
        )
        return q_values[action]

    # Computes the convolution features for the particular layer
    conv_features = network_def.apply(
        network_params, obs, method=getattr(network_def, feature_method)
    )
    width, height, num_features = conv_features.shape

    # This finds the gradients for the convolutional features with respect with to the q values
    gradient = jax.grad(feature_q_value)(conv_features)

    # Finds the guided gradients and weights for each feature (averaging for each pixel)
    guided_gradients = (conv_features > 0) * (gradient > 0) * gradient
    assert guided_gradients.shape == conv_features.shape

    weights = jnp.mean(guided_gradients, axis=(0, 1))
    assert weights.shape == (num_features,)

    grad_cam = jnp.sum(weights * conv_features, axis=2)
    assert grad_cam.shape == (width, height)

    resized_grad_cam = cv2.resize(onp.array(grad_cam), obs.shape[:2])
    normalised_grad_cam = (resized_grad_cam / resized_grad_cam.max() * 255).astype(
        onp.uint8
    )

    return normalised_grad_cam, grad_cam


"""-------------------------------- Perturbation saliency --------------------------------"""


def _generate_perturbation_mask(
    center: tuple[int, int], mask_points: onp.ndarray, radius: int = 25
):
    """Generates the saliency mask

    Args:
        center: The mask center position
        mask_points: Numpy array of mask points
        radius: The radius of the mask

    Returns:
        Normalised mask over the mask points at the center (mean) and radius (standard deviation)
    """
    mask = multivariate_normal.pdf(mask_points, center, radius)
    return mask / mask.max()


def generate_atari_perturbation_saliency_explanation(
    obs: onp.ndarray,
    network_def: nn.Module,
    network_params: flax.core.FrozenDict,
    perturbation_spacing: int = 5,
) -> tuple[onp.ndarray, onp.ndarray]:
    """
    Generate the saliency map for a particular observation and JAX network definition following Greydanus et al., 2018

    Args:
        obs: observations of the agent
        network_def: Network definition
        network_params: Network parameters
        perturbation_spacing: The perturbation patch size, this speeds up the algorithm. For optimal saliency map
        use a spacing of 1, the default is 5 middle ground

    Returns:
         Tuple of the saliency map and the raw saliency values in a 2d array
    """
    if obs.shape == (84, 84, 4):
        obs = onp.expand_dims(obs, 0)

    assert obs.shape == (
        1,
        84,
        84,
        4,
    ), "for simplicity, we only allow a single observation, propose merge if clever solutions can be found"
    assert 0 <= perturbation_spacing, "perturbation spacing must be greater than zero"

    # Finds the blurred observations and mask points for generating the mask
    blurred_obs = gaussian_filter(obs, sigma=3)
    mask_points = onp.dstack(onp.meshgrid(onp.arange(84), onp.arange(84)))
    assert blurred_obs.shape == (1, 84, 84, 4) and mask_points.shape == (84, 84, 2), (
        f"Blurred obs: {blurred_obs.shape} - expected (1, 84, 84, 4) and "
        f"mask points: {mask_points.shape} - expected (84, 84, 2)"
    )

    perturbation_per_row = 84 // perturbation_spacing + 1

    # For each point on the observation, generating the perturbation
    #   This allows for all the perturbed observation to be run in parallel with vmap network_q_values
    perturbed_obs = onp.zeros((perturbation_per_row**2, 84, 84, 4))
    for x, x_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
        for y, y_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
            # calculate the mask for the center points and repeat for 4 dimensions to match the actual obs (84, 84, 4)
            mask = _generate_perturbation_mask(
                (x_center, y_center), mask_points, perturbation_spacing**2
            )
            obs_mask = onp.repeat(mask[:, :, onp.newaxis], 4, axis=2)

            # Compute the perturbed observation using the equation from the paper
            perturbed_obs[y * perturbation_per_row + x] = (
                obs * (1 - obs_mask) + blurred_obs * obs_mask
            )

    # assert onp.all(perturbed_ob != obs for perturbed_ob in perturbed_obs)

    @jax.vmap
    def _network_q_values(state):
        return network_def.apply(network_params, state)

    # calculate the true q-values and perturbed q-values
    true_q_values = network_def.apply(network_params, obs)
    perturbed_q_values = _network_q_values(perturbed_obs)

    # The saliency is the mean square error between the output policies, assuming a dqn agent
    saliency = jnp.mean(jnp.square(true_q_values - perturbed_q_values), axis=1)
    assert saliency.shape == (perturbation_per_row**2,)
    # assert onp.all(0 < saliency)

    # calculate the 2d saliency map based on the saliency list
    saliency_map = onp.zeros((84, 84))
    saliency_values = onp.zeros(
        (84 // perturbation_spacing + 1, 84 // perturbation_spacing + 1)
    )
    for x, x_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
        for y, y_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
            mask = _generate_perturbation_mask(
                (x_center, y_center), mask_points, perturbation_spacing**2
            )
            saliency_map += saliency[y * perturbation_per_row + x] * mask
            saliency_values[
                y_center // perturbation_spacing, x_center // perturbation_spacing
            ] = saliency[y * perturbation_per_row + x]

    # for the map, normalise the saliency values to between 0 and 255
    #   while the values are the raw saliency values (84, 84). For a list of saliency values, .flatten()
    return (
        onp.array(255 * saliency_map / onp.max(saliency_map), dtype=onp.uint16),
        saliency_values,
    )


def atari_greyscale_saliency_map(
    agent_obs: onp.ndarray,
    saliency_map: onp.ndarray,
    channel: int = 0,
    clip_bound: int = 0,
):
    """Using the saliency map description in Wang et al., 2016 the agent's observation (just the most recent)
        that are greyscale are shown as the green channel while the saliency map is used for the red and blue channels

    Args:
        agent_obs: The agent observations
        saliency_map: The saliency map
        channel: The RGB channel to add the saliency map to the agent obs
        clip_bound: The bound for showing part of the saliency map

    Returns:
        An RGB image representing the agent saliency map
    """
    assert isinstance(agent_obs, onp.ndarray)
    if agent_obs.ndim == 4:
        assert agent_obs.shape[0] == 1
        agent_obs = agent_obs[0]
    assert agent_obs.shape == (84, 84, 4)
    assert onp.all(0 <= agent_obs) and onp.all(agent_obs <= 255)

    assert (
        saliency_map.shape == (84, 84)
        and onp.all(0 <= saliency_map)
        and onp.all(saliency_map <= 255)
    ), f"Saliency map: {saliency_map.shape}, min: {onp.min(saliency_map)}, max: {onp.max(saliency_map)}"

    colour_agent_obs = onp.dstack([agent_obs[:, :, -1] for _ in range(3)])
    # colour_agent_obs = cv2.cvtColor(_agent_obs[0, ..., -1], cv2.COLOR_GRAY2RGB)
    saliency_map = (saliency_map > clip_bound) * saliency_map
    colour_saliency_map = onp.zeros((84, 84, 3))
    colour_saliency_map[:, :, channel] = saliency_map
    return (colour_saliency_map + colour_agent_obs).clip(0, 255).astype(onp.uint8)
