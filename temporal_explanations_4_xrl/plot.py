"""
Visualisation functions for the planning model.
"""
from __future__ import annotations

import random as rnd

import matplotlib.colors as colours
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

from temporal_explanations_4_xrl.dataset import DatasetEpisode
from temporal_explanations_4_xrl.skill import SkillInstance


def plot_embedding(
    embedding: np.ndarray,
    state_values: np.ndarray,
    trajectories: list[DatasetEpisode] = None,
    title: str = None,
    figsize: tuple[int, int] = None,
    fontsize: int = 10,
    point_size: int = 4,
) -> tuple[plt.Figure, plt.Axes | tuple[plt.Axes, plt.Axes]]:
    """Visualise a 2-dimensional embedding using the state values and trajectories.

    Additional, on hover capacity with obs argument.

    Args:
        embedding: A two-dimensional array that represents the embedding of a high-dimensional dataset
        state_values: The state value of embedding
        trajectories: A list of trajectories that optionally shows the trajectory stage
        title: The figure title
        figsize: The font size of the titles
        fontsize: The figure size
        point_size: Each embedding point size

    Returns:
        The figure and it's axes
    """
    assert (
        len(embedding.shape) == 2 and embedding.shape[-1] == 2
    ), f"Expected embedding shape to be (N, 2), actual shape is {embedding.shape}"
    assert state_values.shape == (
        len(embedding),
    ), f"Expected state value shape to be ({len(embedding)},), actual shape is {state_values.shape}"

    point_colours, ax_titles = [state_values], ["State values"]
    if trajectories:
        point_colours.append(
            np.concatenate(
                [np.linspace(0, 1, traj_len) for traj_len, _, _ in trajectories]
            )
        )
        ax_titles.append("Trajectory stages")

    fig, axs = plt.subplots(ncols=len(point_colours), figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=fontsize)
    if len(point_colours) == 1:
        axs = [axs]
    for ax, colour, ax_title in zip(axs, point_colours, ax_titles):
        ax.set_title(ax_title)
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=point_size * np.ones(len(embedding)),
            c=colour,
            edgecolor="none",
            picker=5,
            cmap="rainbow",
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(ax_title)

    plt.tight_layout()
    return fig, axs


def plot_skill_clusters(
    embedding: np.ndarray,
    skill_labels: np.ndarray,
    cluster_centers: np.ndarray = None,
    title: str = None,
    figsize: tuple[int, int] = None,
    point_size: int = 4,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualise a skill clusters in a 2-dimensional embedding.

    Additional, on hover capacity with obs argument.

    Args:
        embedding: A two-dimensional array that represents the embedding of a high-dimensional dataset
        skill_labels: An array of skill labels for each embedding
        cluster_centers: An array of cluster centers for each skill
        title: The figure title
        figsize: The figure size
        point_size: Each embedding point size

    Returns:
        The figure and it's axes
    """

    assert (
        len(embedding.shape) == 2 and embedding.shape[-1] == 2
    ), f"Expected embedding shape to be (N, 2), actual shape is {embedding.shape}"

    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=point_size * np.ones(len(embedding)),
        c=skill_labels,
        label=skill_labels,
        cmap="rainbow",
    )

    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color="red")
        for skill, (x, y, _) in enumerate(cluster_centers):
            ax.text(x, y, skill, fontweight="bold")

    plt.tight_layout()
    return fig, ax


def plot_skill_trajectory(
    embedding: np.ndarray,
    skill_transitions: list[SkillInstance],
    skill: int,
    title: str = None,
    figsize: tuple[int, int] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualises an example trajectories of a skills in the embedding

    Args:
        embedding: A two-dimensional array for the embedding
        skill_transitions: A list of skill transitions
        skill: The skill cluster
        title: Figure title
        figsize: Figure size

    Returns:
        Figure and axe
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)

    skill_colours, transition_colours = {}, {}

    all_skill_colours = list(colours.TABLEAU_COLORS.values())
    all_transition_colours = list(colours.CSS4_COLORS.values())
    for transition in filter(lambda t: t.type == skill, skill_transitions):
        transition_embedding = embedding[
            transition.dataset_start_index : transition.dataset_start_index
            + transition.length
        ]

        if transition.previous_type not in skill_colours:
            skill_colours[transition.previous_type] = all_skill_colours[
                len(skill_colours)
            ]
        if transition.next_type not in skill_colours:
            skill_colours[transition.next_type] = all_skill_colours[len(skill_colours)]
        if (transition.previous_type, transition.next_type) not in transition_colours:
            transition_colours[
                (transition.previous_type, transition.next_type)
            ] = rnd.choice(all_transition_colours)

        ax.plot(
            transition_embedding[:, 0],
            transition_embedding[:, 1],
            color=transition_colours[(transition.previous_type, transition.next_type)],
        )
        ax.scatter(
            transition_embedding[0, 0],
            transition_embedding[0, 1],
            color=skill_colours[transition.previous_type],
        )
        ax.scatter(
            transition_embedding[-1, 0],
            transition_embedding[-1, 1],
            color=skill_colours[transition.next_type],
        )

    plt.legend(
        handles=[
            Patch(color=colour, label=f"Skill {skill}")
            for skill, colour in skill_colours.items()
        ],
        title="Skills",
        loc="lower right",
    )
    plt.tight_layout()
    return fig, ax


def plot_trajectory_skills(
    skills: list[SkillInstance],
    state_values: np.ndarray,
    title: str = None,
    figsize: tuple[int, int] = None,
):
    """Visualise the skills within a trajectory.

    Args:
        skills: Array of state values
        state_values: list of skill transitions
        title: The figure title
        figsize: The figure size

    Returns:
        Figure and axe
    """
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    ax.set_ylabel("State value")
    ax.set_xlabel("Time step")

    # skill_info = ax.text(
    #     1,
    #     onp.max(
    #         state_values[skills[0].dataset_start_index : skills[-1].dataset_end_index]
    #     )
    #     - 1,
    #     "Skills info",
    # )

    skill_colours = np.random.choice(
        list(colours.CSS4_COLORS.values()),
        max(transition.type for transition in skills) + 1,
        replace=False,
    )
    transition_plots = []
    for transition in skills:
        transition_plots.append(
            ax.plot(
                np.arange(transition.dataset_start_index, transition.dataset_end_index),
                state_values[
                    transition.dataset_start_index : transition.dataset_end_index
                ],
                color=skill_colours[transition.type],
            )[0]
        )

    plt.legend(
        handles=[
            Patch(color=colour, label=f"Skill {skill}")
            for skill, colour in enumerate(skill_colours)
        ],
        title="Skills",
        bbox_to_anchor=(1.1, 1.0),
    )

    # plt.tight_layout()
    return fig, ax


def animate_observations(
    obs: np.ndarray,
    frame_interval: int = 200,
    figsize: tuple[int, int] = (8, 8),
    fig_title: str = None,
    return_html_animation: bool = True,
):
    """Animate a list of observations.

    Args:
        obs: The observations to animate
        frame_interval: The number of milliseconds between frames (observations)
        figsize: The figure size
        fig_title: The figure title
        return_html_animation: If to return the html animation or figure, axes and animation function

    Returns:
        Either a HTML script for the video or the figure, axes and animation fn depending on return_htm_animation argument.
    """
    assert len(obs.shape) == 3

    fig, ax = plt.subplots(figsize=figsize)
    if fig_title:
        fig.suptitle(fig_title)

    ax.axis("off")
    obs_plot = ax.imshow(obs[0, :, :], cmap="gray")
    ax.set_xlabel("0")

    def _animate_time_step(time_step: int):
        obs_plot.set_data(obs[time_step])
        ax.set_xlabel(time_step)
        return [obs_plot, ax]

    animation_fn = FuncAnimation(
        fig, _animate_time_step, frames=obs.shape[0], interval=frame_interval
    )
    if return_html_animation:
        plt.close()
        return HTML(animation_fn.to_html5_video())
    else:
        return fig, ax, animation_fn
