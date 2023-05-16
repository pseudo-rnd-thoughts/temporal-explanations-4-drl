from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from temporal_explanations_4_xrl.skill import (
    SkillInstance,
    normalised_skill_cluster_alignment,
)
from temporal_explanations_4_xrl.utils import flatten


class Plan:
    """
    A planning model that utilises a list of skill transitions to compute the skill transition size and length that
        can be used for finding the transition probability and representative observations
    """

    def __init__(
        self,
        skill_instances: list[SkillInstance] | list[list[SkillInstance]],
        skill_knowledge: np.ndarray | None = None,
        skill_transition_knowledge: np.ndarray | None = None,
    ):
        """The number of skills should not include the initial or terminal skill"""
        if isinstance(skill_instances[0], list):
            skill_instances = flatten(skill_instances)
        # The skills used to construct the plan
        self.all_skill_instances = skill_instances

        # The total number of skill transitions in the planning model
        self.total_skills = len(skill_instances)

        self.unique_skill_types = len({skill.type for skill in skill_instances})
        self.skill_instances: list[list[SkillInstance]] = [
            [] for _ in range(self.unique_skill_types)
        ]
        for skill in skill_instances:
            self.skill_instances[skill.type].append(skill)

        self.transition_shape = (
            self.unique_skill_types + 1,
            self.unique_skill_types + 1,
        )

        # The skill transitions between skill A to skill B, shape=(skill A, skill B)
        self.transitions = np.zeros(self.transition_shape, dtype=object)
        for index in np.ndindex(*self.transition_shape):
            self.transitions[index] = []

        for skill in skill_instances:
            if skill.is_initial_skill():
                self.transitions[-1, skill.type].append(skill)

            if skill.is_final_skill():
                self.transitions[skill.type, -1].append(skill)
            else:
                self.transitions[skill.type, skill.next_type].append(skill)

        self.transition_count = np.zeros(self.transition_shape, dtype=np.int32)
        self.transition_lengths = np.zeros(self.transition_shape, dtype=object)
        self.transition_actions = np.zeros(self.transition_shape, dtype=object)
        for index in np.ndindex(*self.transition_shape):
            self.transition_count[index] = len(self.transitions[index])
            self.transition_lengths[index] = [
                skill.length for skill in self.transitions[index]
            ]
            self.transition_actions[index] = [
                skill.actions for skill in self.transitions[index]
            ]

        self.transition_probability = self.transition_count / np.repeat(
            self.transition_count.sum(axis=0), self.unique_skill_types + 1
        ).reshape(self.unique_skill_types + 1, self.unique_skill_types + 1)
        assert np.allclose(self.transition_probability.sum(axis=1), 1)

        # Expert Knowledge
        if skill_knowledge is not None and skill_transition_knowledge is not None:
            assert (
                isinstance(skill_knowledge, np.ndarray)
                and skill_knowledge.dtype == object
            )
            assert skill_knowledge.ndim == 1 and skill_knowledge.shape == (
                self.unique_skill_types,
            )

            assert (
                isinstance(skill_transition_knowledge, np.ndarray)
                and skill_transition_knowledge.dtype == object
            )
            assert (
                skill_transition_knowledge.ndim == 2
                and skill_transition_knowledge.shape == self.transition_shape
            )
            self.skill_knowledge = skill_knowledge
            self.skill_transition_knowledge = skill_transition_knowledge
        else:
            self.skill_knowledge = np.full(self.unique_skill_types, "", dtype=object)
            self.skill_transition_knowledge = np.full(
                self.transition_shape, "", dtype=object
            )

    def transition_entropy(self) -> np.ndarray:
        """Computes the entropy for each skill."""
        return stats.entropy(self.transition_probability, axis=1)

    def entropy(self) -> float:
        """Computes the entropy for the plan."""
        return float(np.mean(self.transition_entropy()))

    def transition_length_distribution(
        self, min_skill_instances: int = 5
    ) -> list[dict[int, float]]:
        """Computes the skill length distribution."""
        return [
            {
                next_skill_type: stats.kstest(
                    self.transition_lengths[skill_type, next_skill_type],
                    "poisson",
                    args=(
                        np.mean(self.transition_lengths[skill_type, next_skill_type]),
                    ),
                ).pvalue
                for next_skill_type in range(self.unique_skill_types + 1)
                if self.transition_count[skill_type, next_skill_type]
                > min_skill_instances
            }
            for skill_type in range(self.unique_skill_types + 1)
        ]

    def length_distribution(self) -> float:
        return float(
            np.mean(
                flatten(
                    [
                        list(skill_distributions.values())
                        for skill_distributions in self.transition_length_distribution()
                    ]
                )
            )
        )

    def transition_alignment(
        self,
        penalty: float,
        policy_distance: callable,
        max_iters: int = 100,
        min_skill_instances: int = 3,
    ):
        return [
            {
                next_skill_type: np.mean(
                    normalised_skill_cluster_alignment(
                        self.transitions[skill_type, next_skill_type],
                        penalty,
                        policy_distance,
                        max_iters=max_iters,
                    )
                )
                for next_skill_type in range(self.unique_skill_types + 1)
                if self.transition_count[skill_type, next_skill_type]
                > min_skill_instances
            }
            for skill_type in tqdm(
                range(self.unique_skill_types + 1),
                desc="Transition alignment",
                mininterval=3.0,
            )
        ]

    def alignment(self, penalty: float, policy_distance: callable):
        return np.mean(
            flatten(
                [
                    list(skill_alignments.values())
                    for skill_alignments in self.transition_alignment(
                        penalty, policy_distance
                    )
                ]
            )
        )

    def lengths(self, min_skill_instances: int = 5) -> np.ndarray:
        return np.mean(
            [
                np.mean(self.transition_lengths[index])
                for index in np.ndindex(*self.transition_shape)
                if self.transition_count[index] > min_skill_instances
            ]
        )

    def add_skill_knowledge(self, skill: int, knowledge: str):
        self.skill_knowledge[skill] = knowledge

    def add_skill_transition_knowledge(
        self, skill: int, next_skill: int, knowledge: str
    ):
        self.skill_transition_knowledge[skill, next_skill] = knowledge

    def save(self, filename: str):
        """Saves the plan to a filename."""
        np.savez_compressed(
            filename,
            {
                "transition_count": self.transition_count,
                "transition_lengths": self.transition_lengths,
                "transition_actions": self.transition_actions,
                "skill_knowledge": self.skill_knowledge,
                "skill_transition_knowledge": self.skill_transition_knowledge,
            },
        )

    def plot(self, figsize: tuple[int, int] | None) -> tuple[plt.Figure, plt.Axes]:
        """Plot the plan as a direct bipartite graph.

        Args:
            figsize: The figure size

        Returns:
            The figure and axe
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.set_ylim(0.5, 2.4)

        # Draw the nodes
        next_skill_nodes = ax.scatter(
            np.arange(2, self.unique_skill_types + 3),
            np.full(self.unique_skill_types + 1, 2),
            color="red",
        )
        current_skill_nodes = ax.scatter(
            np.arange(2, self.unique_skill_types + 2),
            np.full(self.unique_skill_types, 1.5),
            color="red",
        )

        # Draw the node text
        for index in np.arange(2, self.unique_skill_types + 2):
            ax.text(
                index - 0.5,
                2.05 + 0.08 * (index % 2),
                f"Skill {index - 2}",
                color="black",
            )  # Next skill text
            ax.text(
                index - 0.5,
                0.85 - 0.08 * (index % 2),
                f"Skill {index - 2}",
                color="black",
            )  # Previous skill text
        ax.text(
            self.unique_skill_types + 1.2,
            2.05 + 0.08 * ((self.unique_skill_types + 2) % 2),
            "Termination",
        )
        ax.text(0.44, 0.8, "Starting")

        # Node titles
        ax.text(self.unique_skill_types // 2 + 1, 2.3, "Next skills", fontsize=13)

        debug_text = ax.text(2, 0.3, "debug text")

        # Draw the transition edges
        next_skill_transition_edges = {}
        for skill in np.arange(self.unique_skill_types):
            for next_skill in np.arange(self.unique_skill_types + 1):
                if 0 < self.transition_probability[skill, next_skill]:
                    (edge,) = ax.plot(
                        [skill + 2, next_skill + 2],
                        [1.5, 2],
                        linewidth=100 * self.transition_probability[skill, next_skill],
                        color="black",
                    )
                    next_skill_transition_edges[(skill, next_skill)] = edge

        def _on_click(event):
            if current_skill_nodes.contains(event)[0]:
                debug_text.set_text(
                    f'Current skill node: {current_skill_nodes.contains(event)[1]["ind"]}'
                )
            elif next_skill_nodes.contains(event)[0]:
                debug_text.set_text(
                    f'Next skill node: {next_skill_nodes.contains(event)[1]["ind"]}'
                )

            for (_skill, _next_skill), _edge in next_skill_transition_edges.items():
                if _edge.contains(event)[0]:
                    debug_text.set_text(f"Skill edge: {_skill}-{_next_skill}")
                    break

        fig.canvas.mpl_connect("button_press_event", _on_click)
        plt.tight_layout()
        return fig, ax
