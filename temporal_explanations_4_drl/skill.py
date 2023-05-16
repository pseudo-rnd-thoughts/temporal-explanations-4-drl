"""
Implements a planning model class and a subclass expert planning model class that allows expert model to be embedded
    into the plan
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyalign
except ImportError:
    pyalign = None

from temporal_explanations_4_xrl.dataset import DatasetEpisode

INITIAL_SKILL = -1
TERMINAL_SKILL = -2
# todo - add truncation skill


@dataclass(frozen=True)
class SkillInstance:
    """A skill instance discovered in a dataset.

    - Previous skill, this instance's skill and next skill number
    - The actions for the instance
    - The dataset start index
    """

    previous_type: int
    type: int
    next_type: int

    actions: np.ndarray
    dataset_start_index: int

    @property
    def dataset_end_index(self) -> int:
        """Computes the dataset end index using the dataset start index and the skill length.

        Returns:
             The dataset end index
        """
        return self.dataset_start_index + self.length

    @property
    def length(self):
        """The length of the skill."""
        return len(self.actions)

    def is_initial_skill(self) -> bool:
        """Computes if the skill is the first skill in a trajectory.

        Returns:
            If the skill is the first skill in a trajectory
        """
        return self.previous_type == INITIAL_SKILL

    def is_final_skill(self) -> bool:
        """Computes if the skill is the last skill in a trajectory.

        Returns:
             If the skill is the last skill in a trajectory
        """
        return self.next_type == TERMINAL_SKILL


def skill_labels_to_trajectory_skills(
    skill_labels: np.ndarray, actions: np.ndarray, trajectories: list[DatasetEpisode]
) -> list[list[SkillInstance]]:
    """Converts the skill labels to skills.

    Args:
        skill_labels: The skill label for each embedding
        actions: A dataset of actions for each embedding
        trajectories: The dataset trajectories, assumed to be in order

    Returns:
        List of trajectory skill instances
    """
    assert (
        len(skill_labels) == trajectories[-1].end
    ), f"The number of skill label ({len(skill_labels)}) != last trajectory end ({trajectories[-1].end})"
    assert (
        len(actions) == trajectories[-1].end
    ), f"The number of actions ({len(actions)}) != last trajectory end ({trajectories[-1].end})"

    trajectory_skills, pos = [], 0
    for trajectory in trajectories:
        skill_instances = []
        # The first skill in the trajectory
        previous_skill_type = INITIAL_SKILL

        while pos < trajectory.end:
            # Computes the length of a skill
            skill_length = next(
                (
                    length - 1
                    for length in range(1, trajectory.end - pos + 1)
                    if not np.all(skill_labels[pos : pos + length] == skill_labels[pos])
                ),
                trajectory.end - pos,
            )

            # Computes the next skill
            if pos + skill_length < trajectory.end:
                next_skill_type = skill_labels[pos + skill_length]
            else:
                # If the trajectory is terminated, use the num_skills as special case representing terminating skill
                next_skill_type = TERMINAL_SKILL

            # Add the skill transition
            skill_instances.append(
                SkillInstance(
                    previous_type=previous_skill_type,
                    type=skill_labels[pos],
                    next_type=next_skill_type,
                    actions=actions[pos : pos + skill_length],
                    dataset_start_index=pos,
                )
            )

            # Update the previous skills and dataset position
            previous_skill_type, pos = skill_labels[pos], pos + skill_length
        trajectory_skills.append(skill_instances)

    return trajectory_skills


# todo - fix
# def bag_of_skills(
#     trajectory_skills: list[list[SkillInstance]],
#     bag_size: int = 2,
# ) -> np.ndarray:
#     """Groups the skill into a bag of skills, a similar idea to Bag of Words (BOW) in NLP (natural language processing).
#
#     Args:
#         trajectory_skills: A list of skill for each trajectory
#         bag_size: The bag size, by default 2 being the previous skill and the current skill
#
#     Returns:
#         A matrix of number of skill bags
#     """
#     assert 2 <= bag_size, f"Bag size ({bag_size}) must be greater or equal to 2"
#
#     # To include the special codes of starting and end skills (=num_skills) then extra space required
#     bag = np.zeros(
#         (num_skills + 1,) + (num_skills,) * (bag_size - 2) + (num_skills + 1,),
#         dtype=np.int32,
#     )
#     for skill_transition in trajectory_skills:
#         # Get all previous skills
#         bag[
#             tuple(
#                 transition.previous_skill_number for transition in skill_transition[:bag_size]
#             )
#         ] += 1
#
#         # Get all current skills
#         for pos in range(len(skill_transition) - bag_size + 1):
#             bag[
#                 tuple(
#                     transition.skill_number
#                     for transition in skill_transition[pos : pos + bag_size]
#                 )
#             ] += 1
#
#         # Get the final transition skills
#         bag[
#             tuple(transition.next_skill_number for transition in skill_transition[-bag_size:])
#         ] += 1
#     return bag


def discrete_policy_distance(x_action: int, y_action: int) -> int:
    """Policy distance for environments with discrete action spaces, computing the total variational distance.

    Args:
        x_action: An action
        y_action: An action

    Returns:
        The policy distance
    """
    return np.equal((x_action - y_action), 0)


def skill_alignment(
    skill_1: SkillInstance,
    skill_2: SkillInstance,
    penalty: float,
    policy_distance: callable,
    direction: str = "maximize",
) -> tuple[float, pyalign.solve.Solution]:
    """Computes the skill alignment between two skills using the Smith-Waterman Algorithm.

    Args:
        skill_1: A skill instance
        skill_2: A skill instance
        penalty: The penalty for action gaps (insertions or deletions)
        policy_distance: The policy distance metric
        direction: The direction of the policy distance, "maximize" -> higher distance is greater similarity

    Returns:
        A tuple of the alignment score and alignment table
    """
    pf = pyalign.problems.general(policy_distance, direction=direction)
    solver = pyalign.solve.GlobalSolver(
        gap_cost=pyalign.gaps.LinearGapCost(penalty), codomain=pyalign.solve.Solution
    )
    problem = pf.new_problem(skill_1.actions, skill_2.actions)
    solution = solver.solve(problem)
    return solution.score, solution


def normalised_skill_cluster_alignment(
    skills: list[SkillInstance],
    penalty: float,
    policy_distance: callable,
    direction: str = "maximize",
    max_iters: int = 100,
) -> np.ndarray:
    """Computes normalised skill cluster alignment score.

    Args:
        skills: List of skill instances
        penalty: The penalty for gaps used by the skill_alignment function
        policy_distance: The policy distance used by the skill_alignment function
        direction: The direction of the policy distance
        max_iters: The maximum number of alignment to compute, otherwise compute random alignments

    Returns:
        List of normalised skill alignments for a cluster
    """
    pf = pyalign.problems.general(policy_distance, direction=direction)
    solver = pyalign.solve.LocalSolver(
        gap_cost=pyalign.gaps.LinearGapCost(penalty), codomain=pyalign.solve.Solution
    )

    indexes = np.array(
        [[i, j] for i in range(len(skills) - 1) for j in range(i, len(skills))]
    )
    if len(indexes) > max_iters:
        indexes = indexes[np.random.choice(len(indexes), size=max_iters, replace=False)]

    normalised_alignment = np.zeros(len(indexes))
    for pos, (i, j) in enumerate(indexes):
        problem = pf.new_problem(skills[i].actions, skills[j].actions)
        solution: pyalign.solve.Solution = solver.solve(problem)

        normalised_alignment[pos] = np.max(solution.score) / min(
            skills[i].length, skills[j].length
        )
    return normalised_alignment
