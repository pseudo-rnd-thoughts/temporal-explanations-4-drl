import itertools
import os

import joblib
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
from temporal_explanations_4_xrl.dataset import (
    load_atari_human_obs,
    load_atari_obs,
    load_discrete_actions,
    load_q_values,
    load_trajectories,
)
from temporal_explanations_4_xrl.explain import (
    atari_greyscale_saliency_map,
    generate_atari_grad_cam_explanation,
    generate_atari_perturbation_saliency_explanation,
    generate_dataset_explanation,
    generate_obs_to_explain,
    generate_plan_explanation,
    generate_skill_explanation,
    save_observation_with_explanation,
    save_observation_with_two_explanations,
)
from temporal_explanations_4_xrl.plan import Plan
from temporal_explanations_4_xrl.skill import skill_labels_to_trajectory_skills
from temporal_explanations_4_xrl.utils import load_embedding


def generate_explanation_obs(
    dataset_folder: str, explain_trajectory_end: int
) -> np.ndarray:
    """Generates the explanation obs with their index in the trajectories dataset."""
    q_values = load_q_values(f"{dataset_folder}/trajectories")

    explanation_q_values = q_values[explain_trajectory_end:]
    explain_obs_dataset_indexes = generate_obs_to_explain(
        explanation_q_values, NUM_EXPLANATIONS
    )

    np.savez_compressed(
        f"{dataset_folder}/explanations.npz",
        explain_obs_dataset_indexes=explain_obs_dataset_indexes,
    )
    return explain_obs_dataset_indexes


def generate_dataset_explanations(
    dataset_folder: str, agent_name: str, env_name: str, clustering_algo: str
):
    """Generates all explanations for a dataset."""
    if all(
        os.path.exists(f"{dataset_folder}/explanation-{i}")
        for i in range(NUM_EXPLANATIONS)
    ):
        return

    trajectories = load_trajectories(f"{dataset_folder}/trajectories")
    explain_trajectory_end = trajectories[NUM_EXPLANATION_TRAJECTORIES].end
    assert (
        sum(traj.length for traj in trajectories[: NUM_EXPLANATION_TRAJECTORIES + 1])
        == explain_trajectory_end
    )

    if os.path.exists(f"{dataset_folder}/explanations.npz"):
        with np.load(
            f"{dataset_folder}/explanations.npz",
            allow_pickle=True,
        ) as file:
            explain_obs_dataset_indexes = file["explain_obs_dataset_indexes"]

            if len(explain_obs_dataset_indexes) != NUM_EXPLANATIONS:
                print("Regenerating explanation observations")
                explain_obs_dataset_indexes = generate_explanation_obs(
                    dataset_folder, explain_trajectory_end
                )

    else:
        print("Generating explanation observations")
        explain_obs_dataset_indexes = generate_explanation_obs(
            dataset_folder, explain_trajectory_end
        )

    dataset_embedding = load_embedding(f"{dataset_folder}/embedding/tsne-dense-50.npz")
    unseen_dataset_embedding = dataset_embedding[explain_trajectory_end:]

    human_obs = load_atari_human_obs(f"{dataset_folder}/trajectories")
    unseen_human_obs = human_obs[explain_trajectory_end:]
    explain_human_obs = human_obs[explain_obs_dataset_indexes]

    agent_obs = load_atari_obs(f"{dataset_folder}/trajectories")
    explain_agent_obs = agent_obs[explain_obs_dataset_indexes]

    actions = load_discrete_actions(f"{dataset_folder}/trajectories")
    explain_actions = actions[explain_obs_dataset_indexes]

    with np.load(
        f"{dataset_folder}/{clustering_algo}/summary.npz", allow_pickle=True
    ) as file:
        optimal_model = file["optimal_model"].item()
        optimal_model.pop("index")

    filename = "-".join(f"{key}-{value}" for key, value in optimal_model.items())
    with open(
        f"{dataset_folder}/{clustering_algo}/{filename}.joblib", "rb"
    ) as optimal_model_file:
        data = joblib.load(optimal_model_file)
        skill_labels = data["skill_labels"]
        plan = data["plan"]

        assert all(knowledge != "" for knowledge in plan.skill_knowledge)

    skills = skill_labels_to_trajectory_skills(skill_labels, actions, trajectories)
    unseen_plan = Plan(
        skills[NUM_EXPLANATION_TRAJECTORIES + 1 :],
        skill_knowledge=plan.skill_knowledge,
        skill_transition_knowledge=plan.skill_transition_knowledge,
    )

    if agent_name == "dqn_adam_mse":
        model_def, model_params = load_dopamine_dqn_flax_model(
            env_name, f"{MODEL_ROOT_FOLDER}dopamine/jax"
        )
    elif agent_name == "rainbow":
        model_def, model_params = load_dopamine_rainbow_flax_model(
            env_name, f"{MODEL_ROOT_FOLDER}dopamine/jax"
        )
    else:
        raise Exception(f"Unknown agent name: {agent_name}")

    # For each observation, generate all explanations types
    for i in tqdm(range(NUM_EXPLANATIONS)):
        if not os.path.exists(f"{dataset_folder}/explanation-{i}"):
            os.mkdir(f"{dataset_folder}/explanation-{i}")
        else:
            print(f"Skipping generating explanation {i}, already done")
            continue

        obs_embedding = dataset_embedding[explain_obs_dataset_indexes[i]]

        # Dataset similarity explanation
        dataset_explanations = generate_dataset_explanation(
            explanation_obs_embedding=obs_embedding,
            embedded_dataset=unseen_dataset_embedding,
            visualise_dataset_obs=unseen_human_obs,
        )
        dataset_explanations = dataset_explanations.reshape(
            (-1,) + dataset_explanations.shape[2:]
        )
        save_observation_with_explanation(
            obs=explain_human_obs[i],
            explanation=dataset_explanations,
            filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-dataset-explanation",
        )

        # Skill similarity explanation
        skill_explanations = generate_skill_explanation(
            explain_obs_embedding=obs_embedding,
            explain_obs_skill=skill_labels[explain_obs_dataset_indexes[i]],
            dataset_embedding=dataset_embedding,
            plan=unseen_plan,
            visualise_dataset_obs=human_obs,
        )
        skill_explanations = np.concatenate(skill_explanations)
        save_observation_with_explanation(
            obs=explain_human_obs[i],
            explanation=skill_explanations,
            filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-skill-explanation",
        )

        # Plan explanation
        plan_explanations = generate_plan_explanation(
            explain_obs_embedding=obs_embedding,
            explain_obs_skill=skill_labels[explain_obs_dataset_indexes[i]],
            dataset_embedding=dataset_embedding,
            plan=unseen_plan,
            visualise_dataset_obs=human_obs,
        )
        plan_visual_explanations, plan_expert_knowledge_explanation = plan_explanations
        plan_explanations = (
            np.concatenate(plan_visual_explanations),
            plan_expert_knowledge_explanation,
        )
        save_observation_with_explanation(
            obs=explain_human_obs[i],
            explanation=plan_explanations,
            filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-plan-explanation",
        )

        # GradCam - We use the second layer as provide more information than the first or final layer
        gradcam_explanation, _ = generate_atari_grad_cam_explanation(
            model_def,
            model_params,
            explain_agent_obs[i],
            explain_actions[i],
            feature_method="feature_1",
            q_network_method="q_network_1",
        )
        gradcam_explanation = np.expand_dims(
            atari_greyscale_saliency_map(explain_agent_obs[i], gradcam_explanation),
            axis=0,
        )
        save_observation_with_explanation(
            obs=explain_human_obs[i],
            explanation=gradcam_explanation,
            filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-gradcam-explanation",
        )

        # Perturbation-based Saliency Map
        (
            perturbation_explanation,
            _,
        ) = generate_atari_perturbation_saliency_explanation(
            explain_agent_obs[i], model_def, model_params
        )
        perturbation_explanation = np.expand_dims(
            atari_greyscale_saliency_map(
                explain_agent_obs[i], perturbation_explanation
            ),
            axis=0,
        )
        save_observation_with_explanation(
            obs=explain_human_obs[i],
            explanation=perturbation_explanation,
            filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-perturbation-explanation",
        )

        # Generate all explanation pairs
        for (explanation_1_name, explanation_1), (
            explanation_2_name,
            explanation_2,
        ) in itertools.combinations(
            (
                ("dataset", dataset_explanations),
                ("skill", skill_explanations),
                ("plan", plan_explanations),
                ("gradcam", gradcam_explanation),
                ("perturbation", perturbation_explanation),
            ),
            2,
        ):
            # Check if the explanations are skill and plan
            if {explanation_1_name, explanation_2_name} == {"skill", "plan"}:
                continue

            if np.random.random() > 0.5:
                explanation_1, explanation_2 = explanation_2, explanation_1
                explanation_1_name, explanation_2_name = (
                    explanation_2_name,
                    explanation_1_name,
                )
            save_observation_with_two_explanations(
                obs=explain_human_obs[i],
                explanation_1=explanation_1,
                explanation_2=explanation_2,
                filename=f"{dataset_folder}/explanation-{i}/{agent_name}-{env_name}-{i}-contrast-{explanation_1_name}-{explanation_2_name}",
            )


DATASET_ROOT_FOLDER = ""
MODEL_ROOT_FOLDER = "../models/"
NUM_EXPLANATIONS = 4
NUM_EXPLANATION_TRAJECTORIES = 2

SURVEY_AGENT_ENVS = [
    ("dqn_adam_mse", "Breakout"),
    ("dqn_adam_mse", "SpaceInvaders"),
    ("dqn_adam_mse", "Seaquest"),
]


if __name__ == "__main__":
    # Explain the Atari agents
    for _agent_name, _env_name in SURVEY_AGENT_ENVS:
        print(f"Agent: {_agent_name}, environment: {_env_name}")
        generate_dataset_explanations(
            dataset_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}",
            agent_name=_agent_name,
            env_name=_env_name,
            clustering_algo="kmeans-st",
        )

    print("================ Generating All Explanations ================")
    for _agent_name, _env_name in (
        DOPAMINE_DQN_ATARI_AGENT_ENVS + DOPAMINE_RAINBOW_ATARI_AGENT_ENVS
    ):
        print(f"Agent: {_agent_name}, environment: {_env_name}")
        generate_dataset_explanations(
            dataset_folder=f"{DATASET_ROOT_FOLDER}{_agent_name}-{_env_name}",
            agent_name=_agent_name,
            env_name=_env_name,
            clustering_algo="kmeans-st",
        )
