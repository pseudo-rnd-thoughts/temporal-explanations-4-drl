import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox

from temporal_explanations_4_xrl.dataset import load_atari_human_obs
from temporal_explanations_4_xrl.plan import Plan
from temporal_explanations_4_xrl.skill import SkillInstance

"""
Must be run from the terminal

 - Given a summary of clustering models, the optimal model is found
 - Using the optimal model's plan, the skill knowledge is replaced with the updated plan containing skill knowledge
"""

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--agent_name", default="dqn_adam_mse")
parser.add_argument("--env_name", default="Breakout")
parser.add_argument("--root", default="datasets/")
parser.add_argument("--clustering_model", default="kmeans-st/summary.npz")
args = parser.parse_args()

# Constant
NUM_VIEWERS = 5


# Load optimal plan
cluster_model_filename = args.clustering_model
if ".npz" in cluster_model_filename:
    print(
        f"Finding optimal model: {args.root}{args.agent_name}-{args.env_name}/{cluster_model_filename}"
    )
    with np.load(
        f"{args.root}{args.agent_name}-{args.env_name}/{cluster_model_filename}",
        allow_pickle=True,
    ) as file:
        optimal_model = file["optimal_model"].item()
        optimal_model.pop("index")

        folder = "/".join(cluster_model_filename.split("/")[:-1])
        filename = "-".join(f"{key}-{value}" for key, value in optimal_model.items())

print(
    f"Loading model: {args.root}{args.agent_name}-{args.env_name}/{folder}/{filename}.joblib"
)
with open(
    f"{args.root}{args.agent_name}-{args.env_name}/{folder}/{filename}.joblib", "rb"
) as file:
    data = joblib.load(file)

    plan = data["plan"]
    if not hasattr(plan, "all_skill_instances"):
        plan = Plan(plan.skill_instances)

    if any(knowledge != "" for knowledge in plan.skill_knowledge):
        print(f"Skill knowledge: {plan.skill_knowledge}")

obs = load_atari_human_obs(f"{args.root}{args.agent_name}-{args.env_name}/trajectories")
print(f"Number of skill types: {len(plan.skill_instances)}")

current_skill: int = 0
skills: list[SkillInstance] = [
    plan.skill_instances[current_skill][i]
    for i in np.random.choice(len(plan.skill_instances[0]), NUM_VIEWERS)
]
skill_pos = np.zeros((NUM_VIEWERS,), dtype=np.int32)

fig, obs_axs = plt.subplots(ncols=NUM_VIEWERS, figsize=(12, 3))
fig.suptitle(f"Expert knowledge for Skill: {current_skill}")

obs_images = []
for i, ax in enumerate(obs_axs):
    obs_images.append(ax.imshow(obs[skills[i].dataset_start_index]))
    ax.axis("off")


def skill_description(knowledge):
    global current_skill

    plan.add_skill_knowledge(current_skill, knowledge)
    current_skill += 1
    print(f"Current skill: {current_skill}")

    if current_skill >= len(plan.skill_instances):
        fig.suptitle("Finished")

        animation.event_source.stop()
        print("Saving data")
        with open(
            f"{args.root}{args.agent_name}-{args.env_name}/{folder}/{filename}.joblib",
            "rb",
        ) as file:
            data = joblib.load(file)
            data["plan"] = plan

        joblib.dump(
            data,
            f"{args.root}{args.agent_name}-{args.env_name}/{folder}/{filename}.joblib",
        )
        print("Updated plan dumped")
    else:
        fig.suptitle(f"Expert knowledge for Skill: {current_skill}")

        global skills, skill_pos
        skills = [
            plan.skill_instances[current_skill][i]
            for i in np.random.choice(
                len(plan.skill_instances[current_skill]), NUM_VIEWERS
            )
        ]
        skill_pos = np.zeros((NUM_VIEWERS,), dtype=np.int32)


text_ax = fig.add_axes([0.25, 0, 0.6, 0.1])
text_box = TextBox(text_ax, "Skill description")
text_box.on_submit(skill_description)
text_box.set_val("")


def update(time_step):
    global skills, skill_pos
    skill_pos += 1
    for i in range(NUM_VIEWERS):
        if skill_pos[i] >= skills[i].length:
            skills[i] = np.random.choice(plan.skill_instances[current_skill])
            skill_pos[i] = 0

    for i, obs_image in enumerate(obs_images):
        index = skills[i].dataset_start_index + skill_pos[i]
        obs_image.set_data(obs[index])
    return obs_images


animation = FuncAnimation(fig, update, interval=100)
plt.show()
