import argparse
import json
import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib._color_data import TABLEAU_COLORS
from matplotlib.patches import Rectangle

"""
Must be run from the terminal

 - Generates a new file in the agent-env dataset folder called, "hand-cluster.json"
 - Contains a dictionary of trajectory's and skill labels
 - If data already exists then the old skill labels and loaded and shown to the user
 - Otherwise, -1 skill labels are generated for every observation
 - A window shows the episode observation with arrow keys used to move through the episode
"""

POSSIBLE_CLUSTERS = string.digits
COLOURS = list(TABLEAU_COLORS.keys())
COLOURS.remove("tab:gray")

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--agent_name", default="dqn_adam_mse")
parser.add_argument("--env_name", default="Breakout")
parser.add_argument("--root", default="datasets/")
parser.add_argument("--trajectory", default=0)
args = parser.parse_args()


# Load trajectory information
with np.load(
    f"{args.root}{args.agent_name}-{args.env_name}/trajectories/trajectory-{args.trajectory}.npz"
) as file:
    all_human_obs = file["human_obs"]
    trajectory_length = int(file["length"])

if os.path.exists(
    f"{args.root}{args.agent_name}-{args.env_name}/hand-labelled-skills.json"
):
    with open(
        f"{args.root}/{args.agent_name}-{args.env_name}/hand-labelled-skills.json"
    ) as file:
        data = json.load(file)
        prior_skill_labels = data["skill_labels"]
        skill_meanings = data["skill_meanings"]

        if args.trajectory in prior_skill_labels:
            print(
                f"Trajectory {args.trajectory} already exists in prior skill labels, {prior_skill_labels.keys()}",
                file=sys.stderr,
            )

            skill_labels = prior_skill_labels[args.trajectory]
        else:
            skill_labels = np.full(trajectory_length, -1, dtype=np.int32)
else:
    prior_skill_labels = {}
    skill_labels = np.full(trajectory_length, -1, dtype=np.int32)
    skill_meanings = []
    while (meaning := input(f"Skill {len(skill_meanings)} meaning: ")) != "":
        skill_meanings.append(meaning)

current_label = -1
trajectory_position = 0


# Matplotlib figure
fig = plt.figure(figsize=(12, 5))
fig.suptitle(f"Current cluster label: {current_label}")
gs = fig.add_gridspec(ncols=4, nrows=2, height_ratios=[1, 0.1])

previous_obs_ax = fig.add_subplot(gs[0, 0])
previous_obs_ax.set_title("Previous time step observation")
previous_obs_plt = previous_obs_ax.imshow(np.full((210, 180, 3), [0, 0, 0]))
previous_obs_ax.axis("off")

current_obs_ax = fig.add_subplot(gs[0, 1])
current_obs_ax.set_title("Current time step observation")
current_obs_plt = current_obs_ax.imshow(all_human_obs[0])
current_obs_ax.axis("off")

next_obs_ax = fig.add_subplot(gs[0, 2])
next_obs_ax.set_title("Next time step observation")
next_obs_plt = next_obs_ax.imshow(all_human_obs[1])
next_obs_ax.axis("off")

info_ax = fig.add_subplot(gs[0, 3])
info_ax.set_title("Cluster Info")
for i in range(len(skill_meanings)):
    info_ax.text(0, 0.9 - 0.1 * i, f"{i}.")
    info_ax.add_patch(
        Rectangle((0.1, 0.9 - 0.1 * i), width=0.15, height=0.04, facecolor=COLOURS[i])
    )
    info_ax.text(0.3, 0.9 - 0.1 * i, skill_meanings[i])
info_ax.axis("off")


skill_ax = fig.add_subplot(gs[1, :])
skill_ax.set_yticklabels([])
skill_ax.axvline(x=20, linewidth=1)
initial_colours = ["w"] * 20 + [
    "tab:gray" if skill == -1 else COLOURS[skill] for skill in skill_labels[:20]
]
skill_plt = skill_ax.bar(np.arange(40), np.ones(40), color=initial_colours)
skill_ax.set_xticks(np.linspace(0, 40, 9, dtype=np.int32))
skill_ax.set_xticklabels(np.linspace(-20, 20, 9, dtype=np.int32))
plt.tight_layout()


def update_figure():
    fig.suptitle(f"Current label: {current_label}")

    # Update the previous, current and next observation
    if trajectory_position > 0:
        previous_obs_plt.set_data(all_human_obs[trajectory_position - 1])
    else:
        previous_obs_plt.set_data(np.full((210, 180, 3), [0, 0, 0]))

    current_obs_plt.set_data(all_human_obs[trajectory_position])

    if trajectory_position <= trajectory_length:
        next_obs_plt.set_data(all_human_obs[trajectory_position + 1])
    else:
        next_obs_plt.set_data(np.full((210, 180, 3), [0, 0, 0]))

    # Update the skills
    skill_ax.set_xticklabels(
        np.linspace(
            trajectory_position - 20, trajectory_position + 20, 9, dtype=np.int32
        )
    )
    for index, rect in zip(
        range(trajectory_position - 20, trajectory_position + 20), skill_plt
    ):
        if index < 0 or index >= trajectory_length:
            rect.set_color("w")
        elif skill_labels[index] == -1:
            rect.set_color("tab:gray")
        else:
            rect.set_color(COLOURS[skill_labels[index]])


def on_keypress(event):
    """Update the figure on keypress."""
    global trajectory_position, current_label

    if event.key == "left":
        if trajectory_position > 0:
            trajectory_position -= 1
            if current_label != -1:
                skill_labels[trajectory_position] = current_label
            update_figure()
            fig.canvas.draw_idle()
        else:
            print("At the beginning, cannot go back.")
    elif event.key == "right":
        if trajectory_position < trajectory_length - 1:
            trajectory_position += 1
            if current_label != -1:
                skill_labels[trajectory_position] = current_label
            update_figure()
            fig.canvas.draw_idle()
        else:
            print("At the end, cannot go forward")
    elif event.key in " ":
        current_label = -1
        update_figure()
    elif event.key in POSSIBLE_CLUSTERS:
        current_label = POSSIBLE_CLUSTERS.index(event.key)
        skill_labels[trajectory_position] = current_label
        update_figure()
    elif event.key == "escape":
        print("Save labels")
        prior_skill_labels[args.trajectory] = skill_labels.tolist()
        with open(
            f"{args.root}{args.agent_name}-{args.env_name}/hand-labelled-skills.json",
            "w",
        ) as file:
            json.dump(
                {
                    "skill_labels": prior_skill_labels,
                    "trajectory_folder": f"{args.root}{args.agent_name}-{args.env_name}/trajectories/",
                    "skill_meanings": skill_meanings,
                    "n_clusters": len(skill_meanings),
                },
                file,
            )
        plt.close()
    else:
        print(f"Unknown key: '{event.key}'")


fig.canvas.mpl_connect("key_press_event", on_keypress)
plt.show()
