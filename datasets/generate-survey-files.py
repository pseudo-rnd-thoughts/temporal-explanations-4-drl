import itertools
import json
import os
import shutil
from typing import Optional

import numpy as np

SURVEY_ENVS = ["Breakout", "SpaceInvaders", "Seaquest"]

EXPLANATION_ORDER = [
    ("dataset", "mp4"),
    ("skill", "mp4"),
    ("plan", "mp4"),
    ("gradcam", "png"),
    ("perturbation", "png"),
]
NUM_EXPLANATIONS = 2


def old_contrastive_generation_method():
    # Contrastive explanation
    survey_1_filenames = []
    survey_2_filenames = []
    for env in SURVEY_ENVS:
        survey_1_filenames += [
            f"dqn_adam_mse-{env}/explanation-2/{filename}"
            for filename in sorted(os.listdir(f"dqn_adam_mse-{env}/explanation-2"))
            if "contrast" in filename
        ]

        survey_2_filenames += [
            f"dqn_adam_mse-{env}/explanation-3/{filename}"
            for filename in sorted(os.listdir(f"dqn_adam_mse-{env}/explanation-3"))
            if "contrast" in filename
        ]

    np.random.seed(123)
    np.random.shuffle(survey_1_filenames)
    np.random.shuffle(survey_2_filenames)

    data = []
    for filename in survey_1_filenames + survey_2_filenames:
        elements = filename.split(".")[0].split("-")
        explanation_1 = elements[-2]
        explanation_2 = elements[-1]
        data.append(
            {
                "filename": filename,
                "explanation_1": explanation_1,
                "explanation_2": explanation_2,
            }
        )

    with open("survey/contrastive.json", "w") as file:
        json.dump(data, file)

    for i, filename in enumerate(survey_1_filenames + survey_2_filenames):
        _, ending = filename.split(".")
        shutil.copy(filename, f"survey/contrastive-{i}.{ending}")


if __name__ == "__main__":
    if not os.path.exists("survey"):
        os.mkdir("survey")

    # Individual explanations
    for pos, (explanation, file_type) in enumerate(EXPLANATION_ORDER):
        for env_name in SURVEY_ENVS:
            for num in range(NUM_EXPLANATIONS):
                if not os.path.exists(f"survey/algorithm-{pos}-{env_name}-{num}"):
                    shutil.copy(
                        f"dqn_adam_mse-{env_name}/explanation-{num}/dqn_adam_mse-{env_name}-{num}-{explanation}-explanation.{file_type}",
                        f"survey/algorithm-{pos}-{env_name}-{num}.{file_type}",
                    )

    # Generate the contrastive explanations
    np.random.seed(123)

    num_possible_explanations = 4

    contrastive_explanations = list(
        itertools.combinations(
            ("dataset", "skill", "plan", "gradcam", "perturbation"), 2
        )
    )
    contrastive_explanations.remove(("skill", "plan"))
    assert len(contrastive_explanations) == 9
    np.random.shuffle(contrastive_explanations)

    possible_explanations = [
        (env, num) for env in SURVEY_ENVS for num in range(num_possible_explanations)
    ]

    data: list[Optional[dict]] = [
        None for _ in range(len(contrastive_explanations) * num_possible_explanations)
    ]

    for j in range(4):
        # np.random.shuffle(contrastive_explanations)
        for i, (explanation_1, explanation_2) in enumerate(contrastive_explanations):
            if {explanation_1, explanation_2} == {"gradcam", "perturbation"}:
                file_type = "png"
            else:
                file_type = "mp4"
            env_source, source_num = possible_explanations[
                np.random.choice(len(possible_explanations))
            ]
            contrastive_file_number = j * len(contrastive_explanations) + i

            filename = f"dqn_adam_mse-{env_source}/explanation-{source_num}/dqn_adam_mse-{env_source}-{source_num}-contrast-{explanation_1}-{explanation_2}.{file_type}"
            if os.path.exists(filename):
                data[contrastive_file_number] = {
                    "filename": filename,
                    "environment": env_source,
                    "explanation_num": source_num,
                    "explanation_1": explanation_1,
                    "explanation_2": explanation_2,
                    "contrastive-file-number": contrastive_file_number,
                }
            else:
                filename = f"dqn_adam_mse-{env_source}/explanation-{source_num}/dqn_adam_mse-{env_source}-{source_num}-contrast-{explanation_2}-{explanation_1}.{file_type}"
                data[contrastive_file_number] = {
                    "filename": filename,
                    "environment": env_source,
                    "explanation_num": source_num,
                    "explanation_1": explanation_2,
                    "explanation_2": explanation_1,
                    "contrastive-file-number": contrastive_file_number,
                }

            print(filename, f"survey/contrastive-{contrastive_file_number}.{file_type}")
            shutil.copy(
                filename, f"survey/contrastive-{contrastive_file_number}.{file_type}"
            )
        print()

    with open("survey/contrastive.json", "w") as file:
        json.dump(data, file)
