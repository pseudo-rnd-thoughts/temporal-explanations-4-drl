
# Temporal Explanations for Deep Reinforcement Learning

> Despite significant progress in deep reinforcement learning across a range of environments, there are still limited tools to understand why agents make decisions. In particular, we consider how certain actions enable an agent to collect rewards or achieve its goals. Understanding this temporal context for actions is critical to explaining an agent’s choices. To date, however, little research has explored such explanations, and those that do depend on domain knowledge. We address this by developing three novel types of local temporal explanations, two of which do not require domain knowledge, and two novel metrics to evaluate agent skills. We conduct a comprehensive user survey of our explanations against two state-of-the-art local non-temporal explanations for Atari environments and find that our explanations are preferred by users 80.7% of the time over the state-of-the-art explanations.

## Example Explanations

The video below is an example contrastive questions from the user survey conducted with an observation from breakout, our novel Plan explanation and on the right a perturbation-based saliency map. All observation / explanations used in the user survey are contained in [user-survey](user-survey) along with the survey [results](user-survey/survey-1-results.xlsx) and [analysis](user-survey/survey-analysis.ipynb).

https://user-images.githubusercontent.com/14953262/235640238-638d8f1a-89e6-437d-83f7-42eeabe88b03.mp4

Click on the following dropdowns to see more examples with all the evaluated explanations mechanisms (Dataset Similarity Explanation, Plan Explanation, Grad-CAM and Perturbation-based Saliency Map).

<details>
<summary>Example observation for Breakout</summary>
<img src="analysis/figs/examples/Breakout/observation.png" alt="Breakout observation">

<details class="inner">
<summary>Dataset Similarity Explanation</summary>

https://user-images.githubusercontent.com/14953262/235640312-9d76126b-1c4f-4a55-b59a-0e35c4d91385.mp4
</details>
<details class="inner">
<summary>Skill Explanation</summary>

https://user-images.githubusercontent.com/14953262/235640340-17f2fbe7-8461-48e2-aa05-661d7190e634.mp4
</details>
<details class="inner">
<summary>Plan Explanation</summary>

https://user-images.githubusercontent.com/14953262/235640597-dd0a79df-ee34-487a-9c30-deec4ae04107.mp4
</details>
<details class="inner">
<summary>Grad-CAM Explanation</summary>
<img src="analysis/figs/examples/Breakout/grad-cam.png" alt="Grad-CAM explanation">
</details>
<details class="inner">
<summary>Perturbation-based Saliency Maps</summary>
<img src="analysis/figs/examples/Breakout/perturbation-based-saliency-map.png" alt="Perturbation-based Saliency map">
</details>
</details>

<details>
<summary>Example observation for Space Invaders</summary>
<img src="analysis/figs/examples/SpaceInvaders/observation.png" alt="Space Invader observation">
<details class="inner">
<summary>Dataset Similarity Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641388-5367a411-dbd1-4db9-8d3b-2893778d6efd.mp4
</details>
<details class="inner">
<summary>Skill Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641418-3886ab15-4574-483b-87aa-93366d374e6e.mp4
</details>
<details class="inner">
<summary>Plan Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641428-1fa21eea-2ac1-4ecc-8f84-379c30d27e6f.mp4
</details>
<details class="inner">
<summary>Grad-CAM Explanation</summary>
<img src="analysis/figs/examples/SpaceInvaders/grad-cam.png" alt="Grad-CAM explanation">
</details>
<details class="inner">
<summary>Perturbation-based Saliency Maps</summary>
<img src="analysis/figs/examples/SpaceInvaders/perturbation-based-saliency-map.png" alt="Perturbation-based Saliency map">
</details>
</details>

<details>
<summary>Example observation for Seaquest</summary>
<img src="analysis/figs/examples/Seaquest/observation.png" alt="Seaquest observation">
<details class="inner">
<summary>Dataset Similarity Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641598-f4f71d22-d164-43ae-b79c-2d40f74d700f.mp4
</details>
<details class="inner">
<summary>Skill Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641607-a6b1fdc7-333f-40ed-a0c1-9e2b2699800f.mp4
</details>
<details class="inner">
<summary>Plan Explanation</summary>

https://user-images.githubusercontent.com/14953262/235641620-38a63ee8-26a5-4ee4-b6ff-54775e02f71b.mp4
</details>
<details class="inner">
<summary>Grad-CAM Explanation</summary>
<img src="analysis/figs/examples/Seaquest/grad-cam.png" alt="Grad-CAM explanation">
</details>
<details class="inner">
<summary>Perturbation-based Saliency Maps</summary>
<img src="analysis/figs/examples/Seaquest/perturbation-based-saliency-map.png" alt="Perturbation-based Saliency map">
</details>
</details>

## User Survey results

Figure 4 in the paper presenting the user ratings for each explanation mechanism across four different questions.

![User rating](analysis/figs/survey/individual-evaluation.png)

Figure 5 in the paper presenting a heatmap of the user preference for each question and between each explanation mechanism. Each grid element is equal to the percentage that the row explanation mechanism was preferred over the column explanation mechanism.

![Comparative Rating](analysis/figs/survey/contrastive-evaluation.png)

All observation explanations shown to the users are provided in [user-survey](user-survey/observation-explanations) with the raw survey [data](user-survey/survey-1-results.xlsx) and analysis [notebook](user-survey/survey-analysis.ipynb).

## Code

Python requirements can be found in [requirement.txt](requirements.txt) and installed with `pip install -r requirements.txt`. Additionally, to use the project might require installing `temporal_explanations_4_drl` using `pip install -e .` in the root directory, no pypi exists currently.

To understand the project structure, we have outlined the purpose of the most important files.
* `temporal_explanations_4_drl/explain.py` - Explanation code for all of our novel explanation, code to save the explanations with the relevant observation (both individually and to compare) along with implementations of Grad-CAM and Perturbation-based Saliency Maps.
* `temporal_explanations_4_drl/skill.py` - Skill instance class and skill alignment and distribution metric implementations
* `temporal_explanations_4_drl/plan.py` - Plan class with methods for computing several metrics across all skills and each skill individually
* `temporal_explanations_4_drl/graying_the_black_box.py` - Implementation of Zahavy et al., 2016 "Graying the black box: Understanding DQNs"
* `datasets/annotate-domain-knowledge.py` - A command line based python script to load pre-defined skills for a set of episode and provide text-based explanations of the purpose for each skill.
* `datasets/hand-label-skills.py` - A command line based python script to hand-label skilled for individual episodes, each observation can be provided an individual skill number between 0 and 9
* `datasets/generate_datasets.py` - A python script to generate datasets for a several environments with options for the size, agent types, etc
* `datasets/discover_skills.py` - A python script using pre-generated datasets to discover agent skills using the algorithm proposed by Zahavy et al., 2016


<style>
details.inner {
  margin: 3%;
}
</style>
