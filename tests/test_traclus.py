import matplotlib.pyplot as plt
import numpy as np

from temporal_explanations_4_xrl.traclus import TraClus

embeddings = np.array(
    [[3, 0.2], [2, -0.1], [1, -0.2], [0, 0], [-0.2, 0.5], [0, 1], [1, 1.15], [2, 1]]
)

# minimum_description_length(embeddings[:3])
# minimum_description_length(embeddings[:4])


def plot():
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ax.plot(embeddings[:, 0], embeddings[:, 1])
    ax.scatter(embeddings[[0, -1], 0], embeddings[[0, -1], 1], c="red")

    traclus = TraClus()
    characteristic_points = traclus.approximate_trajectory_partitioning(embeddings)
    ax.plot(characteristic_points[:, 0], characteristic_points[:, 1], ls="-")

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def test_example():
    s_i = np.array([0, 0])
    e_i = np.array([2, 0])
    s_j = np.array([0.5, 1])
    e_j = np.array([1.5, 1.5])

    l_i = e_i - s_j
    l_j = e_j - s_j
    print(f"{l_i=}, {l_j=}")

    se_i = e_i - s_i
    se_j = e_j - s_j

    se_i_length = np.linalg.norm(se_i)
    se_j_length = np.linalg.norm(se_j)
    print(f"{se_i_length=}, {se_j_length}")

    u_1 = (s_j - s_i) * se_i / np.square(se_i_length)
    u_2 = (e_j - s_i) * se_i / np.square(se_i_length)

    p_s = s_i + u_1 * se_i
    p_e = s_i + u_2 * se_i
    print(f"{p_s=}, {p_e=}")

    l1_perpendicular = np.linalg.norm(s_j - p_s)
    l2_perpendicular = np.linalg.norm(e_j - p_e)
    print(f"{l1_perpendicular=}")
    print(f"{l2_perpendicular=}")

    l1_parallel = np.linalg.norm(s_i - p_s)
    l2_parallel = np.linalg.norm(e_i - p_e)
    print(f"{l1_parallel=}")
    print(f"{l2_parallel=}")

    cos_theta = (se_i @ se_j) / (se_i_length * se_j_length)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    print(f"{cos_theta=}, {theta=}, {sin_theta=}")

    perpendicular_distance = (
        np.square(l1_perpendicular) + np.square(l2_perpendicular)
    ) / (l1_perpendicular + l2_perpendicular)
    parallel_distance = np.min([l1_parallel, l2_parallel])
    angle_distance = np.linalg.norm(se_j) * (
        np.degrees(np.sin(theta)) if theta <= np.pi / 2 else 1
    )

    print(f"{perpendicular_distance=}")
    print(f"{parallel_distance=}")
    print(f"{angle_distance=}")


def test_minimum_description_length():
    traclus = TraClus()
    traclus.minimum_description_length(embeddings[:5])


def test_characteristic_points():
    traclus = TraClus()
    characteristic_points = traclus.approximate_trajectory_partitioning(embeddings)
    assert len(characteristic_points) == 4
    assert np.all(characteristic_points == np.array([[3, 0.2], [0, 0], [0, 1], [2, 1]]))


if __name__ == "__main__":
    plot()
