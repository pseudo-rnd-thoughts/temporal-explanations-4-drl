import numpy as np
from tqdm import tqdm


class TraClus:
    """Implementation of "Trajectory Clustering: A Partition-and-Group Framework" by JC Lee et al., 2007"""

    def __init__(
        self,
        epsilon: float = 1,
        min_lines: int = 5,
        distance_weighting: np.ndarray = np.ones(3),
    ):
        """Constructor

        Args:
            epsilon: Epsilon variable from Lee et al., 2007
            min_lines: minLine variable from Lee et al., 2007
            distance_weighting: The distance weighting
        """
        self.epsilon = epsilon
        self.min_lines = min_lines
        self.distance_weighting = distance_weighting

        self.line_segment_distance_matrix = None

    def fit(self, trajectory_embeddings: list[np.ndarray]) -> np.ndarray:
        total_embeddings = sum(len(traj) for traj in trajectory_embeddings)

        # Convert the embeddings to line segments (algo 1)
        line_segments, segment_trajectory = self.embeddings_to_line_segments(
            trajectory_embeddings
        )
        self.line_segment_distance_matrix = self.compute_line_segment_distance_matrix(
            line_segments, self.distance_weighting
        )

        # Cluster the line segments (algo 2)
        line_segment_cluster = self.cluster_line_segments(
            line_segments, segment_trajectory, self.line_segment_distance_matrix
        )

        # Convert the line segment cluster to embedding clusters
        embedding_clusters = self.segment_cluster_to_embedding_cluster(
            trajectory_embeddings, line_segments, line_segment_cluster
        )
        assert len(embedding_clusters) == total_embeddings

        return embedding_clusters

    def fit_parameter_search(
        self,
        trajectory_embeddings: list[np.ndarray],
        initial_epsilon=1,
        initial_min_lines=5,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        total_embeddings = sum(len(traj) for traj in trajectory_embeddings)

        # Convert the embeddings to line segments (algo 1)
        line_segments, segment_trajectory = self.embeddings_to_line_segments(
            trajectory_embeddings
        )
        line_segment_distance_matrix = self.compute_line_segment_distance_matrix(
            line_segments, self.distance_weighting
        )

        # Compute the optimal epsilon and min-lines
        # todo - write the gradient descent for this
        segments_within_epsilon = np.sum(
            line_segment_distance_matrix < initial_epsilon, axis=1
        )
        probability = segments_within_epsilon / np.sum(segments_within_epsilon)
        info = -probability * np.log2(probability)

        optimal_epsilon = min(info)
        optimal_min_lines = np.mean(segments_within_epsilon)

        # Cluster the line segments (algo 2)
        line_segment_cluster = self.cluster_line_segments(
            line_segments, segment_trajectory, line_segment_distance_matrix
        )

        # Convert the line segment cluster to embedding clusters
        embedding_clusters = self.segment_cluster_to_embedding_cluster(
            trajectory_embeddings, line_segments, line_segment_cluster
        )
        assert len(embedding_clusters) == total_embeddings

        return embedding_clusters, optimal_epsilon, optimal_min_lines

    def embeddings_to_line_segments(
        self, trajectory_embeddings: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the trajectory embeddings to line segments and segment trajectory"""
        # expected shape == (trajectory length, embedding features)
        assert all(
            len(traj.shape) == 2 for traj in trajectory_embeddings
        ), trajectory_embeddings[0].shape
        embedding_features = trajectory_embeddings[0].shape[1]

        # Convert the embeddings to line segments
        line_segments, segment_trajectory = [], []
        for num, trajectory in enumerate(trajectory_embeddings):
            characteristic_points = self.approximate_trajectory_partitioning(trajectory)

            line_segments.append(
                np.moveaxis(
                    np.array([characteristic_points[:-1], characteristic_points[1:]]),
                    0,
                    1,
                )
            )
            segment_trajectory.append(np.full(len(characteristic_points) - 1, num))

        # Expected shape = (number of line segments, 2 (start and end), embedding features)
        line_segments = np.concatenate(line_segments)
        segment_trajectory = np.concatenate(segment_trajectory)
        assert len(line_segments) == len(
            segment_trajectory
        ), f"{line_segments.shape=}, {segment_trajectory.shape=}"
        assert np.all(segment_trajectory) >= 0
        assert len(segment_trajectory.shape) == 1
        assert len(line_segments.shape) == 3  # (num segments, 2 (start, end), features)
        assert line_segments.shape[1] == 2
        assert line_segments.shape[2] == embedding_features

        # Segment trajectory is a np.ndarray of which trajectory each segment originates from.
        return line_segments, segment_trajectory

    def compute_line_segment_distance_matrix(
        self, line_segments: np.ndarray, distance_weighting: np.ndarray
    ) -> np.ndarray:
        print(f"{line_segments.shape=}")
        line_segment_distance_matrix = np.zeros(
            (len(line_segments), len(line_segments))
        )

        # step 0 - compute line segment distance matrix
        line_segment_lengths = np.linalg.norm(
            line_segments[:, 0, :] - line_segments[:, 1, :], axis=1
        )
        print(f"{line_segment_lengths.shape=}")
        sorted_segment_lengths = np.argsort(line_segment_lengths)
        # print(f'{sorted_segment_lengths.shape=}')
        for pos in tqdm(range(len(line_segments) - 1)):
            se_i = line_segments[sorted_segment_lengths[pos]]
            se_j = line_segments[sorted_segment_lengths[pos + 1 :]]
            # print(f'{se_i.shape=}, {se_j.shape=}')
            distances = self.line_segment_distance(
                se_i[0], se_i[1], se_j[:, 0], se_j[:, 1]
            )
            # print(f'{distance_weighting.shape=}, {distance_weighting}')
            weighted_distances = np.sum(
                distances, axis=0
            )  # todo add distance_weighting
            # print(f'{weighted_distances.shape=}')

            # print(f'{sorted_segment_lengths[pos + 1:].shape=}')
            # print(f'{line_segment_distance_matrix[sorted_segment_lengths[pos]].shape=}')
            # print(f'{line_segment_distance_matrix[sorted_segment_lengths[pos], sorted_segment_lengths[pos + 1:]].shape=}')

            line_segment_distance_matrix[
                sorted_segment_lengths[pos], sorted_segment_lengths[pos + 1 :]
            ] = weighted_distances
            line_segment_distance_matrix[
                sorted_segment_lengths[pos + 1 :], sorted_segment_lengths[pos]
            ] = weighted_distances

        return line_segment_distance_matrix

    def segment_cluster_to_embedding_cluster(
        self,
        trajectory_embeddings: list[np.ndarray],
        line_segments: np.ndarray,
        line_segment_clusters: np.ndarray,
    ) -> np.ndarray:
        # Convert the line segment (with cluster info) to an embedding (with cluster info)
        embedding_clusters = np.zeros(
            sum(len(traj) for traj in trajectory_embeddings), dtype=np.int32
        )
        pos = 0
        for segment, cluster in zip(line_segments, line_segment_clusters):
            embedding_clusters[pos : pos + len(segment)] = cluster
            pos += len(segment) - 1  # todo test

        return embedding_clusters

    def cluster_line_segments(
        self,
        line_segments: np.ndarray,
        segment_trajectory: np.ndarray,
        line_segment_distance_matrix: np.ndarray,
    ):
        cluster_id = 0

        line_segments_type = np.zeros(
            len(line_segments), dtype=np.int8
        )  # 0 -> unclassified, 1 -> noise, 2 -> classified
        line_segment_cluster = np.full(len(line_segments), -1, dtype=np.int32)

        # step 1
        for pos in range(len(line_segments)):
            if line_segments_type[pos] == 0:
                close_segments = np.where(
                    line_segment_distance_matrix[pos] < self.epsilon
                )
                if len(close_segments) >= self.min_lines:
                    line_segment_cluster[close_segments] = cluster_id
                    queue = [i for i in close_segments if i != pos]

                    # step 2
                    self.expand_cluster(
                        queue,
                        cluster_id,
                        line_segment_distance_matrix,
                        line_segments_type,
                        line_segment_cluster,
                    )
                    cluster_id += 1
                else:
                    line_segments_type[pos] = 1

        # all line segments are either noise or classified
        assert np.all(line_segments_type != 0)
        # All segment labels as noise are also assigned to the noise cluster
        assert np.all((line_segment_cluster == -1) == (line_segments_type == 1))

        # Step 3
        available_cluster_id = 0
        for cluster in range(cluster_id):
            cluster_instances = np.where(line_segment_cluster == cluster)
            unique_trajectories = np.unique(segment_trajectory[cluster_instances])
            if unique_trajectories > self.min_lines:
                line_segment_cluster[cluster_instances] = available_cluster_id
                available_cluster_id += 1
            else:
                # Label as a noise cluster
                line_segment_cluster[cluster_instances] = -1
        noise_instances = np.where(line_segment_cluster == -1)
        line_segment_cluster[noise_instances] = available_cluster_id

        return line_segment_cluster

    def line_segment_distance(
        self,
        s_i: np.ndarray,
        e_i: np.ndarray,
        s_j: np.ndarray,
        e_j: np.ndarray,
        delta=0.0001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the line segment distance where (s_j, e_j) can be a vector of points

        Args:
            s_i: The starting point of the longer line
            e_i: The ending point of the longer line
            s_j: A vector of starting points of lines shorter than |se_i|
            e_j: A vector of ending point of lines shorter than |se_i|
            delta: Tiny amount to prevent divide by zero

        Returns:
            A tuple of perpendicular_distance, parallel_distance, angle_distance for each of the lines
        """
        assert len(s_i.shape) == len(e_i.shape) == 1 and s_i.shape == e_i.shape
        assert len(s_j.shape) == len(e_j.shape) == 2 and s_j.shape == e_j.shape
        assert s_j.shape[0] >= 1 and e_j.shape[0] >= 1, f"{s_j.shape=}, {e_j.shape=}"
        assert s_j.shape[1] == s_i.shape[0], f"{s_j.shape=}, {s_i.shape=}"

        num_segments = len(s_j)
        features = s_i.shape[0]
        segment_shape = (num_segments, features)

        se_i = e_i - s_i
        se_i_length = np.linalg.norm(se_i) + delta

        # Computes the perpendicular distance and angle distance for all intermediate line segments
        #   perpendicular distances
        u_1 = (s_j - s_i) * se_i / np.square(se_i_length)
        u_2 = (e_j - s_i) * se_i / np.square(se_i_length)
        assert u_1.shape == segment_shape, f"{u_1.shape=}, {segment_shape=}"
        assert u_2.shape == segment_shape, f"{u_2.shape=}, {segment_shape=}"

        p_s = s_i + u_1 * se_i
        p_e = s_i + u_2 * se_i
        assert p_s.shape == segment_shape, f"{p_s.shape=}, {segment_shape=}"
        assert p_e.shape == segment_shape, f"{p_e.shape=}, {segment_shape=}"

        l_1 = np.linalg.norm(s_j - p_s, axis=1) + delta
        l_2 = np.linalg.norm(e_j - p_e, axis=1) + delta
        assert l_1.shape == (num_segments,), f"{l_1.shape=}, {num_segments=}"
        assert l_2.shape == (num_segments,), f"{l_2.shape=}, {num_segments=}"

        perpendicular_distance = (np.square(l_1) + np.square(l_2)) / (l_1 + l_2)
        assert perpendicular_distance.shape == (
            num_segments,
        ), f"{perpendicular_distance=}, {num_segments=}"

        # === Parallel distance
        l_1 = np.linalg.norm(s_i - p_s, axis=1) + delta
        l_2 = np.linalg.norm(e_i - p_e, axis=1) + delta

        parallel_distance = np.min(np.array([l_1, l_2]), axis=0)
        assert parallel_distance.shape == (
            num_segments,
        ), f"{parallel_distance=}, {num_segments=}"

        # ===  Angle distance
        se_j = e_j - s_j
        assert se_j.shape == segment_shape
        se_j_length = np.linalg.norm(se_j, axis=1) + delta
        assert se_j_length.shape == (
            num_segments,
        ), f"{se_j_length.shape=}, {num_segments=}"

        cos_theta = np.clip(
            (se_j @ se_i) / (se_i_length * se_j_length), a_min=-1.0, a_max=1.0
        )
        assert cos_theta.shape == (
            num_segments,
        ), f"{cos_theta.shape=}, {num_segments=}"
        theta = np.arccos(cos_theta)
        assert theta.shape == (num_segments,), f"{theta.shape=}, {num_segments=}"
        angle_distance = se_j_length * np.where(
            theta <= np.pi / 2, np.degrees(np.sin(theta)), np.ones(num_segments)
        )
        assert angle_distance.shape == (
            num_segments,
        ), f"{angle_distance=}, {num_segments=}"

        return perpendicular_distance, parallel_distance, angle_distance

    def minimum_description_length(
        self, embeddings: np.ndarray, delta: float = 0.0001
    ) -> tuple[float, float, float]:
        """The minimum description length (MDL_par) equation in the TraClus paper.

        Args:
            embeddings: A list of embeddings
            delta: A minimal value to prevent log2(0)

        Returns:
            A tuple of length info, perpendicular distance info gain and angle distance info gain
        """
        s_i, e_i = embeddings[0], embeddings[-1]
        s_j, e_j = embeddings[:-1], embeddings[1:]

        perpendicular_distance, _, angle_distance = self.line_segment_distance(
            s_i, e_i, s_j, e_j
        )

        # Compute info gain
        length_info = np.log2(np.linalg.norm(e_i - s_i) + delta)
        perpendicular_info_gain = np.log2(np.sum(perpendicular_distance) + delta)
        angle_info_gain = np.log2(np.sum(angle_distance) + delta)

        return length_info, perpendicular_info_gain, angle_info_gain

    def approximate_trajectory_partitioning(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Implementation of the approximate trajectory partitioning (algorithm 1)

        Args:
            embeddings: The embedding for the whole trajectory

        Returns:
            Numpy array of characteristic points
        """
        initial_index, length = 1, 3
        characteristic_points = [embeddings[0]]
        while initial_index + length < len(embeddings):
            final_index = initial_index + length

            # The mdl_par
            (
                mdl_length_info,
                mdl_perpendicular_info,
                mdl_angle_info,
            ) = self.minimum_description_length(embeddings[initial_index:final_index])
            mdl_info = 0.3 * (mdl_length_info + mdl_perpendicular_info + mdl_angle_info)

            # The mdl_nopar
            embedding_vectors = (
                embeddings[initial_index + 1 : final_index]
                - embeddings[initial_index : final_index - 1]
            )
            embedding_distance = (
                np.sum(np.linalg.norm(embedding_vectors, axis=1)) + 0.000001
            )
            embedding_length_info = np.log2(embedding_distance)

            if mdl_info > embedding_length_info:
                characteristic_points.append(embeddings[final_index - 1])
                initial_index, length = final_index - 1, 2
            else:
                length += 1

        characteristic_points.append(embeddings[-1])
        return np.array(characteristic_points)

    def expand_cluster(
        self,
        queue: list[np.ndarray],
        cluster_id: int,
        line_segment_distances: np.ndarray,
        line_segments_type: np.ndarray,
        line_segment_cluster: np.ndarray,
    ):
        while len(queue):
            m = queue.pop(0)
            close_segments = np.where(line_segment_distances[m] < self.epsilon)[0]
            if len(close_segments) >= self.min_lines:
                for x in close_segments:
                    if line_segments_type[x] != 2:  # is unclassified or noise
                        line_segment_cluster[x] = cluster_id
                    if line_segments_type[x] == 0:
                        queue.append(x)

    def save(self, filename, extra_info=None):
        np.savez_compressed(
            filename,
            line_segment_distance_matrix=self.line_segment_distance_matrix,
            epsilon=self.epsilon,
            min_lines=self.min_lines,
            extra_info=extra_info,
        )
