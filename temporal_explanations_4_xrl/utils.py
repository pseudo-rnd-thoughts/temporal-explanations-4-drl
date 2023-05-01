from __future__ import annotations

import os

import cv2
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def create_directory(folder: str):
    """Creates all the subdirectories for the folder

    Args:
        folder: The folder path to create
    """
    if not os.path.exists(folder):
        sub_folders = folder.split("/")
        for pos in range(len(sub_folders)):
            if not os.path.exists("/".join(sub_folders[: pos + 1])):
                os.mkdir("/".join(sub_folders[: pos + 1]))


def load_embedding(file_path: str) -> np.ndarray:
    """Loads the T-SNE dataset from the file path

    Args:
        file_path: The file path of where to find the dataset

    Returns:
        A numpy array for the embedding
    """
    with np.load(file_path, allow_pickle=True) as file:
        return file["embedding"]


def flatten(values: list[list]) -> list:
    """Flattens a list of lists to a list.

    Args:
        values: A double stacked list

    Returns:
        List of values
    """
    return [val for value in values for val in value]


def animate_observations(frames: np.ndarray | list[np.ndarray], filename: str):
    """Animates the observations and saves to the filename.

    Args:
        frames: List of frames
        filename: String for the filename
    """
    if isinstance(frames, np.ndarray):
        frames = [obs for obs in frames]

    assert isinstance(frames, list)
    assert all(isinstance(obs, np.ndarray) for obs in frames)

    if len(frames[0].shape) == 2:
        print("Converting greyscale to RGB")
        frames = [cv2.cvtColor(obs[:, :, -1], cv2.COLOR_GRAY2RGB) for obs in frames]
    assert all(len(obs.shape) == 3 and obs.shape[2] == 3 for obs in frames)

    clip = ImageSequenceClip(frames, fps=12)
    clip.write_videofile(filename)
