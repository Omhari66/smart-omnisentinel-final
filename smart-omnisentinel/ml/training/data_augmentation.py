"""
ml/training/data_augmentation.py
-----------------------------------
Temporal and spatial augmentation strategies for pose feature sequences.
Applied during training to improve generalization to:
  - Varied lighting conditions
  - Camera angles
  - Subject body sizes
  - Action speeds

Each augmentation is a standalone function operating on (window_size, feature_dim)
numpy arrays so they can be composed freely.
"""
from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    window: np.ndarray,
    sigma: float = 0.01,
    keypoint_cols: slice = slice(0, 34),
) -> np.ndarray:
    """
    Add Gaussian noise to keypoint coordinates only.
    Simulates sub-pixel detection noise and calibration error.
    """
    window = window.copy()
    noise = np.random.normal(0, sigma, window[:, keypoint_cols].shape)
    window[:, keypoint_cols] += noise
    window[:, keypoint_cols] = np.clip(window[:, keypoint_cols], 0.0, 1.0)
    return window


def horizontal_flip(window: np.ndarray) -> np.ndarray:
    """
    Mirror all X coordinates (every other column in [0:34]).
    Doubles the dataset by flipping left/right orientation.
    Note: flips X coordinates of keypoints only (not confidences/motion).
    """
    window = window.copy()
    # X coords are at even indices in the flattened keypoints [0, 2, 4, ..., 32]
    window[:, 0:34:2] = 1.0 - window[:, 0:34:2]
    return window


def temporal_jitter(window: np.ndarray, drop_prob: float = 0.3) -> np.ndarray:
    """
    Randomly drop one frame and repeat a neighbor.
    Simulates variable playback speed and frame drops.
    """
    window = window.copy()
    if np.random.rand() >= drop_prob:
        return window
    n = len(window)
    drop_idx = np.random.randint(1, n - 1)
    window = np.delete(window, drop_idx, axis=0)
    repeat_idx = max(0, drop_idx - 1)
    window = np.insert(window, drop_idx, window[repeat_idx], axis=0)
    return window


def temporal_reverse(window: np.ndarray, prob: float = 0.15) -> np.ndarray:
    """
    Reverse the temporal order of the sequence.
    Falls should look like getting up; violence is still violence reversed.
    Apply sparingly — reversal changes the action semantics.
    """
    if np.random.rand() < prob:
        return window[::-1].copy()
    return window


def scale_motion_features(
    window: np.ndarray,
    scale_range: tuple = (0.7, 1.3),
    motion_cols: slice = slice(51, 55),
) -> np.ndarray:
    """
    Scale motion magnitude features by a random factor.
    Simulates cameras at different heights/zoom levels.
    """
    window = window.copy()
    scale = np.random.uniform(*scale_range)
    window[:, motion_cols] = np.clip(window[:, motion_cols] * scale, 0.0, 1.0)
    return window


def occlude_keypoints(
    window: np.ndarray,
    max_occlude: int = 4,
    keypoint_range: tuple = (0, 17),
) -> np.ndarray:
    """
    Zero out a random subset of keypoints (simulate occlusion).
    Sets both coordinates AND confidence to 0 for occluded keypoints.
    """
    window = window.copy()
    n_occlude = np.random.randint(0, max_occlude + 1)
    if n_occlude == 0:
        return window
    keypoint_indices = np.random.choice(
        range(*keypoint_range), size=n_occlude, replace=False
    )
    for kp_idx in keypoint_indices:
        # Zero out x, y coords (at 2*kp_idx and 2*kp_idx+1)
        window[:, 2 * kp_idx] = 0.0
        window[:, 2 * kp_idx + 1] = 0.0
        # Zero out confidence (at 34 + kp_idx)
        if 34 + kp_idx < window.shape[1]:
            window[:, 34 + kp_idx] = 0.0
    return window


def compose(*augmentations):
    """
    Compose multiple augmentation functions into a single transform.

    Usage:
        transform = compose(
            add_gaussian_noise,
            horizontal_flip,
            temporal_jitter,
        )
        augmented = transform(window)
    """
    def apply(window: np.ndarray) -> np.ndarray:
        for aug in augmentations:
            window = aug(window)
        return window
    return apply


# Default training augmentation pipeline
DEFAULT_TRAIN_AUGMENTATION = compose(
    lambda w: add_gaussian_noise(w, sigma=0.01),
    lambda w: horizontal_flip(w) if np.random.rand() < 0.5 else w,
    lambda w: temporal_jitter(w, drop_prob=0.3),
    lambda w: scale_motion_features(w),
    lambda w: occlude_keypoints(w, max_occlude=3),
)
