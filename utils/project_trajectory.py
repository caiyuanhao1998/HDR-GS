"""Periodic project-page camera paths with constant winding and C2 reversals."""
import numpy as np


def project_page_cameras(preset, fps=None):
    fps = int(preset['fps'] if fps is None else fps)
    duration = float(preset['duration_seconds'])
    frames = int(round(duration * fps))
    if fps <= 0 or frames < 4 or frames % 2:
        raise ValueError('Choose a positive fps with an even number of video frames')
    blend = float(preset['turn_blend_seconds']) / (duration / 2)
    if not 0 < blend < 0.5:
        raise ValueError('Turn blend must be shorter than one quarter of the video')
    cycle = np.arange(frames, dtype=np.float64) / frames
    progress = 1 - np.abs(2 * cycle - 1)
    norm = 1 - 2 * blend + 4 * blend / np.pi
    low = lambda x: 2 * blend / np.pi * (1 - np.cos(np.pi * x / (2 * blend)))
    axial = np.where(progress < blend, low(progress) / norm,
                     np.where(progress > 1 - blend, 1 - low(1 - progress) / norm,
                              (progress - blend + 2 * blend / np.pi) / norm))
    angle = 4 * np.pi * float(preset['turns_per_leg']) * cycle
    start = np.asarray(preset['camera_to_world_start'], dtype=np.float64)
    if start.shape != (4, 4) or not np.isfinite(start).all():
        raise ValueError('camera_to_world_start must be a finite 4 x 4 matrix')
    offsets = np.stack([float(preset['lateral_radius_x']) * (np.cos(angle) - 1),
                        float(preset['lateral_radius_y']) * np.sin(angle),
                        float(preset['forward_travel']) * axial], axis=1)
    cameras = np.repeat(start[None], frames, axis=0)
    cameras[:, :3, 3] = start[:3, 3] + offsets @ start[:3, :3].T
    return cameras, cycle * duration
