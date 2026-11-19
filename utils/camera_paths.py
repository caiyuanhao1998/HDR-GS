"""Portable, forward-first camera paths in OpenCV camera-to-world coordinates."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from utils.project_trajectory import project_page_cameras

ROOT = Path(__file__).resolve().parents[1]


def model_candidates(root, scene):
    """Find the published checkpoint layout and existing local scene directories."""
    entry = next(item for item in scene_registry() if item['id'] == scene)
    published = entry['checkpoint_directory']
    legacy = entry.get('legacy_checkpoint_directory', entry['model_directory'])
    directories = dict.fromkeys([published, legacy, scene, entry['model_directory']])
    return [Path(root) / directory for directory in directories]


def scene_registry():
    return json.loads((ROOT / 'config/scenes.json').read_text())


def scene_preset(scene):
    entry = next((s for s in scene_registry() if s['id'] == scene), None)
    if entry is None:
        raise ValueError('Unknown scene: {}'.format(scene))
    return json.loads((ROOT / 'config/trajectories' / entry['trajectory']).read_text())


def retime_preset(preset, duration_seconds, fps):
    """Change playback speed and sampling without changing the spatial path."""
    result = copy.deepcopy(preset)
    duration = float(duration_seconds)
    original = float(result['duration_seconds'])
    if not np.isfinite(duration) or duration <= 0 or original <= 0:
        raise ValueError('Camera duration must be positive')
    ratio = duration / original
    result['duration_seconds'] = duration
    result['fps'] = int(fps)
    if 'turn_blend_seconds' in result:
        result['turn_blend_seconds'] *= ratio
    for keyframe in result.get('keyframes', []):
        keyframe['time_seconds'] *= ratio
    build_camera_path(result)
    return result


def validate_pose_matrices(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError('Camera poses must be an N x 4 x 4 array with at least three poses')
    if not np.isfinite(poses).all():
        raise ValueError('Camera poses must be finite')
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Every pose must end in [0, 0, 0, 1]')
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4):
        raise ValueError('Camera rotation matrices must be orthonormal')
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError('Camera rotations must be right handed')
    return poses


def validate_forward_return(poses):
    """Require one outbound leg followed by a return to the same camera pose."""
    poses = validate_pose_matrices(poses)
    forward = poses[0, :3, 2]
    centers = poses[:, :3, 3]
    depth = (centers - centers[0]) @ forward
    tolerance = max(float(np.ptp(centers, axis=0).max()) * 1e-5, 1e-6)
    peak = int(np.argmax(depth))
    if peak == 0 or peak == len(depth) - 1 or depth[peak] <= tolerance:
        raise ValueError('The trajectory must first move forward, then return')
    if depth.min() < -tolerance or (np.diff(depth[:peak + 1]) < -tolerance).any():
        raise ValueError('The trajectory starts backward or reverses before reaching its forward peak')
    if (np.diff(depth[peak:]) > tolerance).any():
        raise ValueError('The return leg must move backward toward the start')
    if not np.allclose(centers[-1], centers[0], atol=tolerance):
        raise ValueError('The final camera position must equal the starting position')
    if not np.allclose(poses[-1, :3, :3], poses[0, :3, :3], atol=1e-4):
        raise ValueError('The final camera orientation must equal the starting orientation')
    return poses


def validate_intrinsics(intrinsics):
    for key in ['width', 'height', 'fx', 'fy']:
        if key not in intrinsics or not np.isfinite(intrinsics[key]) or intrinsics[key] <= 0:
            raise ValueError('camera_intrinsics.{} must be positive'.format(key))


def build_camera_path(preset, fps=None):
    """Return half-open periodic samples. JSON keyframes include the closed endpoint."""
    preset = copy.deepcopy(preset)
    validate_intrinsics(preset['camera_intrinsics'])
    crop = float(preset.get('viewport_crop', 1))
    if not np.isfinite(crop) or crop <= 0:
        raise ValueError('viewport_crop must be positive')
    rate = int(preset['fps'] if fps is None else fps)
    duration = float(preset['duration_seconds'])
    if rate <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError('fps and duration_seconds must be positive')
    frames = int(round(rate * duration))
    if frames < 4 or frames % 2:
        raise ValueError('Choose a duration and fps producing an even number of at least four frames')
    if 'keyframes' in preset:
        keys = preset['keyframes']
        key_times = np.asarray([k['time_seconds'] for k in keys], dtype=np.float64)
        poses = validate_forward_return([k['camera_to_world'] for k in keys])
        if (not np.isfinite(key_times).all() or not np.all(np.diff(key_times) > 0)
                or abs(key_times[0]) > 1e-8 or abs(key_times[-1] - duration) > 1e-8):
            raise ValueError('Keyframe times must increase strictly from 0 to duration_seconds')
        times = np.arange(frames, dtype=np.float64) / rate
        centers = PchipInterpolator(key_times, poses[:, :3, 3], axis=0)(times)
        rotations = Slerp(key_times, Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
        result = np.repeat(np.eye(4)[None], frames, axis=0)
        result[:, :3, :3], result[:, :3, 3] = rotations, centers
        validate_forward_return(np.concatenate([result, result[:1]], axis=0))
        return result, times
    if float(preset['forward_travel']) <= 0:
        raise ValueError('forward_travel must be positive: move forward before returning')
    motion = preset.get('motion', 'spiral')
    turns = float(preset['turns_per_leg'])
    if not np.isfinite(turns) or turns <= 0 or not np.isclose(2 * turns, round(2 * turns)):
        raise ValueError('turns_per_leg must be a positive multiple of 0.5 to close the loop')
    if motion == 'spiral':
        result, times = project_page_cameras(preset, rate)
        validate_forward_return(np.concatenate([result, result[:1]], axis=0))
        return result, times
    if motion not in ['dolly', 'horizontal', 'vertical', 'figure-eight']:
        raise ValueError('Unknown motion: {}'.format(motion))
    base = copy.deepcopy(preset)
    base['lateral_radius_x'] = base['lateral_radius_y'] = 0
    result, times = project_page_cameras(base, rate)
    start = np.asarray(preset['camera_to_world_start'], dtype=np.float64)
    angle = 4 * np.pi * float(preset['turns_per_leg']) * times / duration
    x = float(preset['lateral_radius_x']) * np.sin(angle) if motion in ['horizontal', 'figure-eight'] else np.zeros(frames)
    y = float(preset['lateral_radius_y']) * np.sin(angle * (2 if motion == 'figure-eight' else 1)) if motion in ['vertical', 'figure-eight'] else np.zeros(frames)
    result[:, :3, 3] += np.stack([x, y, np.zeros(frames)], axis=1) @ start[:3, :3].T
    validate_forward_return(np.concatenate([result, result[:1]], axis=0))
    return result, times


def export_keyframes(preset):
    """Export an editable closed JSON path; full frame samples preserve the spiral."""
    poses, times = build_camera_path(preset)
    keys = [dict(time_seconds=float(t), camera_to_world=p.tolist()) for t, p in zip(times, poses)]
    keys.append(dict(time_seconds=float(preset['duration_seconds']), camera_to_world=poses[0].tolist()))
    return dict(version=1, convention='OpenCV C2W: +X right, +Y down, +Z forward',
                duration_seconds=float(preset['duration_seconds']), fps=int(preset['fps']),
                camera_intrinsics=preset['camera_intrinsics'], viewport_crop=preset.get('viewport_crop', 1),
                keyframes=keys, brightness_minimum=preset.get('brightness_minimum', .065),
                brightness_maximum=preset.get('brightness_maximum', .5), base_exposure=preset.get('base_exposure', 1))
