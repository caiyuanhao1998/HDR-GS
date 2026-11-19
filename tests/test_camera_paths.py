import copy
import unittest
from pathlib import Path

import numpy as np

from utils.camera_paths import (
    build_camera_path, export_keyframes, model_candidates, retime_preset, scene_preset, scene_registry,
)


class CameraPathTests(unittest.TestCase):
    def test_checkpoint_paths_prefer_grouped_downloads(self):
        root = Path('pretrained_weights')
        for scene in scene_registry():
            with self.subTest(scene=scene['id']):
                self.assertEqual(model_candidates(root, scene['id'])[0], root / scene['group'] / scene['id'])

        self.assertEqual(model_candidates(root, 'bear')[1:], [
            root / 'refined-2026-11-18' / 'synthetic' / 'bear', root / 'bear',
        ])
        self.assertEqual(model_candidates(root, 'sofa')[1:], [root / 'sofa_2', root / 'sofa'])

    def test_all_twelve_project_presets_move_forward_and_return(self):
        self.assertEqual(len(scene_registry()), 12)
        for scene in scene_registry():
            with self.subTest(scene=scene['id']):
                preset = scene_preset(scene['id'])
                path, times = build_camera_path(preset)
                depth = (path[:, :3, 3] - path[0, :3, 3]) @ path[0, :3, 2]
                self.assertEqual(path.shape, (960, 4, 4))
                self.assertTrue((np.diff(depth[:481]) > 0).all())
                self.assertTrue((np.diff(depth[480:]) < 0).all())
                self.assertAlmostEqual(times[-1], 16 - 1 / 60)

    def test_eight_second_playback_preserves_all_twelve_spatial_paths(self):
        for scene in scene_registry():
            preset = scene_preset(scene['id'])
            original, _ = build_camera_path(preset)
            faster = retime_preset(preset, 8, 30)
            path, times = build_camera_path(faster)
            self.assertEqual(path.shape, (240, 4, 4))
            np.testing.assert_allclose(path, original[::4], atol=1e-10)
            self.assertAlmostEqual(times[-1], 8 - 1 / 30)
            self.assertEqual(preset['duration_seconds'], 16)

    def test_custom_keyframe_retiming_preserves_poses(self):
        original = scene_preset('bathroom')
        reference, _ = build_camera_path(original)
        custom = export_keyframes(original)
        faster = retime_preset(custom, 8, 30)
        path, _ = build_camera_path(faster)
        np.testing.assert_allclose(path, reference[::4], atol=1e-10)
        self.assertEqual(faster['keyframes'][-1]['time_seconds'], 8)

    def test_exported_sampled_camera_json_round_trips(self):
        preset = scene_preset('bathroom')
        original, _ = build_camera_path(preset)
        restored, _ = build_camera_path(export_keyframes(preset))
        np.testing.assert_allclose(restored, original, atol=1e-10)

    def test_all_motion_presets_close_with_no_backward_first_step(self):
        for motion in ['spiral', 'dolly', 'horizontal', 'vertical', 'figure-eight']:
            preset = scene_preset('bathroom')
            preset['motion'] = motion
            path, _ = build_camera_path(export_keyframes(preset))
            direction = path[0, :3, 2]
            self.assertGreater(float((path[1, :3, 3] - path[0, :3, 3]) @ direction), 0)

    def test_custom_backward_first_and_nonclosing_paths_are_rejected(self):
        preset = scene_preset('bathroom')
        preset['duration_seconds'] = 4
        first = np.eye(4)
        peak = first.copy()
        peak[2, 3] = 1
        preset['keyframes'] = [dict(time_seconds=t, camera_to_world=pose.tolist())
                               for t, pose in [(0, first), (2, peak), (4, first)]]
        path, _ = build_camera_path(preset)
        self.assertGreater(path[1, 2, 3], 0)
        backwards = copy.deepcopy(preset)
        backwards['keyframes'][1]['camera_to_world'][2][3] = -1
        with self.assertRaises(ValueError):
            build_camera_path(backwards)
        nonclosing = copy.deepcopy(preset)
        nonclosing['keyframes'][-1]['camera_to_world'][0][3] = .5
        with self.assertRaises(ValueError):
            build_camera_path(nonclosing)

    def test_invalid_time_rotation_and_spiral_winding_are_rejected(self):
        for change in [dict(fps=0), dict(turns_per_leg=.7), dict(forward_travel=-1)]:
            preset = scene_preset('bathroom')
            preset.update(change)
            with self.assertRaises(ValueError):
                build_camera_path(preset)
        preset = export_keyframes(scene_preset('bathroom'))
        preset['keyframes'][1]['time_seconds'] = 0
        with self.assertRaises(ValueError):
            build_camera_path(preset)
        preset = export_keyframes(scene_preset('bathroom'))
        preset['keyframes'][1]['camera_to_world'][0][0] = 2
        with self.assertRaises(ValueError):
            build_camera_path(preset)


if __name__ == '__main__':
    unittest.main()
