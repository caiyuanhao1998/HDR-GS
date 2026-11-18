import tempfile
import unittest
from pathlib import Path
import numpy as np
from utils.camera_paths import build_camera_path, retime_preset, scene_preset
from utils.viewer_cache import color_key, geometry_key, read_entry, restore_entry, store_entry


class ViewerCacheTests(unittest.TestCase):
    def test_video_timing_does_not_invalidate_gaussian_geometry(self):
        original = scene_preset('bathroom')
        faster = retime_preset(original, 8, 30)
        a, _ = build_camera_path(original)
        b, _ = build_camera_path(faster)
        key = lambda pose: geometry_key('model', pose, original['camera_intrinsics'], 1, 600000)
        self.assertEqual(key(a[0]), key(b[0]))
        moved = a[0].copy(); moved[0, 3] += .1
        self.assertNotEqual(key(a[0]), key(moved))
        self.assertNotEqual(key(a[0]), geometry_key('new-model', a[0], original['camera_intrinsics'], 1, 600000))

    def test_brightness_and_tone_mapper_invalidate_colors_only(self):
        base = color_key('geometry', 'mapper', .3, 2)
        self.assertNotEqual(base, color_key('geometry', 'mapper', .4, 2))
        self.assertNotEqual(base, color_key('geometry', 'new-mapper', .3, 2))
        self.assertNotEqual(base, color_key('geometry', 'mapper', .3, 4))

    def test_corrupted_cache_is_rejected_and_restored_outputs_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.splat'; source.write_bytes(b'original bytes')
            cache = root / 'cache'; out = root / 'output'
            store_entry(cache, {'hdr.splat': source}, {'range': 10})
            self.assertEqual(read_entry(cache), {'range': 10})
            restore_entry(cache, out, ['hdr.splat'])
            # Callers unlink old output links before regenerating a new exposure.
            (out / 'hdr.splat').unlink(); (out / 'hdr.splat').write_bytes(b'new export')
            self.assertEqual((cache / 'hdr.splat').read_bytes(), b'original bytes')
            (cache / 'hdr.splat').write_bytes(b'corrupt')
            self.assertIsNone(read_entry(cache))


if __name__ == '__main__':
    unittest.main()
