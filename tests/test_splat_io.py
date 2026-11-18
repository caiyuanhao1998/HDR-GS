import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from utils.splat_io import compact_splats


class SharedSplatTests(unittest.TestCase):
    def test_shared_geometry_restores_both_original_splat_files_exactly(self):
        random=np.random.RandomState(7)
        hdr=random.randint(0,256,(101,32)).astype(np.uint8)
        ldr=hdr.copy();ldr[:,24:27]=random.randint(0,256,(101,3))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);hdr.tofile(str(folder/'hdr.splat'));ldr.tofile(str(folder/'ldr.splat'))
            metadata=compact_splats(folder)
            decoded=np.frombuffer(gzip.decompress((folder/'gaussians.splat.gz').read_bytes()),dtype=np.uint8).reshape(32,-1).T.copy()
            self.assertEqual(decoded.tobytes(),hdr.tobytes())
            decoded[:,24:27]=np.frombuffer(gzip.decompress((folder/'ldr.rgb.gz').read_bytes()),dtype=np.uint8).reshape(-1,3)
            self.assertEqual(decoded.tobytes(),ldr.tobytes())
            self.assertEqual(hashlib.sha256(decoded.tobytes()).hexdigest(),metadata['ldrSplatSHA256'])

    def test_different_geometry_and_truncated_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'hdr.splat').write_bytes(bytes(32));(folder/'ldr.splat').write_bytes(bytes([1])+bytes(31))
            with self.assertRaises(ValueError):compact_splats(folder)
            (folder/'hdr.splat').write_bytes(bytes(31))
            with self.assertRaises(ValueError):compact_splats(folder)
