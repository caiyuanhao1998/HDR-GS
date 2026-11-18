"""Lossless shared geometry for browser LDR/HDR Gaussian previews."""
import gzip
import hashlib
import io
from pathlib import Path
import numpy as np


def gzip_bytes(data):
    output = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return output.getvalue()


def compact_splats(folder):
    folder = Path(folder)
    hdr_bytes = (folder / 'hdr.splat').read_bytes()
    if not hdr_bytes or len(hdr_bytes) % 32:
        raise ValueError('A splat file must contain complete 32-byte Gaussian records')
    hdr = np.frombuffer(hdr_bytes, dtype=np.uint8).reshape(-1, 32)
    # Group corresponding bytes to compress float exponents more efficiently.
    (folder / 'gaussians.splat.gz').write_bytes(gzip_bytes(hdr.T.tobytes()))
    metadata = dict(gaussianEncoding='byte-planar-gzip', gaussianAsset='gaussians.splat.gz',
                    hdrSplatSHA256=hashlib.sha256(hdr_bytes).hexdigest())
    if (folder / 'ldr.splat').is_file():
        ldr_bytes = (folder / 'ldr.splat').read_bytes()
        ldr = np.frombuffer(ldr_bytes, dtype=np.uint8).reshape(-1, 32)
        if ldr.shape != hdr.shape or not np.array_equal(ldr[:, :24], hdr[:, :24]) or not np.array_equal(ldr[:, 27:], hdr[:, 27:]):
            raise ValueError('LDR and HDR splats must share geometry, rotation and opacity')
        (folder / 'ldr.rgb.gz').write_bytes(gzip_bytes(ldr[:, 24:27].tobytes()))
        metadata.update(ldrColorAsset='ldr.rgb.gz', ldrSplatSHA256=hashlib.sha256(ldr_bytes).hexdigest())
    return metadata
