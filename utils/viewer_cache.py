"""Content-addressed Gaussian preview caches, independent of video timing."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _key(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def geometry_key(model_hash, reference_pose, intrinsics, crop, max_points):
    # Reference-view SH colors depend on the camera; timing and the rest of the
    # trajectory do not. Use the float32 matrix consumed by the renderer.
    view = np.linalg.inv(np.asarray(reference_pose, dtype=np.float64)).astype('<f4')
    return _key(dict(version=1, model=model_hash, reference_view=view.tobytes().hex(),
                     intrinsics={k: float(intrinsics[k]) for k in ['width', 'height', 'fx', 'fy']},
                     crop=float(crop), max_points=int(max_points)))


def color_key(hdr_key, mapper_hash, brightness, base_exposure):
    return _key(dict(version=1, hdr=hdr_key, mapper=mapper_hash,
                     brightness=float(brightness), base_exposure=float(base_exposure)))


def link_or_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def read_entry(folder):
    folder = Path(folder)
    try:
        manifest = json.loads((folder / 'cache.json').read_text())
        for name, record in manifest['files'].items():
            if Path(name).name != name:
                return None
            path = folder / name
            if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
                return None
        return manifest['metadata']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store_entry(folder, sources, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, source in sources.items():
        temporary = folder / (name + '.tmp-' + uuid.uuid4().hex)
        link_or_copy(source, temporary)
        temporary.replace(folder / name)
        records[name] = dict(bytes=(folder / name).stat().st_size, sha256=file_sha256(folder / name))
    temporary = folder / ('cache.tmp-' + uuid.uuid4().hex + '.json')
    temporary.write_text(json.dumps(dict(metadata=metadata, files=records), indent=2) + '\n')
    temporary.replace(folder / 'cache.json')


def restore_entry(source, destination, names):
    for name in names:
        link_or_copy(Path(source) / name, Path(destination) / name)
