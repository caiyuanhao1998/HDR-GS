"""The HDR-GS studio on Hugging Face ZeroGPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import contextvars
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

# Import before the renderer so ZeroGPU can initialize its CUDA runtime.
import spaces
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

from app import ROOT, SCENES, make_demo
from usage import UsageCounter

MODEL_REPOSITORY = 'CaiYuanhao/HDR-GS'
MODEL_REVISION = '6e5e69ae1cf06f463e1c88d04925dc89823534d1'
UPLOAD_BYTES = 384 * 1024 ** 2


def gpu_duration(mode, arguments, progress, cache):
    settings = json.loads(Path(arguments[arguments.index('--trajectory') + 1]).read_text())
    frame_count = math.ceil(float(settings['duration_seconds']) * int(settings['fps']))
    cap = 75 if mode == 'ldr' else 42
    minimum = 28 if mode == 'ldr' else 19
    return max(minimum, min(cap, int(15 + cap * frame_count / 240)))


@spaces.GPU(duration=gpu_duration)
def render_on_gpu(mode, arguments, progress, cache):
    from utils.video_rendering import add_arguments, render_video

    parser = add_arguments(argparse.ArgumentParser(), mode)
    args = parser.parse_args(arguments)
    args.gpu_id = None  # ZeroGPU assigns the device.
    args.progress_json = False
    args.progress_file = Path(progress)
    args.viewer_cache = Path(cache)
    args.weights_only = True
    render_video(mode, args)


class HostedRenderer:
    def __init__(self, storage):
        self.storage = storage

    def download_model(self, scene):
        item = next(entry for entry in SCENES if entry['id'] == scene)
        return tuple(Path(hf_hub_download(
            MODEL_REPOSITORY, filename=item['checkpoint_directory'] + '/' + name,
            revision=MODEL_REVISION, cache_dir=self.storage / 'checkpoints', token=False))
            for name in ('point_cloud.ply', 'tone_mapper.pth'))

    def check_ply(self, path):
        if path.stat().st_size > UPLOAD_BYTES:
            raise ValueError('Upload an HDR-GS PLY under 384 MiB.')
        with path.open('rb') as stream:
            header = stream.read(64 * 1024).replace(b'\r\n', b'\n').split(b'end_header\n', 1)
        if len(header) != 2:
            raise ValueError('Invalid HDR-GS PLY header.')
        items = [line.split() for line in header[0].decode('ascii').splitlines()]
        elements = [line for line in items if line and line[0] == 'element']
        properties = [line for line in items if line and line[0] == 'property']
        canonical = {'x', 'y', 'z', 'opacity'}
        canonical.update('f_dc_{}'.format(n) for n in range(3))
        canonical.update('f_rest_{}'.format(n) for n in range(45))
        canonical.update('scale_{}'.format(n) for n in range(3))
        canonical.update('rot_{}'.format(n) for n in range(4))
        if (len(items) < 2 or items[0] != ['ply'] or
                items[1] != ['format', 'binary_little_endian', '1.0'] or
                len(elements) != 1 or len(elements[0]) != 3 or elements[0][1] != 'vertex' or
                not 0 < int(elements[0][2]) <= 1600000 or len(properties) > 64 or
                any(len(prop) != 3 or prop[1] not in ('float', 'float32') for prop in properties) or
                not canonical.issubset({prop[2] for prop in properties})):
            raise ValueError('Upload a binary HDR-GS Gaussian PLY with the original point fields.')

    def check_uploads(self, ply, mapper):
        self.check_ply(ply)
        if mapper.stat().st_size > 2 * 1024 ** 2 or not zipfile.is_zipfile(mapper):
            raise ValueError('Upload the tone_mapper.pth saved by HDR-GS.')
        with zipfile.ZipFile(mapper) as archive:
            if len(archive.infolist()) > 100 or sum(file.file_size for file in archive.infolist()) > 4 * 1024 ** 2:
                raise ValueError('This tone-mapper file is larger than expected.')

    @staticmethod
    def check_timing(preset):
        seconds, fps = float(preset['duration_seconds']), int(preset['fps'])
        if not 2 <= seconds <= 8 or fps not in (12, 24, 30):
            raise ValueError('Online rendering supports 2–8 seconds at 12, 24 or 30 FPS.')

    @staticmethod
    def check_size(size):
        if int(size) not in (240, 480, 720):
            raise ValueError('Choose a video long side of 240, 480 or 720 pixels.')

    @staticmethod
    def clean_results(outputs):
        cutoff = time.time() - 6 * 3600
        for path in outputs.iterdir():
            is_job = (len(path.name) == 32 and all(char in '0123456789abcdef' for char in path.name))
            if (is_job or path.name.startswith('path-')) and path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)

    def render_events(self, mode, arguments, log):
        progress = log.with_suffix('.progress.json')
        context = contextvars.copy_context()
        view_cache = self.storage / 'viewers'
        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(context.run, render_on_gpu, mode, arguments, str(progress), str(view_cache))
            previous = ''
            while True:
                try:
                    job.result(timeout=.5)
                    complete = True
                except TimeoutError:
                    complete = False
                encoded = progress.read_text() if progress.is_file() else ''
                yield json.loads(encoded) if encoded and encoded != previous else {}
                previous = encoded
                if complete:
                    break


def main():
    storage = Path(os.environ.get('HDRGS_SPACE_CACHE', '/tmp/hdr-gs-studio'))
    outputs = storage / 'results'
    backend = HostedRenderer(storage)
    usage = UsageCounter('/mnt/hdr-gs-usage')
    demo = make_demo(Path(sys.executable), storage / 'models', outputs, '0', hosted=backend, usage=usage)
    assets = FastAPI()

    @assets.get('/api/usage')
    def usage_summary():
        return usage.summary() or {'available': False}

    assets.mount('/results', StaticFiles(directory=str(outputs)), name='results')
    assets.mount('/studio-assets', StaticFiles(directory=str(ROOT / 'demo/static')), name='studio-assets')
    # Gradio.launch registers the ZeroGPU worker before the viewer routes are added.
    live, _, _ = demo.queue(max_size=8).launch(
        server_name='0.0.0.0', server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
        prevent_thread_lock=True, ssr_mode=False, show_error=True,
        max_file_size='384mb', allowed_paths=[str(outputs), str(ROOT / 'demo/static')])
    live.router.routes[0:0] = assets.router.routes
    demo.block_thread()


if __name__ == '__main__':
    main()
