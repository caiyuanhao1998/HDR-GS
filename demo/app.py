"""Local HDR-GS studio. A modern UI drives the existing CUDA Python environment."""
import argparse
import copy
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load pandas before Gradio starts callback threads; Plotly inspects its types.
import pandas as pd
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from plyfile import PlyData
import uvicorn

from utils.camera_paths import build_camera_path, export_keyframes, scene_preset, scene_registry

SCENES = scene_registry()
MOTIONS = {'Spiral forward & backward': 'spiral', 'Forward & backward': 'dolly',
           'Horizontal sway': 'horizontal', 'Vertical sway': 'vertical', 'Figure eight': 'figure-eight'}
LOCK = threading.Lock()


def estimate_custom_preset(ply_path):
    vertex = PlyData.read(ply_path)['vertex']
    points = np.stack([vertex[k] for k in ['x', 'y', 'z']], axis=1)
    center = np.median(points, axis=0)
    extent = max(float(np.ptp(np.percentile(points, [2, 98], axis=0), axis=0).max()), .1)
    start = np.eye(4)
    start[:3, 3] = center - [0, 0, extent * 1.6]
    return dict(name='Custom model', fps=60, duration_seconds=16, motion='spiral',
                camera_to_world_start=start.tolist(), camera_intrinsics=dict(width=720, height=720, fx=864., fy=864.),
                forward_travel=extent * .5, lateral_radius_x=extent * .08, lateral_radius_y=extent * .05,
                turns_per_leg=2, turn_blend_seconds=.6, viewport_crop=1., base_exposure=1.,
                brightness_minimum=.065, brightness_maximum=.5)


def plot_path(preset):
    poses, times = build_camera_path(preset)
    centers = poses[:, :3, 3]
    forward = poses[0, :3, 2]
    peak = int(np.argmax((centers - centers[0]) @ forward))
    figure = go.Figure()
    for points, label, color in [(centers[:peak + 1], 'Forward', '#479cae'),
                                  (np.vstack([centers[peak:], centers[:1]]), 'Backward', '#e39a55')]:
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines',
                                     name=label, line=dict(color=color, width=5)))
    for index, label, color in [(0, 'Start / finish', '#333333'), (peak, 'Forward peak', '#ca6a31')]:
        point = centers[index]
        figure.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', name=label,
                                     marker=dict(size=5, color=color)))
    figure.update_layout(height=380, margin=dict(l=0, r=0, b=0, t=10),
                         legend=dict(orientation='h', y=1), scene=dict(aspectmode='data',
                         xaxis_title='World X', yaxis_title='World Y', zaxis_title='World Z'))
    return figure


def make_demo(renderer_python, models_root, outputs, gpu_id):
    outputs.mkdir(parents=True, exist_ok=True)

    def model_for(scene, ply, mapper):
        if scene == 'custom':
            if not ply or not mapper:
                raise gr.Error('Upload both an HDR-GS point_cloud.ply and its matching tone_mapper.pth.')
            return Path(ply), Path(mapper)
        entry = next(s for s in SCENES if s['id'] == scene)
        candidates = [models_root / scene, models_root / entry['model_directory']]
        for directory in candidates:
            if (directory / 'point_cloud.ply').is_file() and (directory / 'tone_mapper.pth').is_file():
                return directory / 'point_cloud.ply', directory / 'tone_mapper.pth'
        raise gr.Error('Model not found. Put point_cloud.ply and tone_mapper.pth in {}.'.format(candidates[0]))

    def preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high):
        if not 0 < float(low) <= float(high) < 1:
            raise gr.Error('Choose brightness limits with 0 < minimum <= peak < 1.')
        if custom:
            try:
                preset = json.loads(Path(custom).read_text())
                preset.update(brightness_minimum=float(low), brightness_maximum=float(high))
                build_camera_path(preset)
                return preset
            except (ValueError, KeyError, TypeError) as error:
                raise gr.Error('Invalid camera JSON: {}'.format(error))
        if scene == 'custom':
            if not ply:
                raise gr.Error('Upload your model before previewing its automatic camera path.')
            preset = estimate_custom_preset(ply)
        else:
            preset = copy.deepcopy(scene_preset(scene))
        preset.update(motion=MOTIONS[motion], duration_seconds=float(seconds), fps=int(fps),
                      turns_per_leg=float(turns), brightness_minimum=float(low), brightness_maximum=float(high))
        preset['forward_travel'] *= float(forward)
        preset['lateral_radius_x'] *= float(horizontal)
        preset['lateral_radius_y'] *= float(vertical)
        preset['turn_blend_seconds'] = min(.6, float(seconds) / 8)
        build_camera_path(preset)
        return preset

    def preview(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high):
        preset = preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high)
        folder = outputs / ('path-' + uuid.uuid4().hex)
        folder.mkdir()
        destination = folder / 'camera-path.json'
        destination.write_text(json.dumps(export_keyframes(preset), indent=2) + '\n')
        return plot_path(preset), str(destination)

    def render_pair(scene, ply, mapper, custom, motion, forward, horizontal, vertical, turns, seconds, fps,
                    low, high, brightness_mode, fixed, size, hdr_ev, progress=gr.Progress()):
        model_file, tone_file = model_for(scene, ply, mapper)
        preset = preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high)
        job = uuid.uuid4().hex
        folder = outputs / job
        folder.mkdir()
        trajectory = folder / 'camera-path.json'
        trajectory.write_text(json.dumps(preset, indent=2) + '\n')
        editable = folder / 'camera-path-editable.json'
        editable.write_text(json.dumps(export_keyframes(preset), indent=2) + '\n')
        common = ['--scene', scene if scene != 'custom' else 'bathroom', '--model_path', str(model_file),
                  '--tone_mapper', str(tone_file), '--trajectory', str(trajectory), '--output', str(folder),
                  '--max_size', str(int(size)), '--gpu_id', str(gpu_id)]
        env = os.environ.copy()
        env.update(TMPDIR=str(outputs / '.tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', OPENCV_IO_ENABLE_OPENEXR='1')
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        # One local GPU render job at a time, while the UI remains responsive.
        with LOCK:
            for index, mode in enumerate(['hdr', 'ldr']):
                progress((index, 2), desc='Rendering {} video and Gaussian previews'.format(mode.upper()))
                command = [str(renderer_python), str(ROOT / ('render_' + mode + '.py'))] + common
                if mode == 'hdr':
                    command += ['--exposure', str(hdr_ev)]
                elif brightness_mode == 'Fixed brightness':
                    command += ['--brightness', str(fixed)]
                log = folder / (mode + '-render.log')
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise gr.Error('{} rendering failed:\n{}'.format(mode.upper(), log.read_text()[-2500:]))
        for original, name in [(model_file, 'point_cloud.ply'), (tone_file, 'tone_mapper.pth')]:
            target = folder / name
            try:
                os.link(str(original), str(target))
            except OSError:
                import shutil
                shutil.copy2(str(original), str(target))
        base = '/results/' + job
        def viewer(mode):
            return '<iframe title="{} Gaussian point cloud" src="{}/viewer/index.html?scene=render&mode={}&embed=1" style="width:100%;height:410px;border:0;border-radius:8px" allow="fullscreen"></iframe>'.format(mode.upper(), base, mode)
        links = [('LDR video', 'ldr.mp4'), ('HDR video', 'hdr.mp4'),
                 ('LDR visualization splat', 'assets/scenes/render/ldr.splat'),
                 ('HDR visualization splat', 'assets/scenes/render/hdr.splat'),
                 ('Original HDR-GS point cloud', 'point_cloud.ply'), ('Tone mapper', 'tone_mapper.pth'),
                 ('Editable camera JSON', 'camera-path-editable.json'), ('HDR display settings', 'tonemap.json'),
                 ('Raw HDR frame (EXR)', 'frames_hdr/{:06d}.exr'.format(len(build_camera_path(preset)[0]) // 2))]
        downloads = '<div style="display:flex;flex-wrap:wrap;gap:10px">' + ''.join(
            '<a href="{}/{}" download style="padding:7px 12px;border:1px solid #d6dce0;border-radius:6px">{}</a>'.format(base, path, html.escape(label))
            for label, path in links) + '</div>'
        progress(1, desc='Ready')
        return ('Rendered both videos. The two viewers use corresponding LDR and HDR colors.',
                str(folder / 'ldr.mp4'), str(folder / 'hdr.mp4'), viewer('ldr'), viewer('hdr'), downloads,
                plot_path(preset), str(editable))

    def choose_scene(scene):
        preset = scene_preset(scene) if scene != 'custom' else scene_preset('bathroom')
        return gr.update(visible=scene == 'custom'), preset.get('brightness_maximum', .5), plot_path(preset)

    example = dict(version=1, convention='OpenCV C2W: +X right, +Y down, +Z forward',
                   duration_seconds=4, fps=30, camera_intrinsics=dict(width=720, height=720, fx=864, fy=864),
                   keyframes=[dict(time_seconds=t, camera_to_world=[[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])
                              for t,z in [(0,0),(2,1),(4,0)]])
    css = '.gradio-container{max-width:1540px!important} .studio-note{color:#6a6a6a}'
    with gr.Blocks(title='HDR-GS Studio', css=css, theme=gr.themes.Soft(primary_hue='cyan')) as demo:
        gr.Markdown('# HDR-GS Studio\nRender a forward-and-back camera path with LDR exposure control and HDR radiance.')
        scene = gr.Dropdown(choices=[(s['label'], s['id']) for s in SCENES] + [('Upload custom model', 'custom')],
                            value='bathroom', label='Scene')
        with gr.Group(visible=False) as uploads:
            with gr.Row():
                ply = gr.File(label='HDR-GS point_cloud.ply', file_types=['.ply'], type='filepath')
                mapper = gr.File(label='Matching tone_mapper.pth', file_types=['.pth', '.pt'], type='filepath')
        with gr.Row():
            with gr.Column(scale=1):
                motion = gr.Dropdown(list(MOTIONS), value='Spiral forward & backward', label='Camera path')
                forward = gr.Slider(.1, 2, 1, step=.05, label='Forward travel × scene preset')
                with gr.Row():
                    horizontal = gr.Slider(0, 2, 1, step=.05, label='Horizontal amplitude ×')
                    vertical = gr.Slider(0, 2, 1, step=.05, label='Vertical amplitude ×')
                turns = gr.Slider(.5, 4, 2, step=.5, label='Spiral turns per leg')
                with gr.Row():
                    seconds = gr.Slider(2, 30, 16, step=1, label='Duration (seconds)')
                    fps = gr.Dropdown([12,24,30,60], value=60, label='FPS')
                    size = gr.Dropdown([240,480,720,960], value=720, label='Video long side (px)')
                with gr.Accordion('Custom camera JSON', open=False):
                    custom = gr.File(label='Upload camera path (overrides path controls)', file_types=['.json'], type='filepath')
                    gr.Markdown('OpenCV camera-to-world matrices: **+X right, +Y down, +Z forward**. Start at 0 seconds, move forward first, then return to the same camera pose. The last keyframe is the closed endpoint; it is omitted from the encoded loop.')
                    gr.Code(json.dumps(example, indent=2), language='json', label='Editable trajectory format')
                brightness_mode = gr.Radio(['Project-page gradient', 'Fixed brightness'], value='Project-page gradient', label='LDR brightness')
                fixed = gr.Slider(.01, .9, .35, step=.005, label='Fixed mean luminance')
                with gr.Row():
                    low = gr.Slider(.01, .5, .065, step=.005, label='Gradient minimum')
                    high = gr.Slider(.1, .9, .525, step=.005, label='Gradient peak')
                hdr_ev = gr.Slider(-4,4,0,step=.1,label='HDR display exposure (EV)')
            with gr.Column(scale=1):
                trajectory_plot = gr.Plot(value=plot_path(scene_preset('bathroom')), label='Camera trajectory')
                preview_button = gr.Button('Preview trajectory')
                trajectory_download = gr.File(label='Download editable camera JSON', interactive=False)
                gr.Markdown('The camera always travels forward before returning. Spirals keep the same winding direction. HDR videos use one fixed Reinhard calibration for the entire clip.', elem_classes='studio-note')
                render_button = gr.Button('Render LDR + HDR', variant='primary', size='lg')
                status = gr.Markdown('Choose a scene and preview its camera path, or render both outputs.')
        with gr.Row():
            ldr_video = gr.Video(label='LDR · learned tone mapper', format='mp4', interactive=False)
            hdr_video = gr.Video(label='HDR · Reinhard display', format='mp4', interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown('### LDR Gaussian point cloud')
                ldr_viewer = gr.HTML('<p>Render a scene to explore its LDR point cloud.</p>')
            with gr.Column():
                gr.Markdown('### HDR Gaussian point cloud')
                hdr_viewer = gr.HTML('<p>Render a scene to explore its HDR point cloud.</p>')
        gr.Markdown('### Downloads\nMP4 previews play on standard screens. EXR frames retain float32 HDR radiance; the original PLY retains all trained Gaussians and spherical harmonics. Interactive previews bake reference-view colors.')
        downloads = gr.HTML()
        camera_inputs = [scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high]
        scene.change(choose_scene, [scene], [uploads, high, trajectory_plot])
        preview_button.click(preview, camera_inputs, [trajectory_plot, trajectory_download], api_name='preview_trajectory')
        render_button.click(render_pair, [scene, ply, mapper, custom, motion, forward, horizontal, vertical,
            turns, seconds, fps, low, high, brightness_mode, fixed, size, hdr_ev],
            [status, ldr_video, hdr_video, ldr_viewer, hdr_viewer, downloads, trajectory_plot, trajectory_download], api_name='render_pair', concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer-python', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, default=ROOT / 'pretrained_weights')
    parser.add_argument('--outputs', type=Path, default=ROOT / 'demo/outputs')
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7863)
    args = parser.parse_args()
    demo = make_demo(args.renderer_python.resolve(), args.models_root.resolve(), args.outputs.resolve(), args.gpu_id)
    app = FastAPI()
    app.mount('/results', StaticFiles(directory=str(args.outputs.resolve())), name='results')
    app = gr.mount_gradio_app(app, demo.queue(max_size=8), path='/', allowed_paths=[str(args.outputs.resolve())])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
