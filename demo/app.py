"""Local HDR-GS studio. A modern UI drives the existing CUDA Python environment."""
import argparse
import copy
import html
import json
import os
import queue
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load pandas before Gradio starts callback threads; Plotly inspects its types.
import pandas as pd
import gradio as gr
from gradio.themes.utils.colors import Color
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from plyfile import PlyData
import uvicorn

from utils.camera_paths import build_camera_path, export_keyframes, model_candidates, retime_preset, scene_preset, scene_registry

SCENES = scene_registry()
PREVIEWS = {item['id']: item for item in json.loads((ROOT / 'demo/scene-previews.json').read_text())}
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
    return dict(name='Custom model', fps=30, duration_seconds=8, motion='spiral',
                camera_to_world_start=start.tolist(), camera_intrinsics=dict(width=720, height=720, fx=864., fy=864.),
                forward_travel=extent * .5, lateral_radius_x=extent * .08, lateral_radius_y=extent * .05,
                turns_per_leg=2, turn_blend_seconds=.3, viewport_crop=1., base_exposure=1.,
                brightness_minimum=.065, brightness_maximum=.5)


def demo_scene_preset(scene):
    return retime_preset(scene_preset(scene), 8, 30)


def plot_path(preset):
    """Compact camera frusta and travel arrows, shown in the starting camera frame."""
    poses, _ = build_camera_path(preset)
    centers = poses[:, :3, 3]
    peak = int(np.argmax((centers - centers[0]) @ poses[0, :3, 2]))

    def display(points):
        local = (np.asarray(points) - centers[0]) @ poses[0, :3, :3]
        return local[:, [0, 2, 1]] * [1, 1, -1]  # right, forward, up

    path = display(centers)
    span = max(float(np.ptp(path, axis=0).max()), .1)
    figure = go.Figure()
    for points, label, color, dash in [
            (path[:peak + 1], 'Forward', '#292929', 'solid'),
            (np.vstack([path[peak:], path[:1]]), 'Backward', '#999999', 'dash')]:
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines', name=label,
            line=dict(color=color, width=4, dash=dash),
            hovertemplate='X %{x:.2f}<br>Z %{y:.2f}<br>−Y %{z:.2f}<extra>'+label+'</extra>'))

    intr = preset['camera_intrinsics']
    crop = float(preset.get('viewport_crop', 1))
    half_x = intr['width'] / (2 * intr['fx'] * crop)
    half_y = intr['height'] / (2 * intr['fy'] * crop)
    corners = np.array([[0, 0, 0], [-half_x, -half_y, 1], [half_x, -half_y, 1],
                        [half_x, half_y, 1], [-half_x, half_y, 1]]) * (span * .065)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    bounds = [path]
    samples = sorted(set(np.linspace(0, len(poses), 6, endpoint=False, dtype=int).tolist() + [peak]))
    for index in samples:
        world = corners @ poses[index, :3, :3].T + centers[index]
        camera = display(world)
        bounds.append(camera)
        segments = []
        for left, right in edges:
            segments.extend([camera[left].tolist(), camera[right].tolist(), [None, None, None]])
        coordinates = list(zip(*segments))
        figure.add_trace(go.Scatter3d(
            x=list(coordinates[0]), y=list(coordinates[1]), z=list(coordinates[2]),
            mode='lines', line=dict(color='#333333' if index in [0, peak] else '#aaaaaa',
                                    width=3 if index in [0, peak] else 1.5),
            name='Start camera' if index == 0 else 'Camera {}'.format(index), hoverinfo='name'))

    for index, label, color, symbol in [(0, 'Start / finish', '#292929', 'circle'),
                                        (peak, 'Forward peak', '#777777', 'diamond')]:
        point = path[index]
        figure.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', name=label,
            marker=dict(size=3.5, color=color, symbol=symbol), hoverinfo='name'))
    for index in [max(1, peak // 2), min(len(path) - 2, peak + (len(path) - peak) // 2)]:
        direction = path[index + 1] - path[index - 1]
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        point = path[index]
        figure.add_trace(go.Cone(
            x=[point[0]], y=[point[1]], z=[point[2]], u=[direction[0]], v=[direction[1]],
            w=[direction[2]], sizemode='absolute', sizeref=span * .06, anchor='tip',
            colorscale=[[0, '#666666'], [1, '#666666']], showscale=False,
            name='Travel direction', hoverinfo='name'))

    bounds = np.concatenate(bounds)
    center = (bounds.max(0) + bounds.min(0)) / 2
    radius = max(float(np.ptp(bounds, axis=0).max()), .1) * .57
    axes = [dict(title=dict(text=label, font=dict(size=9)),
                 range=[float(value - radius), float(value + radius)], nticks=3, tickformat='.2g',
                 tickfont=dict(size=8), color='#777777', gridcolor='#dedede',
                 zerolinecolor='#aaaaaa', linecolor='#cccccc', showbackground=False, showgrid=True)
            for value, label in zip(center, ['X', 'Z', '−Y'])]
    figure.update_layout(
        height=180, showlegend=False, margin=dict(l=8, r=8, b=12, t=4),
        paper_bgcolor='#fafafa', plot_bgcolor='#fafafa',
        font=dict(family='Arial, sans-serif', size=10, color='#666666'),
        hoverlabel=dict(bgcolor='#ffffff', bordercolor='#aaaaaa', font_color='#444444'),
        scene=dict(aspectmode='cube', domain=dict(x=[0, 1], y=[0, 1]),
                   bgcolor='#fafafa', xaxis=axes[0], yaxis=axes[1], zaxis=axes[2],
                   camera=dict(eye=dict(x=2.2, y=-2.4, z=1.8), up=dict(x=0, y=0, z=1))),
        uirevision='compact-camera-preview')
    return figure


def studio_theme():
    gray = Color('#fafafa', '#f5f5f5', '#e5e5e5', '#d4d4d4', '#a3a3a3', '#737373',
                 '#525252', '#404040', '#262626', '#171717', '#0a0a0a', name='gray')
    theme = gr.themes.Base(primary_hue=gray, secondary_hue=gray, neutral_hue=gray,
                           font=['Arial', 'Helvetica', 'sans-serif'],
                           font_mono=['Consolas', 'monospace'])
    settings = dict(
        body_background_fill='#ffffff', body_text_color='#202020', body_text_color_subdued='#777777',
        background_fill_primary='#ffffff', background_fill_secondary='#fafafa',
        block_background_fill='#ffffff', block_border_color='#dddddd', block_shadow='none',
        block_label_background_fill='#ffffff', block_label_text_color='#666666',
        block_label_border_color='#dddddd', block_label_shadow='none',
        input_background_fill='#ffffff', input_background_fill_focus='#ffffff',
        input_background_fill_hover='#fafafa', input_border_color='#d9d9d9',
        input_border_color_focus='#777777', input_border_color_hover='#a3a3a3',
        input_shadow='none', input_shadow_focus='0 0 0 1px #a3a3a3',
        button_primary_background_fill='#242424', button_primary_background_fill_hover='#111111',
        button_primary_border_color='#242424', button_primary_border_color_hover='#111111',
        button_primary_text_color='#ffffff', button_primary_text_color_hover='#ffffff',
        button_primary_shadow='none', button_primary_shadow_hover='none',
        button_primary_shadow_active='none',
        button_secondary_background_fill='#ffffff', button_secondary_background_fill_hover='#f5f5f5',
        button_secondary_border_color='#d4d4d4', button_secondary_border_color_hover='#999999',
        button_secondary_text_color='#444444', button_secondary_text_color_hover='#222222',
        button_secondary_shadow='none', button_secondary_shadow_hover='none',
        button_secondary_shadow_active='none',
        checkbox_background_color_selected='#444444', checkbox_border_color_selected='#444444',
        checkbox_label_background_fill_selected='#eeeeee', checkbox_label_border_color_selected='#777777',
        link_text_color='#555555', link_text_color_hover='#222222',
        link_text_color_visited='#555555', link_text_color_active='#222222',
        color_accent='#555555', color_accent_soft='#eeeeee', slider_color='#555555',
        radio_circle='#444444', loader_color='#555555', error_background_fill='#f5f5f5',
        error_border_color='#aaaaaa', error_text_color='#333333', error_icon_color='#666666')
    available = theme.to_dict()['theme']
    # Use the same light monochrome theme even when the browser prefers dark mode.
    for key, value in list(settings.items()):
        if key + '_dark' in available:
            settings[key + '_dark'] = value
    return theme.set(**{key: value for key, value in settings.items() if key in available})


def video_html(url, label, poster=None):
    poster_attr = ' poster="{}"'.format(html.escape(str(poster), quote=True)) if poster else ''
    return ('<video controls playsinline loop preload="metadata" aria-label="{}"{} src="{}">'
            'Your browser does not support MP4 video.</video>').format(
                html.escape(label, quote=True), poster_attr, html.escape(str(url), quote=True))


def viewer_html(base, mode, asset_base=None, exposure_ev=0.):
    source = base + '/viewer/index.html?scene=render&mode=' + mode + '&embed=1&controls=external'
    download_base = asset_base or base + '/assets/scenes/render/'
    download = download_base.rstrip('/') + '/' + mode + '.splat'
    if download.startswith('https://huggingface.co/'):
        download += '?download=true'
    if mode == 'hdr':
        adjustment = ('<label class="viewer-exposure">Exposure '
                      '<input type="range" min="-8" max="8" step="0.1" value="{0}" '
                      'aria-label="HDR point cloud exposure"><output>{0:.1f} EV</output></label>').format(float(exposure_ev))
    else:
        adjustment = '<span class="viewer-hint">Drag to orbit · Right-drag to pan · Scroll to zoom</span>'
    return ('<section class="gaussian-panel" data-mode="{mode}">'
            '<div class="gaussian-viewport"><iframe title="{title} Gaussian point cloud" src="{source}" '
            'allow="fullscreen"></iframe></div>'
            '<div class="viewer-action-row">'
            '<button type="button" data-viewer-action="reset" aria-label="Reset {title} view">Reset view</button>'
            '<button type="button" data-viewer-action="fullscreen" aria-label="Fullscreen {title} point cloud">Fullscreen</button>'
            '<a class="viewer-download" href="{download}" download="{mode}.splat">Download splat</a>'
            '</div><div class="viewer-adjustment">{adjustment}</div></section>').format(
                mode=mode, title=mode.upper(), source=html.escape(source, quote=True),
                download=html.escape(download, quote=True), adjustment=adjustment)


def render_progress_html(label, fraction=0., elapsed=0., detail=''):
    percent = max(0., min(100., float(fraction) * 100))
    return ('<section class="render-loading"><div class="render-progress-card" role="status" aria-live="polite">'
            '<strong>{}</strong><div class="render-progress-heading"><span>{}</span><span>{:.0f}%</span></div>'
            '<progress value="{:.2f}" max="100" aria-label="Render progress"></progress>'
            '<p>Elapsed {:.1f} s</p></div></section>').format(
                html.escape(label), html.escape(detail or 'Preparing the renderer'), percent, percent, float(elapsed))


def renderer_events(command, env, log):
    """Stream structured progress while preserving the complete CUDA worker log."""
    events = queue.Queue()
    process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_output():
        with log.open('w') as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith('HDRGS_PROGRESS '):
                    try:
                        events.put(json.loads(line.split(' ', 1)[1]))
                    except (ValueError, TypeError):
                        pass
        events.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        finished = False
        while not finished:
            try:
                event = events.get(timeout=.5)
                finished = event is None
            except queue.Empty:
                event = {}
            if not finished:
                yield event
        reader.join()
        if process.wait():
            raise RuntimeError(log.read_text()[-2500:])
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def download_html(base, links):
    items = []
    for label, path in links:
        url = path if path.startswith(('https://', 'http://', '/')) else base.rstrip('/') + '/' + path
        if url.startswith('https://huggingface.co/'):
            url += '?download=true'
        items.append('<a class="download-link" href="{}" download>{}<span>{}</span></a>'.format(
            html.escape(url, quote=True), html.escape(label),
            html.escape(Path(path.split('?', 1)[0]).suffix.lstrip('.').upper())))
    return '<div class="downloads-grid">' + ''.join(items) + '</div>'


def scene_summary(scene):
    if scene == 'custom':
        return '<div class="scene-info"><strong>Custom model</strong><span>Upload a point cloud and matching tone mapper</span></div>'
    item = PREVIEWS[scene]
    return '<div class="scene-info"><strong>{}</strong><span>{} scene · {:,} Gaussians</span></div>'.format(
        html.escape(item['label']), item['group'].capitalize(), item['sourcePointCount'])


def prepare_previews(outputs):
    """Keep viewer code local and load published scene data directly from Hugging Face."""
    for scene, item in PREVIEWS.items():
        folder = outputs / 'previews' / scene
        shutil.copytree(str(ROOT / 'demo/viewer'), str(folder / 'viewer'), dirs_exist_ok=True)
        (folder / 'assets').mkdir(exist_ok=True)
        metadata = copy.deepcopy(item)
        metadata['id'] = 'render'
        (folder / 'assets/scenes.json').write_text(json.dumps([metadata]))
        (folder / 'camera-path-editable.json').write_text(json.dumps(export_keyframes(demo_scene_preset(scene)), indent=2) + '\n')


def scene_preview(scene, outputs=None):
    if scene == 'custom':
        empty = '<div class="empty-preview">Upload a model, then render to see your results.</div>'
        return empty, empty, empty, empty, '', 'Upload a point cloud and tone mapper to begin.'
    item = PREVIEWS[scene]
    base = '/results/previews/' + scene
    video_folder = outputs / 'previews' / scene / 'videos' if outputs else None
    has_local_preview = video_folder is not None and all(
        (video_folder / (mode + '-8s-30fps.mp4')).is_file() for mode in ['ldr', 'hdr'])
    ldr_url = base + '/videos/ldr-8s-30fps.mp4' if has_local_preview else item.get('previewVideo', item['video'])
    hdr_url = base + '/videos/hdr-8s-30fps.mp4' if has_local_preview else item.get('previewHdrVideo', item['hdrVideo'])
    has_short_preview = has_local_preview or ('previewVideo' in item and 'previewHdrVideo' in item)
    downloads = download_html(item['dataset_scene_url'].rstrip('/'), [
        ('LDR video', ldr_url), ('HDR video', hdr_url),
        ('LDR point cloud', item['assetBase'] + 'ldr.splat'), ('HDR point cloud', item['assetBase'] + 'hdr.splat'),
        ('Camera trajectory', base + '/camera-path-editable.json'), ('HDR display settings', 'tonemap.json')])
    note = ('Scene preview · 8 s / 30 FPS. Adjust settings to render a new version.' if has_short_preview else
            'Project-page preview · 16 s / 60 FPS. Render to apply your settings.')
    return (video_html(ldr_url, item['label'] + ' LDR video',
                       item['dataset_scene_url'] + 'posters/ldr.webp'),
            video_html(hdr_url, item['label'] + ' HDR video',
                       '/studio-assets/thumbnails/' + scene + '.webp'),
            viewer_html(base, 'ldr', item['assetBase']),
            viewer_html(base, 'hdr', item['assetBase'], item.get('hdrDisplayExposureEV', 0.)), downloads, note)


def make_demo(renderer_python, models_root, outputs, gpu_id, hosted=None):
    outputs.mkdir(parents=True, exist_ok=True)
    prepare_previews(outputs)

    def model_for(scene, ply, mapper):
        if scene == 'custom':
            if not ply or not mapper:
                raise gr.Error('Upload both an HDR-GS point_cloud.ply and its matching tone_mapper.pth.')
            if hosted is not None:
                hosted.check_uploads(Path(ply), Path(mapper))
            return Path(ply), Path(mapper)
        candidates = model_candidates(models_root, scene)
        for directory in candidates:
            if (directory / 'point_cloud.ply').is_file() and (directory / 'tone_mapper.pth').is_file():
                return directory / 'point_cloud.ply', directory / 'tone_mapper.pth'
        if hosted is not None:
            return hosted.download_model(scene)
        raise gr.Error('Model not found. Put point_cloud.ply and tone_mapper.pth in {}.'.format(candidates[0]))

    def preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high):
        if not 0 < float(low) <= float(high) < 1:
            raise gr.Error('Choose brightness limits with 0 < minimum <= peak < 1.')
        if custom:
            try:
                if hosted is not None and Path(custom).stat().st_size > 256 * 1024:
                    raise ValueError('Upload a camera JSON below 256 KB.')
                preset = json.loads(Path(custom).read_text())
                preset.update(brightness_minimum=float(low), brightness_maximum=float(high))
                if hosted is not None:
                    hosted.check_timing(preset)
                build_camera_path(preset)
                return preset
            except (ValueError, KeyError, TypeError) as error:
                raise gr.Error('Invalid camera JSON: {}'.format(error))
        if scene == 'custom':
            if not ply:
                raise gr.Error('Upload your model before previewing its automatic camera path.')
            if hosted is not None:
                hosted.check_ply(Path(ply))
            preset = estimate_custom_preset(ply)
        else:
            preset = copy.deepcopy(scene_preset(scene))
        if hosted is not None:
            hosted.check_timing(dict(duration_seconds=seconds, fps=fps))
        preset = retime_preset(preset, seconds, fps)
        preset.update(motion=MOTIONS[motion], duration_seconds=float(seconds), fps=int(fps),
                      turns_per_leg=float(turns), brightness_minimum=float(low), brightness_maximum=float(high))
        preset['forward_travel'] *= float(forward)
        preset['lateral_radius_x'] *= float(horizontal)
        preset['lateral_radius_y'] *= float(vertical)
        build_camera_path(preset)
        return preset

    def preview(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high):
        preset = preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high)
        folder = outputs / ('path-' + uuid.uuid4().hex)
        folder.mkdir()
        destination = folder / 'camera-path.json'
        destination.write_text(json.dumps(export_keyframes(preset), indent=2) + '\n')
        return plot_path(preset), str(destination), 'Trajectory updated. Download its camera JSON or render both videos.'

    def start_render():
        # Clear old media immediately, including any playing videos or WebGL frames.
        return ('Rendering…', '', '', '', '', '', gr.skip(), gr.skip(),
                gr.update(visible=False),
                gr.update(visible=True, value=render_progress_html('Preparing render')),
                gr.update(interactive=False, value='Rendering…'))

    def render_pair(scene, ply, mapper, custom, motion, forward, horizontal, vertical, turns, seconds, fps,
                    low, high, brightness_mode, fixed, size, hdr_ev):
        started = time.perf_counter()
        yield start_render()
        try:
            preset = preset_for(scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high)
            if hosted is not None:
                hosted.check_size(size)
                hosted.clean_results(outputs)
            model_file, tone_file = model_for(scene, ply, mapper)
            job = uuid.uuid4().hex
            folder = outputs / job
            folder.mkdir()
            trajectory = folder / 'camera-path.json'
            trajectory.write_text(json.dumps(preset, indent=2) + '\n')
            editable = folder / 'camera-path-editable.json'
            editable.write_text(json.dumps(export_keyframes(preset), indent=2) + '\n')
            common = ['--scene', scene if scene != 'custom' else 'bathroom', '--model_path', str(model_file),
                      '--tone_mapper', str(tone_file), '--trajectory', str(trajectory), '--output', str(folder),
                      '--max_size', str(int(size)), '--gpu_id', str(gpu_id), '--progress_json']
            env = os.environ.copy()
            env.update(TMPDIR=str(outputs / '.tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', OPENCV_IO_ENABLE_OPENEXR='1')
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            process_times = {}
            with LOCK:
                for mode in ['hdr', 'ldr']:
                    command = [str(renderer_python), str(ROOT / ('render_' + mode + '.py'))] + common
                    if mode == 'hdr':
                        # LDR exports the matching pair of point clouds once both videos are ready.
                        command += ['--exposure', str(hdr_ev), '--no_viewer']
                    elif brightness_mode == 'Fixed brightness':
                        command += ['--brightness', str(fixed)]
                    stage, completed, total = 'loading', 0, 1
                    phase_started = time.perf_counter()
                    last_update = 0.
                    log = folder / (mode + '-render.log')
                    events = (hosted.render_events(mode, command[2:], log) if hosted is not None
                              else renderer_events(command, env, log))
                    for event in events:
                        if event:
                            stage = event['stage']
                            completed, total = event['completed'], max(1, event['total'])
                        now = time.perf_counter()
                        if now - last_update < .4 and not event:
                            continue
                        last_update = now
                        base = 0. if mode == 'hdr' else .46
                        fractions = dict(loading=.01, calibration=.04,
                                         frames=.06 + .35 * completed / total,
                                         encoding=.42, point_clouds=.92, done=.99)
                        fraction = fractions.get(stage, .01)
                        if stage not in ['point_clouds', 'done']:
                            fraction += base
                        elif stage == 'done' and mode == 'hdr':
                            fraction = .45
                        label = 'Preparing point-cloud viewers' if stage == 'point_clouds' else 'Rendering {} video'.format(mode.upper())
                        detail = ('Frame {:,} / {:,}'.format(completed, total) if stage == 'frames' else
                                  dict(loading='Loading model', calibration='Calibrating HDR display',
                                       encoding='Encoding MP4', point_clouds='Preparing both interactive viewers',
                                       done='Finishing outputs').get(stage, 'Preparing renderer'))
                        yield ('Rendering…', gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                               gr.skip(), gr.skip(), gr.skip(),
                               gr.update(visible=True, value=render_progress_html(label, fraction, now - started, detail)),
                               gr.skip())
                    process_times[mode] = time.perf_counter() - phase_started
            originals = [(model_file, 'point_cloud.ply'), (tone_file, 'tone_mapper.pth')]
            for original, name in originals if hosted is None else []:
                target = folder / name
                try:
                    os.link(str(original), str(target))
                except OSError:
                    shutil.copy2(str(original), str(target))
            elapsed = time.perf_counter() - started
            (folder / 'render-timing.json').write_text(json.dumps(dict(
                total_seconds=elapsed, cli_seconds=process_times, frames_per_video=len(build_camera_path(preset)[0]),
                fps=int(preset['fps']), long_side_px=int(size)), indent=2) + '\n')
            base = '/results/' + job
            links = [('LDR video', 'ldr.mp4'), ('HDR video', 'hdr.mp4'),
                     ('LDR point cloud', 'assets/scenes/render/ldr.splat'),
                     ('HDR point cloud', 'assets/scenes/render/hdr.splat'),
                     ('Camera trajectory', 'camera-path-editable.json'), ('HDR display settings', 'tonemap.json'),
                     ('Raw HDR frame', 'frames_hdr/{:06d}.exr'.format(len(build_camera_path(preset)[0]) // 2))]
            if hosted is None:
                links[4:4] = [('Original point cloud', 'point_cloud.ply'), ('Tone mapper', 'tone_mapper.pth')]
            yield ('Render complete in {:.1f} s. Videos and point clouds are ready.'.format(elapsed),
                   video_html(base + '/ldr.mp4', 'Rendered LDR video'),
                   video_html(base + '/hdr.mp4', 'Rendered HDR video'),
                   viewer_html(base, 'ldr'), viewer_html(base, 'hdr', exposure_ev=hdr_ev), download_html(base, links),
                   plot_path(preset), str(editable), gr.update(visible=True), gr.update(visible=False),
                   gr.update(interactive=True, value='Render LDR + HDR'))
        except Exception as error:
            message = getattr(error, 'message', str(error))
            yield ('Render failed. Adjust the settings and try again.', gr.skip(), gr.skip(), gr.skip(),
                   gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=False),
                   gr.update(visible=True, value=render_progress_html('Render failed', 0., time.perf_counter() - started, message)),
                   gr.update(interactive=True, value='Render LDR + HDR'))
            # The failure is already shown in the result area. Finish the stream
            # normally so Gradio keeps the scene controls available for a retry.
            return

    def choose_scene(scene):
        preset = demo_scene_preset(scene) if scene != 'custom' else demo_scene_preset('bathroom')
        gallery_index = next((i for i, item in enumerate(SCENES) if item['id'] == scene), None)
        return (gr.update(visible=scene == 'custom'), preset.get('brightness_maximum', .5),
                plot_path(preset) if scene != 'custom' else None,
                *scene_preview(scene, outputs), scene_summary(scene), gr.update(selected_index=gallery_index),
                str(outputs / 'previews' / scene / 'camera-path-editable.json') if scene != 'custom' else None,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(interactive=True, value='Render LDR + HDR'))

    def select_thumbnail(event: gr.SelectData):
        return SCENES[int(event.index)]['id']

    def brightness_visibility(mode):
        is_fixed = mode == 'Fixed brightness'
        return gr.update(visible=is_fixed), gr.update(visible=not is_fixed)

    example = dict(version=1, convention='OpenCV C2W: +X right, +Y down, +Z forward',
                   duration_seconds=8, fps=30, camera_intrinsics=dict(width=720, height=720, fx=864, fy=864),
                   keyframes=[dict(time_seconds=t, camera_to_world=[[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]])
                              for t,z in [(0,0),(4,1),(8,0)]])
    initial = scene_preview('bathroom', outputs)
    thumbnails = [(str(ROOT / 'demo' / PREVIEWS[item['id']]['thumbnail']), item['label']) for item in SCENES]
    with gr.Blocks(
            title='HDR-GS Studio', css=(ROOT / 'demo/static/app.css').read_text(), theme=studio_theme(),
            js=(ROOT / 'demo/static/studio.js').read_text()) as demo:
        gr.HTML('<header class="app-header"><div><h1>HDR-GS <span>NeurIPS 2024</span></h1>'
                '<p>Novel-view synthesis &amp; interactive Gaussian exploration</p></div></header>', padding=False)
        with gr.Row(elem_id='scene-row', equal_height=True):
            with gr.Column(scale=0, min_width=100, elem_id='scene-toolbar'):
                gr.HTML('<div class="scene-heading">Choose a scene</div>', padding=False)
                upload_button = gr.Button('Upload model', elem_id='upload-model')
            with gr.Column(min_width=0, elem_id='scene-strip'):
                gallery = gr.Gallery(
                    thumbnails, columns=12, rows=1, height=90, allow_preview=False, selected_index=0,
                    object_fit='contain', show_label=False, container=False, interactive=False,
                    show_download_button=False, show_fullscreen_button=False, elem_id='scene-gallery')
        with gr.Group(visible=False, elem_classes='uploads') as uploads:
            with gr.Row():
                ply = gr.File(label='HDR-GS point_cloud.ply', file_types=['.ply'], type='filepath')
                mapper = gr.File(label='Matching tone_mapper.pth', file_types=['.pth', '.pt'], type='filepath')
            builtin_button = gr.Button('Use built-in scenes', size='sm', scale=0)
        with gr.Row(equal_height=False, elem_id='main-layout'):
            with gr.Column(scale=1, min_width=0, elem_id='control-panel'):
                with gr.Column(min_width=0, elem_id='control-fields'):
                    summary = gr.HTML(scene_summary('bathroom'), padding=False, elem_id='scene-summary')
                    scene = gr.Dropdown(
                        choices=[(item['label'], item['id']) for item in SCENES] + [('Custom model', 'custom')],
                        value='bathroom', label='Scene', visible=False)
                    gr.HTML('<div class="control-section-title">Camera</div>', padding=False)
                    motion = gr.Dropdown(list(MOTIONS), value='Spiral forward & backward', label='Camera path')
                    with gr.Group(elem_id='camera-preview'):
                        trajectory_plot = gr.Plot(value=plot_path(demo_scene_preset('bathroom')), show_label=False,
                                                  elem_id='trajectory-plot')
                        gr.HTML('<div class="trajectory-key">X: right · Z: forward · −Y: up<br>Start-camera coordinates<br>Solid: forward · Dashed: backward</div>', padding=False)
                        with gr.Row(elem_classes='trajectory-actions'):
                            preview_button = gr.Button('Preview path', elem_id='preview-button', scale=1, min_width=80)
                            trajectory_download = gr.DownloadButton(
                                'Camera JSON', value=str(outputs / 'previews/bathroom/camera-path-editable.json'),
                                elem_id='download-camera', scale=1, min_width=80)
                    forward = gr.Slider(.1, 2, 1, step=.05, label='Forward travel × scene preset')
                    with gr.Accordion('Camera settings', open=False, elem_id='camera-options'):
                        with gr.Row():
                            horizontal = gr.Slider(0, 2, 1, step=.05, label='Horizontal amplitude ×', min_width=80)
                            vertical = gr.Slider(0, 2, 1, step=.05, label='Vertical amplitude ×', min_width=80)
                        turns = gr.Slider(.5, 4, 2, step=.5, label='Spiral turns per leg')
                        seconds = gr.Slider(2, 8 if hosted else 30, 8, step=1, label='Duration (seconds)')
                        with gr.Row():
                            fps = gr.Dropdown([12, 24, 30] if hosted else [12, 24, 30, 60], value=30, label='FPS', min_width=70)
                            size = gr.Dropdown([240, 480, 720] if hosted else [240, 480, 720, 960], value=720, label='Video long side (px)', min_width=110)
                    with gr.Accordion('LDR / HDR appearance', open=False, elem_id='brightness-options'):
                        brightness_mode = gr.Radio(['Project-page gradient', 'Fixed brightness'],
                                                   value='Project-page gradient', label='LDR brightness')
                        fixed = gr.Slider(.01, .9, .35, step=.005, label='Fixed mean luminance', visible=False)
                        with gr.Group() as gradient_controls:
                            low = gr.Slider(.01, .5, .065, step=.005, label='Gradient minimum')
                            high = gr.Slider(.1, .9, scene_preset('bathroom')['brightness_maximum'],
                                             step=.005, label='Gradient peak')
                        hdr_ev = gr.Slider(-4, 4, 0, step=.1, label='HDR display exposure (EV)')
                        gr.Markdown('HDR uses one fixed Reinhard calibration for the full clip.', elem_classes='helper')
                    with gr.Accordion('Custom camera path', open=False, elem_id='custom-camera'):
                        custom = gr.File(label='Upload camera JSON', file_types=['.json'], type='filepath')
                        gr.Markdown('Overrides the camera controls. Move forward first, then return to the starting pose.',
                                    elem_classes='helper')
                        with gr.Accordion('Camera JSON format', open=False):
                            gr.Markdown('OpenCV camera-to-world: **+X right, +Y down, +Z forward**. '
                                        'The last keyframe closes the loop.', elem_classes='helper')
                            gr.Code(json.dumps(example, indent=2), language='json', label='Editable trajectory format',
                                    elem_classes='format-code')
                render_button = gr.Button('Render LDR + HDR', variant='primary', elem_id='render-button')
                status = gr.Markdown(initial[5], elem_id='render-status')
            with gr.Column(scale=4, min_width=0, elem_id='workspace'):
                render_progress = gr.HTML('', visible=False, padding=False, elem_id='render-progress')
                with gr.Column(min_width=0, elem_id='preview-results') as preview_results:
                    with gr.Row(elem_id='video-row'):
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML('<div class="panel-heading">LDR video <span>Learned tone mapper</span></div>', padding=False)
                            ldr_video = gr.HTML(initial[0], padding=False, elem_id='ldr-video', elem_classes='media-html')
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML('<div class="panel-heading">HDR video <span>Reinhard display</span></div>', padding=False)
                            hdr_video = gr.HTML(initial[1], padding=False, elem_id='hdr-video', elem_classes='media-html')
                    with gr.Row(elem_id='point-row'):
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML('<div class="panel-heading">LDR point cloud <span>Interactive</span></div>', padding=False)
                            ldr_viewer = gr.HTML(initial[2], padding=False, elem_id='ldr-viewer', elem_classes='media-html')
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML('<div class="panel-heading">HDR point cloud <span>Interactive</span></div>', padding=False)
                            hdr_viewer = gr.HTML(initial[3], padding=False, elem_id='hdr-viewer', elem_classes='media-html')
                    with gr.Accordion('Downloads · videos & point clouds', open=False, elem_id='result-downloads'):
                        extra = ('After rendering, download an EXR frame with full HDR radiance.' if hosted
                                 else 'After rendering, download the original model, tone mapper and an EXR frame with full HDR radiance.')
                        gr.HTML('<p class="downloads-note">MP4 previews play on standard screens. '
                                'Splat files are interactive previews. ' + extra + '</p>', padding=False)
                        downloads = gr.HTML(initial[4], padding=False)
        camera_inputs = [scene, ply, custom, motion, forward, horizontal, vertical, turns, seconds, fps, low, high]
        gallery.select(select_thumbnail, outputs=[scene], queue=False, api_name=False, show_progress='hidden')
        upload_button.click(lambda: 'custom', outputs=[scene], queue=False, api_name=False, show_progress='hidden')
        builtin_button.click(lambda: 'bathroom', outputs=[scene], queue=False, api_name=False, show_progress='hidden')
        scene.change(choose_scene, [scene], [uploads, high, trajectory_plot, ldr_video, hdr_video,
                     ldr_viewer, hdr_viewer, downloads, status, summary, gallery, trajectory_download,
                     preview_results, render_progress, render_button], show_progress='hidden')
        brightness_mode.change(brightness_visibility, [brightness_mode], [fixed, gradient_controls],
                               queue=False, api_name=False, show_progress='hidden')
        preview_button.click(preview, camera_inputs, [trajectory_plot, trajectory_download, status],
                             api_name='preview_trajectory', show_progress='minimal')
        render_outputs = [status, ldr_video, hdr_video, ldr_viewer, hdr_viewer, downloads,
                          trajectory_plot, trajectory_download, preview_results, render_progress, render_button]
        render_button.click(start_render, outputs=render_outputs, queue=False, api_name=False,
                            show_progress='hidden').then(
            render_pair, [scene, ply, mapper, custom, motion, forward, horizontal, vertical,
                          turns, seconds, fps, low, high, brightness_mode, fixed, size, hdr_ev],
            render_outputs, api_name='render_pair', concurrency_limit=1, show_progress='hidden')
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
    app.mount('/studio-assets', StaticFiles(directory=str(ROOT / 'demo/static')), name='studio-assets')
    app = gr.mount_gradio_app(app, demo.queue(max_size=8), path='/', allowed_paths=[str(args.outputs.resolve()), str(ROOT / 'demo/static')])
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
