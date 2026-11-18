"""Shared HDR-GS video renderer used by the two CLIs and the Gradio worker."""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_paths import ROOT, build_camera_path, retime_preset, scene_preset, scene_registry
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.hdr_visualization import fit_reinhard, tonemap_hdr, write_exr
from utils.sh_utils import eval_sh
from utils.splat_io import compact_splats, gzip_bytes
from utils.viewer_cache import color_key, file_sha256, geometry_key, read_entry, restore_entry, store_entry


def make_camera(c2w, intrinsics, max_size, crop=1):
    scale = max_size / max(intrinsics['width'], intrinsics['height'])
    width = max(2, int(intrinsics['width'] * scale) // 2 * 2)
    height = max(2, int(intrinsics['height'] * scale) // 2 * 2)
    fovx = 2 * np.arctan(intrinsics['width'] / (2 * intrinsics['fx'] * crop))
    fovy = 2 * np.arctan(intrinsics['height'] / (2 * intrinsics['fy'] * crop))
    w2c = np.linalg.inv(c2w)
    view = torch.tensor(getWorld2View2(w2c[:3, :3].T.astype('float32'), w2c[:3, 3].astype('float32'))).T.cuda()
    projection = getProjectionMatrix(.01, 100, fovx, fovy).T.cuda()
    return SimpleNamespace(FoVx=fovx, FoVy=fovy, image_width=width, image_height=height,
                           world_view_transform=view, full_proj_transform=view @ projection,
                           camera_center=view.inverse()[3, :3])


def resolve_model(path):
    path = Path(path)
    if path.is_file():
        return path
    if (path / 'point_cloud.ply').is_file():
        return path / 'point_cloud.ply'
    candidates = list((path / 'point_cloud').glob('iteration_*/point_cloud.ply'))
    if candidates:
        return max(candidates, key=lambda p: int(p.parent.name.split('_')[-1]))
    raise ValueError('No point_cloud.ply found in {}'.format(path))


def add_arguments(parser, mode):
    parser.add_argument('--scene', default='bathroom', choices=[s['id'] for s in scene_registry()])
    parser.add_argument('--models_root', type=Path, default=Path('pretrained_weights'))
    parser.add_argument('--model_path', type=Path, help='Point-cloud PLY, checkpoint directory, or training run')
    parser.add_argument('--tone_mapper', type=Path, help='Defaults to tone_mapper.pth beside the PLY')
    parser.add_argument('--trajectory', type=Path, help='Procedural preset or custom keyframe JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--fps', type=int, help='Output frame rate; defaults to 30 for built-in paths')
    parser.add_argument('--duration', type=float, help='Output duration in seconds; defaults to 8 for built-in paths')
    parser.add_argument('--max_size', type=int, default=720)
    parser.add_argument('--preview_only', action='store_true', help='Render calibration/contact views only')
    parser.add_argument('--no_viewer', action='store_true', help='Skip interactive viewer exports')
    parser.add_argument('--progress_json', action='store_true', help='Emit structured progress for the Gradio UI')
    parser.add_argument('--viewer_max_points', type=int, default=600000)
    parser.add_argument('--viewer_cache', type=Path, help='Reusable HDR geometry and LDR-color cache; defaults beside output directories')
    if mode == 'ldr':
        parser.add_argument('--brightness', type=float, help='Fixed mean luminance in (0,1); omit for project-page gradient')
    else:
        parser.add_argument('--key', type=float, default=.18)
        parser.add_argument('--white_percentile', type=float, default=99.95)
        parser.add_argument('--saturation', type=float, default=1.)
        parser.add_argument('--exposure', type=float, default=0., help='Fixed HDR display exposure offset in EV')
        parser.add_argument('--save_exr', action='store_true', help='Save all original float32 HDR frames')
    return parser


def write_splat(model, colors, path, keep, selected_colors=False):
    data = np.zeros((len(keep), 32), dtype=np.uint8)
    for offset, values in [(0, model.get_xyz), (12, model.get_scaling)]:
        xyz = values[keep].cpu().numpy().astype('<f4')
        data[:, offset:offset + 12] = np.ascontiguousarray(xyz).view(np.uint8).reshape(-1, 12)
    values = colors if selected_colors else colors[keep]
    data[:, 24:27] = np.round(np.clip(values.cpu().numpy(), 0, 1) * 255).astype('uint8')
    data[:, 27] = np.round(np.clip(model.get_opacity[keep, 0].cpu().numpy(), 0, 1) * 255).astype('uint8')
    data[:, 28:32] = np.round(np.clip(model.get_rotation[keep].cpu().numpy() * 128 + 128, 0, 255)).astype('uint8')
    data.tofile(str(path))


class SceneRenderer:
    def __init__(self, model_file, tone_mapper, preset, max_size):
        self.model = GaussianModel(3)
        self.model.load_ply(str(model_file))
        self.has_mapper = tone_mapper.is_file()
        self.mapper_hash = file_sha256(tone_mapper) if self.has_mapper else None
        if self.has_mapper:
            self.model.load_tonemapper(str(tone_mapper))
            for channel in 'rgb':
                getattr(self.model, 'tone_mapper_' + channel).cuda().eval()
        self.preset = preset
        self.max_size = max_size
        self.pipe = SimpleNamespace(debug=False, convert_SHs_python=True, compute_cov3D_python=False)
        self.background = torch.zeros(3, device='cuda')

    def camera(self, c2w, size=None):
        return make_camera(c2w, self.preset['camera_intrinsics'], size or self.max_size,
                           self.preset.get('viewport_crop', 1.))

    def raw(self, c2w):
        image = render(self.camera(c2w), self.model, self.pipe, self.background,
                       iteration=30000, render_mode='hdr')['render']
        result = image.permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite rendered HDR radiance')
        return result

    def ldr(self, c2w, target, initial_ev=0., size=None):
        """Solve actual learned-MLP exposure, never multiply display pixels."""
        if not self.has_mapper:
            raise ValueError('LDR rendering requires the matching tone_mapper.pth')
        camera = self.camera(c2w, size)
        base = float(self.preset.get('base_exposure', 1.))
        # Geometry visibility does not depend on exposure. Evaluate the MLP only
        # for splats that rasterize into this view, and cache their SH values
        # across exposure solves. This leaves the native training renderer intact.
        visible = render(camera, self.model, self.pipe, self.background, 30000,
                         override_color=torch.zeros_like(self.model.get_xyz), render_mode='hdr')['visibility_filter']
        indices = torch.where(visible)[0]
        if not len(indices):
            raise ValueError('The camera path contains a view outside the Gaussian reconstruction')
        subset = SimpleNamespace(active_sh_degree=self.model.active_sh_degree, max_sh_degree=self.model.max_sh_degree)
        for name in ['get_xyz', 'get_features', 'get_opacity', 'get_scaling', 'get_rotation']:
            setattr(subset, name, getattr(self.model, name)[indices])
        for channel in 'rgb':
            setattr(subset, 'tone_mapper_' + channel, getattr(self.model, 'tone_mapper_' + channel))
        directions = subset.get_xyz - camera.camera_center
        directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        log_rgb = eval_sh(subset.active_sh_degree, subset.get_features.transpose(1, 2), directions) + .5
        lo, hi, ev, best = -24., 24., float(np.clip(initial_ev, -24, 24)), None
        for _ in range(22):
            camera.exps = np.array([base * 2 ** ev])
            image = render(camera, subset, self.pipe, self.background, 30000, override_color=log_rgb, render_mode='ldr')['render'].clamp(0, 1)
            mean = float((image[0] * .2126 + image[1] * .7152 + image[2] * .0722).mean())
            error = mean - target
            if best is None or abs(error) < best[0]:
                best = (abs(error), image, mean, ev)
            if abs(error) <= .0003:
                break
            if error < 0:
                lo = ev
            else:
                hi = ev
            # A secant estimate across the current bracket becomes cheap once
            # the previous frame's exposure has brought us close to the target.
            if _ == 0:
                ev = float(np.clip(ev - error * 8, lo + 1e-7, hi - 1e-7))
            else:
                denom = mean - previous_mean
                proposal = ev - error * (ev - previous_ev) / denom if abs(denom) > 1e-7 else (lo + hi) / 2
                previous_candidate = ev
                ev = proposal if lo + 1e-7 < proposal < hi - 1e-7 else (lo + hi) / 2
                previous_ev, previous_mean = previous_candidate, mean
                continue
            previous_ev, previous_mean = initial_ev, mean
        error, image, mean, ev = best
        if error > .002:
            raise ValueError('Requested brightness {:.4f} is unreachable (closest {:.4f})'.format(target, mean))
        return (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8'), mean, ev

    def export_viewer(self, output, poses, settings, mean_target, exposure_ev, max_points, model_hash, cache_root=None):
        model = self.model
        folder = output / 'assets/scenes' / 'render'
        folder.mkdir(parents=True, exist_ok=True)
        # Output files may be hard links to immutable cache entries. Remove them
        # before generating a different exposure into an existing output folder.
        for name in ['hdr.splat', 'ldr.splat', 'gaussians.splat.gz', 'ldr.rgb.gz']:
            if (folder / name).exists():
                (folder / name).unlink()
        reference = self.camera(poses[0])
        intr = self.preset['camera_intrinsics']
        crop = self.preset.get('viewport_crop', 1.)
        hdr_key = geometry_key(model_hash, poses[0], intr, crop, max_points)
        cache_root = Path(cache_root) if cache_root else output.parent / '.viewer-cache'
        hdr_folder = cache_root / 'hdr' / hdr_key
        hdr_info = read_entry(hdr_folder)
        hdr_hit = hdr_info is not None
        keep, log_rgb = None, None

        def selected_colors():
            nonlocal keep, log_rgb
            if keep is None:
                opacity = model.get_opacity[:, 0]
                scores = model.get_scaling.prod(dim=1) * opacity
                keep = torch.where(opacity > 1 / 255)[0]
                if len(keep) > max_points:
                    keep = keep[torch.topk(scores[keep], max_points, sorted=False).indices]
                keep = keep[torch.argsort(scores[keep], descending=True)]
            if log_rgb is None:
                direction = model.get_xyz[keep] - reference.camera_center
                direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
                log_rgb = eval_sh(model.active_sh_degree, model.get_features[keep].transpose(1, 2), direction) + .5
            return log_rgb

        if hdr_hit:
            restore_entry(hdr_folder, folder, ['hdr.splat', 'gaussians.splat.gz'])
        else:
            radiance = selected_colors().exp()
            hdr_range = max(float(torch.quantile(radiance.reshape(-1), .99999)), 1.)
            encoded = torch.log1p(radiance.clamp(0, hdr_range)) / math.log1p(hdr_range)
            write_splat(model, encoded, folder / 'hdr.splat', keep, selected_colors=True)
            packing = compact_splats(folder)
            hdr_info = dict(hdrRange=hdr_range, webPointCount=len(keep), **packing)
            store_entry(hdr_folder, {name: folder / name for name in ['hdr.splat', 'gaussians.splat.gz']}, hdr_info)

        packing = {key: hdr_info[key] for key in ['gaussianEncoding', 'gaussianAsset', 'hdrSplatSHA256']}
        ldr_exposure, ldr_hit = None, False
        if self.has_mapper:
            ldr_key = color_key(hdr_key, self.mapper_hash, mean_target, self.preset.get('base_exposure', 1.))
            ldr_folder = cache_root / 'ldr' / ldr_key
            ldr_info = read_entry(ldr_folder)
            ldr_hit = ldr_info is not None
            if ldr_hit:
                restore_entry(ldr_folder, folder, ['ldr.splat', 'ldr.rgb.gz'])
            else:
                # The geometry/HDR splat is unchanged. Re-evaluate only the
                # selected Gaussians' MLP colors for a new LDR brightness.
                log_colors = selected_colors()
                _, _, ev = self.ldr(poses[0], mean_target, size=240)
                ldr_exposure = float(self.preset.get('base_exposure', 1.)) * 2 ** ev
                channels = []
                for index, channel in enumerate('rgb'):
                    value = log_colors[:, index:index + 1] + math.log(ldr_exposure)
                    for layer in getattr(model, 'tone_mapper_' + channel):
                        value = layer(value)
                    channels.append(value)
                rgb = np.round(np.clip(torch.cat(channels, dim=1).cpu().numpy(), 0, 1) * 255).astype('uint8')
                data = np.fromfile(str(folder / 'hdr.splat'), dtype=np.uint8).reshape(-1, 32)
                if len(data) != len(rgb):
                    raise ValueError('Cached geometry does not match the source model')
                data[:, 24:27] = rgb
                data.tofile(str(folder / 'ldr.splat'))
                (folder / 'ldr.rgb.gz').write_bytes(gzip_bytes(rgb.tobytes()))
                ldr_info = dict(ldrExposure=ldr_exposure, ldrSplatSHA256=file_sha256(folder / 'ldr.splat'))
                store_entry(ldr_folder, {name: folder / name for name in ['ldr.splat', 'ldr.rgb.gz']}, ldr_info)
            ldr_exposure = ldr_info['ldrExposure']
            packing.update(ldrColorAsset='ldr.rgb.gz', ldrSplatSHA256=ldr_info['ldrSplatSHA256'])

        metadata = dict(id='render', label=self.preset.get('name', 'Custom scene'), group='synthetic',
            sourcePointCount=len(model.get_xyz), webPointCount=hdr_info['webPointCount'], sourceSHA256=model_hash,
            hdrEncoding='log1p-radiance', hdrRange=hdr_info['hdrRange'], hdrToneMap=settings,
            hdrDisplayExposureEV=exposure_ev, ldrExposure=ldr_exposure,
            ldrBrightness=mean_target, camera=dict(worldToCam=reference.world_view_transform.T.cpu().numpy().reshape(-1).tolist(),
                width=reference.image_width, height=reference.image_height,
                fx=intr['fx'] * reference.image_width / intr['width'] * crop,
                fy=intr['fy'] * reference.image_height / intr['height'] * crop),
            webColorPolicy='Reference-view SH colors. Learned MLP LDR; log-encoded linear HDR, decoded before alpha composition and Reinhard display mapping.',
            viewerCache=dict(hdr_hit=hdr_hit, ldr_hit=ldr_hit))
        metadata.update(packing)
        (folder / 'scene.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (output / 'assets/scenes.json').write_text(json.dumps([metadata], indent=2) + '\n')
        source = ROOT / 'demo/viewer'
        if source.is_dir():
            _copy_viewer(source, output / 'viewer')
        return metadata


def _copy_viewer(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(str(path), str(destination / path.name))


def contact_sheet(samples, output):
    side = 240
    canvas = Image.new('RGB', (side * 3, side * 3), 'white')
    for index, frame in enumerate(samples[:9]):
        picture = Image.fromarray(frame)
        picture.thumbnail((side, side))
        canvas.paste(picture, ((index % 3) * side + (side - picture.width) // 2,
                              (index // 3) * side + (side - picture.height) // 2))
    canvas.save(str(output), quality=92)


@torch.no_grad()
def render_video(mode, args):
    started = time.perf_counter()
    timings = {}

    def report(stage, completed=0, total=1):
        if getattr(args, 'progress_json', False):
            print('HDRGS_PROGRESS ' + json.dumps(dict(mode=mode, stage=stage, completed=int(completed),
                  total=int(total), elapsed_seconds=round(time.perf_counter() - started, 3))), flush=True)

    report('loading')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.set_num_threads(2)
    if args.max_size < 2 or args.viewer_max_points < 1:
        raise ValueError('max_size must be >=2 and viewer_max_points must be positive')
    if mode == 'ldr' and args.brightness is not None and not 0 < args.brightness < 1:
        raise ValueError('--brightness must be between 0 and 1')
    preset = json.loads(args.trajectory.read_text()) if args.trajectory else scene_preset(args.scene)
    duration_override = getattr(args, 'duration', None)
    if args.trajectory is None or duration_override is not None or args.fps is not None:
        duration = duration_override if duration_override is not None else (8 if args.trajectory is None else preset['duration_seconds'])
        rate = args.fps if args.fps is not None else (30 if args.trajectory is None else preset['fps'])
        preset = retime_preset(preset, duration, rate)
    poses, times = build_camera_path(preset, args.fps)
    fps = int(preset['fps'] if args.fps is None else args.fps)
    entry = next(s for s in scene_registry() if s['id'] == args.scene)
    model_path = args.model_path
    if model_path is None:
        candidates = [args.models_root / args.scene, args.models_root / entry['model_directory']]
        model_path = next((path for path in candidates if path.exists()), candidates[-1])
    model_file = resolve_model(model_path)
    mapper = args.tone_mapper or model_file.parent / 'tone_mapper.pth'
    model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    renderer = SceneRenderer(model_file, mapper, preset, args.max_size)
    if mode == 'ldr' and not renderer.has_mapper:
        raise ValueError('Missing tone_mapper.pth; use --tone_mapper for the matching weights')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for previous_mode in ['ldr', 'hdr']:
        previous_metadata = output / (previous_mode + '-metadata.json')
        if previous_metadata.is_file() and json.loads(previous_metadata.read_text()).get('model_sha256') != model_hash:
            raise ValueError('Output contains a different checkpoint. Choose a new --output directory.')
    torch.cuda.synchronize()
    timings['loading_seconds'] = time.perf_counter() - started
    calibration_started = time.perf_counter()
    report('calibration')
    settings_path = output / 'tonemap.json'
    raw_samples = {}
    if mode == 'ldr' and settings_path.is_file():
        # Paired renders reuse the HDR calibration; no duplicate HDR samples.
        settings = json.loads(settings_path.read_text())
    else:
        sample_ids = sorted(set(np.linspace(0, len(poses) - 1, 33).round().astype(int).tolist() + [len(poses) // 2]))
        raw_samples = {index: renderer.raw(poses[index]) for index in sample_ids}
        settings = fit_reinhard(raw_samples.values(), key=getattr(args, 'key', .18),
            white_percentile=getattr(args, 'white_percentile', 99.95), saturation=getattr(args, 'saturation', 1.))
        settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    timings['calibration_seconds'] = time.perf_counter() - calibration_started
    video_started = time.perf_counter()
    report('frames', 0, len(poses))
    np.save(str(output / 'camera_to_world.npy'), poses)
    (output / 'trajectory.json').write_text(json.dumps(preset, indent=2) + '\n')
    low, high = float(preset.get('brightness_minimum', .065)), float(preset.get('brightness_maximum', .5))
    if not 0 < low <= high < 1:
        raise ValueError('Brightness limits must satisfy 0 < minimum <= maximum < 1')
    peak = 1 - np.abs(2 * times / float(preset['duration_seconds']) - 1)
    brightness = np.full(len(poses), args.brightness) if mode == 'ldr' and args.brightness is not None else low + (high - low) * peak
    exposure_ev = float(getattr(args, 'exposure', 0.))
    if mode == 'ldr' and (output / 'hdr-metadata.json').is_file():
        exposure_ev = float(json.loads((output / 'hdr-metadata.json').read_text())['fixed_exposure_ev'])
    ev = 0.
    frames_for_contact, stats = [], []
    selected = set(np.linspace(0, len(poses) - 1, 9).round().astype(int).tolist())
    frame_ids = sorted(selected) if args.preview_only else range(len(poses))
    temporary = output / (mode + '.part.mp4')
    writer = None if args.preview_only else imageio.get_writer(str(temporary), fps=fps, codec='libx264', macro_block_size=2,
        ffmpeg_params=['-crf', '18', '-preset', 'medium', '-bf', '0', '-g', str(fps * 2),
            '-keyint_min', str(fps * 2), '-sc_threshold', '0', '-movflags', '+faststart',
            '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-color_primaries', 'bt709',
            '-color_trc', 'iec61966-2-1' if mode == 'hdr' else 'bt709', '-colorspace', 'bt709', '-threads', '2'])
    try:
        for index in frame_ids:
            raw = raw_samples.pop(index) if index in raw_samples else renderer.raw(poses[index]) if mode == 'hdr' else None
            if mode == 'hdr':
                frame = tonemap_hdr(raw, settings, exposure_ev)
                if getattr(args, 'save_exr', False) or index in [0, len(poses) // 2, len(poses) - 1]:
                    write_exr(output / 'frames_hdr' / ('{:06d}.exr'.format(index)), raw)
            else:
                frame, actual, ev = renderer.ldr(poses[index], float(brightness[index]), ev)
            if writer is not None:
                writer.append_data(frame)
            if index in selected:
                frames_for_contact.append(frame)
            if index == 0 or index == len(poses) // 4:
                Image.fromarray(frame).save(str(output / (mode + '.webp')), quality=94)
            item = dict(frame=int(index), seconds=float(times[index]))
            if raw is not None:
                item['radiance_max'] = float(raw.max())
            if mode == 'ldr':
                item.update(brightness_target=float(brightness[index]), brightness_actual=actual,
                            exposure=float(preset.get('base_exposure', 1.)) * 2 ** ev)
            stats.append(item)
            if (index + 1) % max(1, fps // 4) == 0 or index == len(poses) - 1:
                report('frames', index + 1, len(poses))
            if index % fps == 0:
                print('{} FRAME {} / {}'.format(mode.upper(), index, len(poses)), flush=True)
    finally:
        if writer is not None:
            report('encoding')
            writer.close()
    if writer is not None:
        temporary.replace(output / (mode + '.mp4'))
    contact_sheet(frames_for_contact, output / (mode + '-contact.jpg'))
    timings['video_and_encoding_seconds'] = time.perf_counter() - video_started
    viewer_started = time.perf_counter()
    viewer = None
    if not args.no_viewer:
        report('point_clouds')
        viewer_brightness = args.brightness if mode == 'ldr' and args.brightness is not None else (low + high) / 2
        if mode == 'hdr' and (output / 'ldr-metadata.json').is_file():
            previous_ldr = json.loads((output / 'ldr-metadata.json').read_text())
            if previous_ldr.get('brightness_mode') == 'fixed':
                viewer_brightness = float(previous_ldr['frame_stats'][0]['brightness_target'])
        viewer = renderer.export_viewer(output, poses, settings, viewer_brightness, exposure_ev,
                                       args.viewer_max_points, model_hash, getattr(args, 'viewer_cache', None))
    torch.cuda.synchronize()
    timings['point_clouds_seconds'] = time.perf_counter() - viewer_started
    timings['total_seconds'] = time.perf_counter() - started
    metadata = dict(render_mode=mode, frames=len(poses), fps=fps, seconds=len(poses) / fps,
        timings=timings, gpu=torch.cuda.get_device_name(0),
        model_sha256=model_hash, model_matches_project_page=model_hash == preset.get('model_sha256'),
        trajectory=preset, calibration=settings, fixed_exposure_ev=exposure_ev,
        visualization='extended Reinhard + sRGB; SDR H.264 preview' if mode == 'hdr' else 'learned HDR-GS tone mapper',
        brightness_mode=('fixed' if args.brightness is not None else 'project-page gradient') if mode == 'ldr' else 'fixed HDR display exposure',
        frame_stats=stats, resolution=[frame.shape[1], frame.shape[0]], viewer=viewer,
        video=None if args.preview_only else mode + '.mp4')
    (output / (mode + '-metadata.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    report('done', 1, 1)
    print('{} {} READY {}'.format(mode.upper(), 'PREVIEW' if args.preview_only else 'VIDEO', output), flush=True)
    return metadata


def cli(mode):
    parser = add_arguments(argparse.ArgumentParser(description='Render an HDR-GS {} video and interactive Gaussian previews'.format(mode.upper())), mode)
    args = parser.parse_args()
    try:
        render_video(mode, args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
