"""Run the HDR-GS refinement stages described by a scene YAML."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
SCENES = ('bathroom', 'bear', 'chair', 'desk', 'diningroom', 'dog', 'sofa', 'sponza',
          'box', 'computer', 'flower', 'luckycat')
SYNTHETIC = SCENES[:8]


@dataclass(frozen=True)
class Recipe:
    l1: float
    mse: float
    dssim: float
    perceptual: float
    hdr_weight: float = .6
    lr_scale: float = 1.
    freeze_hdr: bool = False
    native_dssim: float = 0.
    mapper_lr_scale: float = 1.


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_scene(name, source, output, base_config):
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene import Scene, GaussianModel
    import yaml
    parser = argparse.ArgumentParser()
    models = ModelParams(parser)
    pipelines = PipelineParams(parser)
    optimizer = OptimizationParams(parser)
    args = parser.parse_args([])
    config = yaml.safe_load(base_config.read_text())
    vars(args).update(config)
    args.eval = True
    args.syn = name in SYNTHETIC
    args.source_path = str(ROOT / args.source_path)
    args.model_path = str(output)
    args.data_device = 'cpu'
    args.render_video = False
    dataset = models.extract(args)
    logger = logging.getLogger('refinement')
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, logger, load_path=str(source), shuffle=False)
    if not dataset.syn:
        dataset.train_exps = sorted({float(camera.exps.item()) for camera in scene.getTrainCameras()})
        test_exposures = {float(camera.exps.item()) for camera in scene.getTestCameras()}
        dataset.test_exps = sorted(test_exposures - set(dataset.train_exps))
        assert len(dataset.train_exps) == 3 and len(dataset.test_exps) == 2
    background = torch.tensor(
        [1., 1., 1.] if dataset.white_background else [0., 0., 0.], device='cuda')
    return scene, model, dataset, pipelines.extract(args), optimizer.extract(args), background


def hdr_transform(radiance):
    # Match the released testing code's per-image normalization and mu-law.
    values = torch.clamp(radiance / radiance.max(), 0., 1.)
    return torch.log(values * 5000 + 1) / torch.log(torch.tensor(5001.))


def prepare_model(model, source):
    model.load_ply(str(source / 'point_cloud.ply'))
    model.load_tonemapper(str(source / 'tone_mapper.pth'))
    for channel in 'rgb':
        getattr(model, 'tone_mapper_' + channel).cuda()



def benchmark(cameras, model, dataset, pipe, background, final=False):
    from gaussian_renderer import render
    from lpipsPyTorch import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    accum = defaultdict(lambda: [0., 0., 0., 0])
    with torch.no_grad():
        for camera in cameras:
            output = render(camera, model, pipe, background, iteration=0, render_mode='ldr')['render']
            target = camera.original_image.to('cuda').clamp(0., 1.)
            if camera.exps in dataset.train_exps:
                track = 'LDR-OE'
            elif camera.exps in dataset.test_exps:
                track = 'LDR-NE'
            else:
                raise ValueError('Unexpected exposure {}'.format(camera.exps))
            pairs = [(track, output.clamp(0., 1.), target)]
            if dataset.syn:
                radiance = render(camera, model, pipe, background, iteration=0, render_mode='hdr')['render']
                pairs.append(('HDR', hdr_transform(radiance), camera.hdr_image.to('cuda').clamp(0., 1.)))
            for mode, image, reference in pairs:
                values = torch.stack((psnr(image, reference).mean(), ssim(image, reference).mean(),
                                      lpips(image, reference, net_type='alex').mean())).cpu().tolist()
                for index, value in enumerate(values):
                    accum[mode][index] += float(value)
                accum[mode][3] += 1
    if final:
        assert len(cameras) == 85 and accum['LDR-OE'][3] == 51 and accum['LDR-NE'][3] == 34
        if dataset.syn:
            assert accum['HDR'][3] == 85
    return {track: dict(psnr=p/n, ssim=s/n, lpips=l/n, count=n)
            for track, (p, s, l, n) in sorted(accum.items())}


def patch_loss(image, target, rng, recipe):
    from lpipsPyTorch import lpips
    from utils.loss_utils import ssim
    height, width = image.shape[-2:]
    size = min(256, height, width)
    y = rng.randrange(height - size + 1)
    x = rng.randrange(width - size + 1)
    patch = image[:, y:y+size, x:x+size].clamp(0., 1.)
    reference = target[:, y:y+size, x:x+size]
    loss = recipe.perceptual * lpips(patch, reference, net_type='alex').mean()
    if recipe.native_dssim:
        loss += recipe.native_dssim * (1. - ssim(patch, reference))
    return loss


def reconstruction_loss(image, target, recipe, rng, use_perceptual=True):
    from utils.loss_utils import ssim
    residual = image - target
    loss = recipe.l1 * residual.abs().mean() + recipe.mse * residual.square().mean()
    pooled = torch.nn.functional.avg_pool2d
    loss += recipe.dssim * (1. - ssim(pooled(image[None], 2), pooled(target[None], 2)))
    if use_perceptual and recipe.perceptual:
        loss += patch_loss(image, target, rng, recipe)
    return loss


def refine(model, scene, opt, pipe, background, cameras, steps, recipe, synthetic, seed, rates):
    from gaussian_renderer import render
    model.spatial_lr_scale = scene.cameras_extent
    model.training_setup(opt)
    learning_rates = dict(rates)
    learning_rates['xyz'] *= scene.cameras_extent
    if recipe.freeze_hdr:
        for group in model.optimizer.param_groups:
            if not group['name'].startswith('tone_mapper_'):
                for parameter in group['params']:
                    parameter.requires_grad_(False)
    rng = random.Random(seed)
    stack = []
    start = time.monotonic()
    for step in range(1, steps + 1):
        decay = 0.25 + 0.75 * (1 + math.cos(math.pi * (step-1) / max(steps-1, 1))) / 2
        for group in model.optimizer.param_groups:
            rate = recipe.lr_scale * decay * learning_rates[group['name']]
            if group['name'].startswith('tone_mapper_'):
                rate *= recipe.mapper_lr_scale
            group['lr'] = rate
        if not stack:
            stack = list(cameras)
            rng.shuffle(stack)
        camera = stack.pop()
        target = camera.original_image.to('cuda')
        image = render(camera, model, pipe, background, iteration=step, render_mode='ldr')['render']
        loss = reconstruction_loss(image, target, recipe, rng)
        if synthetic and recipe.hdr_weight:
            radiance = render(camera, model, pipe, background, iteration=step, render_mode='hdr')['render']
            loss += recipe.hdr_weight * reconstruction_loss(
                hdr_transform(radiance), camera.hdr_image.to('cuda'), recipe, rng, use_perceptual=False)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 50 == 0 or step == steps:
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite training loss at step {}'.format(step))
            print('Step {}/{} loss={:.7f} elapsed={:.1f}s peak={:.0f}MiB'.format(
                step, steps, value, time.monotonic() - start, torch.cuda.max_memory_allocated() / 2**20), flush=True)


def save_benchmark(output, scene_name, source, scene, model, dataset, pipe, background):
    result = benchmark(scene.getTestCameras(), model, dataset, pipe, background, final=True)
    write_json(output / 'benchmark.json', dict(scene=scene_name, source=str(source),
               ply_sha256=sha256(source / 'point_cloud.ply'), mapper_sha256=sha256(source / 'tone_mapper.pth'),
               tracks=result))
    print(json.dumps(result), flush=True)


def read_plan(config):
    import yaml
    plan = yaml.safe_load(config.read_text())
    if plan['version'] != 1 or plan['scene'] not in SCENES:
        raise ValueError('Unknown version or scene in the refinement YAML')
    if (plan['scene'] in SYNTHETIC) != (plan['group'] == 'synthetic'):
        raise ValueError('The scene group does not match the scene')
    if not plan['stages'] or any(stage['steps'] <= 0 for stage in plan['stages']):
        raise ValueError('At least one refinement stage with positive steps is required')
    for stage in plan['stages']:
        Recipe(**stage['loss'])
    names = {'xyz','f_dc','f_rest','opacity','scaling','rotation',
             'tone_mapper_r','tone_mapper_g','tone_mapper_b'}
    if set(plan['learning_rates']) != names or min(plan['learning_rates'].values()) <= 0:
        raise ValueError('Set positive learning rates for every optimizer group')
    return plan


def run_plan(args, plan):
    weights = args.load_path.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start_index = args.start_stage - 1
    if not 0 <= start_index < len(plan['stages']):
        raise ValueError('The start stage is outside the configured stages')
    base_config = (args.config.resolve().parent / plan['base_config']).resolve()
    scene, model, dataset, pipe, opt, background = load_scene(
        plan['scene'], weights, output, base_config)
    cameras = scene.getTrainCameras()
    starting_files = {name: sha256(weights / name) for name in ['point_cloud.ply','tone_mapper.pth']}
    if start_index == 0 and starting_files != plan['initialization']['checkpoint_files']:
        print('Initialization differs from the reference hashes. Results may differ.', flush=True)
    for index, stage in enumerate(plan['stages'][start_index:], start=start_index+1):
        torch.manual_seed(stage['seed'])
        if index > start_index + 1:
            model.optimizer = None
            prepare_model(model, weights)
        recipe = Recipe(**stage['loss'])
        print('Stage {} / {}: {}'.format(index,len(plan['stages']),stage['recipe']),flush=True)
        refine(model,scene,opt,pipe,background,cameras,stage['steps'],recipe,dataset.syn,
               stage['seed'],plan['learning_rates'])
        weights = output / 'stage_{:02d}'.format(index) / 'weights'
        weights.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            model.save_ply(str(weights/'point_cloud.ply'))
            model.save_tone_mapper(str(weights/'tone_mapper.pth'))
    write_json(output/'training-plan.json',plan)
    write_json(output/'run.json',dict(init_files=starting_files, start_stage=args.start_stage,
               weights=str(weights), torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    if args.evaluate:
        model.optimizer = None
        prepare_model(model,weights)
        torch.cuda.empty_cache()
        save_benchmark(output,plan['scene'],weights,scene,model,dataset,pipe,background)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--load_path', type=Path, help='Directory with the matching PLY and tone mapper')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--start_stage', type=int, default=1,
                        help='Resume a later stage using the previous stage checkpoint as --load_path')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--validate_config', action='store_true')
    args = parser.parse_args()
    plan = read_plan(args.config)
    if args.validate_config:
        print('{}: {} training stage(s)'.format(plan['scene'],len(plan['stages'])))
        sys.exit(0)
    if args.load_path is None or args.output is None:
        parser.error('provide --load_path and --output to run training')
    args.config = args.config.resolve()
    args.load_path = args.load_path.resolve()
    args.output = args.output.resolve()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.chdir(str(ROOT))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    run_plan(args, plan)
