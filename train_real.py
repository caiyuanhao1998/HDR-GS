#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, point_constraint
from torchvision.utils import save_image
from gaussian_renderer import render
import sys
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, gen_log
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, time2file_name, min_max_norm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
from pdb import set_trace as stx
import yaml
import time
import imageio

tonemap = lambda x : torch.log(x * 5000 + 1 ) / torch.log(torch.tensor(5000.0 + 1.0))

# flower
# train_exps = [1/3, 0.1, 1/45]
# test_exps = [1/6, 0.05]

# computer
# train_exps = [1/3, 1/15, 1/60]
# test_exps = [1/8, 1/30]

# box
# train_exps = [2/3, 1/6, 0.05]
# test_exps = [1/3, 0.1]

# luckycat
# train_exps = [2, 0.5, 0.125]
# test_exps = [1, 0.25]


# 训练函数,输入的第一个参数就包含 dataset 
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    exp_logger, log_path = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))
    exp_logger.info("Pipeline parameters: {}".format(vars(pipe)))

    # 实例化模型并加载数据
    # config 的参数要传入 dataset 中
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, exp_logger, load_path = args.load_path)    # 此处读数据，存在 scene 里面

    scene_category = dataset.scene
    if scene_category == "flower":
        train_exps = [1/3, 0.1, 1/45]
        test_exps = [1/6, 0.05]
    elif scene_category == "computer":
        train_exps = [1/3, 1/15, 1/60]
        test_exps = [1/8, 1/30]
    elif scene_category == "box":
        train_exps = [2/3, 1/6, 0.05]
        test_exps = [1/3, 0.1]
    elif scene_category == "luckycat":
        train_exps = [2, 0.5, 0.125]
        test_exps = [1, 0.25]

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    if args.test_only:
        #breakpoint()
        with torch.no_grad():
            exp_logger.info("\n[TESTING ONLY]")
            video_inference(0, scene, render, (pipe, background))
            testing_report(exp_logger, [0], scene, render, (pipe, background), log_path, train_exps, test_exps)
            exit()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera，随机取一个 viewpoint_cam
        # 数据部分实际上由 scene.getTrainCameras() 函数得到
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 此处将数据经过模型，bg参数应该要被去掉，所以真正输入数据的应该是 viewpoint_cam ?
        # 先看一下 viewpoint_cam 包含哪些信息
        # stx()
        # viewpoint_cam 是一个类 scene.cameras.Camera
        # 还要在render里面仔细看这个 Camera 的哪些属性被使用到了，后续好打包
        # view camera 一整个丢进去
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, iteration = iteration, render_mode = 'ldr')
        # stx()

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()


        # if iteration == 0 or iteration > 29800:
        #     align_debug_path = os.path.join(log_path, 'train_set_vis')
        #     os.makedirs(align_debug_path,exist_ok=True)
        #     save_image(min_max_norm(gt_image), os.path.join(align_debug_path,f'gt_{viewpoint_cam.image_name}.png'))
        #     save_image(min_max_norm(image), os.path.join(align_debug_path,f'render_{viewpoint_cam.image_name}.png'))
        # stx()

        Ll1 = l1_loss(image, gt_image)
        # if iteration == 600:
        #     stx()
        # exps_loss = point_constraint(gaussians, args.fixed_value, iteration)
        exps_loss = 0
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + args.exps_loss_weight * exps_loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save

            if (iteration in testing_iterations):
                video_inference(iteration, scene, render, (pipe, background))
            training_report(exp_logger, iteration, Ll1, loss, exps_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), log_path, train_exps, test_exps)
            if (iteration in saving_iterations):
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                exp_logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        # args.model_path = os.path.join("./output/", unique_str[0:10])

        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.method, args.scene, date_time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Logger
    exp_logger = gen_log(args.model_path)

    log_path = args.model_path

    return exp_logger, log_path

def training_report(exp_logger, iteration, Ll1, loss, exps_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, log_path, train_exps, test_exps):
    if exp_logger and (iteration == 0 or (iteration) % 100 == 0):
        # exp_logger.info(f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Exps loss={exps_loss.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")
        exp_logger.info(f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Exps loss={exps_loss}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        testing_report(exp_logger, iteration, scene, renderFunc, renderArgs, log_path, train_exps, test_exps)
        torch.cuda.empty_cache()


def testing_report(exp_logger, iteration, scene : Scene, renderFunc, renderArgs, log_path, train_exps, test_exps):
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
    # stx()
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            # ldr-oe, t1, t3, t5
            num_oe = 0
            psnr_test_oe = 0.0
            ssim_test_oe = 0.0
            lpips_test_oe = 0.0

            # ldr-ne, t2, t4
            num_ne = 0
            psnr_test_ne = 0.0
            ssim_test_ne = 0.0
            lpips_test_ne = 0.0

            # 记录测试的时间
            time_cost = 0.0

            for idx, viewpoint in tqdm(enumerate(config['cameras'])):
                time_start = time.time()
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, render_mode = 'ldr', iteration = iteration)["render"], 0.0, 1.0)
                time_cost += time.time() - time_start
                # image_hdr = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, render_mode = 'hdr', iteration = iteration)["render"], 0.0, 1.0)
                image_hdr_raw = renderFunc(viewpoint, scene.gaussians, *renderArgs, render_mode = 'hdr', iteration = iteration)["render"]
                image_hdr = tonemap(min_max_norm(image_hdr_raw))

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                psnr_cur = psnr(image, gt_image).mean().double()
                ssim_cur = ssim(image, gt_image).mean().double()
                lpips_cur = lpips(image, gt_image, net_type='alex').mean().double()

                if viewpoint.exps in train_exps:
                    psnr_test_oe += psnr_cur
                    ssim_test_oe += ssim_cur
                    lpips_test_oe += lpips_cur
                    num_oe += 1
                elif viewpoint.exps in test_exps:
                    psnr_test_ne += psnr_cur
                    ssim_test_ne += ssim_cur
                    lpips_test_ne += lpips_cur
                    num_ne += 1
                else:
                    stx()
                    raise ValueError("Unknown exposure")

                align_debug_path = os.path.join(log_path, 'test_set_vis', str(iteration))
                align_debug_path_ldr_oe = os.path.join(align_debug_path, 'ldr', 'oe')
                align_debug_path_ldr_ne = os.path.join(align_debug_path, 'ldr', 'ne')
                align_debug_path_hdr = os.path.join(align_debug_path, 'hdr')
                os.makedirs(align_debug_path,exist_ok=True)
                os.makedirs(align_debug_path_ldr_oe,exist_ok=True)
                os.makedirs(align_debug_path_ldr_ne,exist_ok=True)
                os.makedirs(align_debug_path_hdr,exist_ok=True)
                # stx()
                # iio.imwrite(os.path.join(align_debug_path,'gt_{}.png'.format(viewpoint_cam.image_name)), (cast_to_image(gt_image[0])*255).astype(np.uint8))
                # iio.imwrite(os.path.join(align_debug_path,'render_{}.png'.format(viewpoint_cam.image_name)), (cast_to_image(image[0])*255).astype(np.uint8))
                if viewpoint.exps in train_exps:
                    save_image(min_max_norm(gt_image), os.path.join(align_debug_path_ldr_oe, 'gt_{}_ldr.png'.format(viewpoint.image_name)))
                    save_image(min_max_norm(image), os.path.join(align_debug_path_ldr_oe, 'render_{}_ldr.png'.format(viewpoint.image_name)))
                if viewpoint.exps in test_exps:
                    save_image(min_max_norm(gt_image), os.path.join(align_debug_path_ldr_ne, 'gt_{}_ldr.png'.format(viewpoint.image_name)))
                    save_image(min_max_norm(image), os.path.join(align_debug_path_ldr_ne, 'render_{}_ldr.png'.format(viewpoint.image_name)))
                save_image(min_max_norm(image_hdr), os.path.join(align_debug_path_hdr, 'render_{}_hdr.png'.format(viewpoint.image_name)))

                imageio.imwrite(os.path.join(align_debug_path_hdr, 'render_{}_hdr.exr'.format(viewpoint.image_name)), image_hdr_raw.permute(1, 2, 0).cpu().numpy())
            
            psnr_test_oe /= num_oe
            ssim_test_oe /= num_oe
            lpips_test_oe /= num_oe

            psnr_test_ne /= num_ne
            ssim_test_ne /= num_ne
            lpips_test_ne /= num_ne

            exp_logger.info("[ITER {}] LDR-OE Evaluating: Number {}, PSNR {}, SSIM {}, LPIPS {}".format(iteration, num_oe, psnr_test_oe, ssim_test_oe, lpips_test_oe))
            exp_logger.info("[ITER {}] LDR-NE Evaluating: Number {}, PSNR {}, SSIM {}, LPIPS {}".format(iteration, num_ne, psnr_test_ne, ssim_test_ne, lpips_test_ne))
            exp_logger.info(f"Time cost: {time_cost} s, Test speed: {len(config['cameras']) / time_cost} fps")

                # if iteration == 30000:
                #     stx()

def video_inference(iteration, scene : Scene, renderFunc, renderArgs):
    save_folder = os.path.join(scene.model_path,"videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs 
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras' : scene.getSpiralCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, iteration = iteration, *renderArgs)
            rgb = render_out["render"]
            image = torch.clamp(rgb, 0.0, 1.0) 
            image = image.detach().cpu().permute(1,2,0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)    
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/lego.yaml', help='Path to the configuration file')
    parser.add_argument("--load_path", type=str, default="", help="link to the pretrained model file")
    parser.add_argument("--test_only", action='store_true', default=False)    
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="7", help="gpu to use")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 使用配置文件中的参数来设置OptimizationParams对象的属性
    for key, value in config.items():
        setattr(args, key, value)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
