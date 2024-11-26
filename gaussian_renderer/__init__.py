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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from pdb import set_trace as stx

min_max_norm = lambda x : (x - x.min()) / (x.max() - x.min())
tonemap = lambda x : torch.log(x * 5000 + 1 ) / torch.log(torch.tensor(5000.0 + 1.0))

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, iteration, scaling_modifier = 1.0, override_color = None, render_mode = 'ldr'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    输入参数:
    (1) viewpoint_camera: 相机的视角和配置, 包含视场角 (FoV)、图像尺寸、变换矩阵
    (2) pc: Gaussian point cloud, 包含点的位置、颜色、不透明度等属性
    (3) pipe: 一些配置和设置, 可能用于控制渲染流程
    (4) bg_color: 背景颜色张量
    (5) scaling_modifier: 缩放修改器, 可能用于调整点的大小或其他属性
    (6) override_color: 可选的覆盖颜色
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个零张量用于存储和返回屏幕空间中点的位置和梯度
    '''
        (1) pc.get_xyz 被用于获取点云的3D坐标, 这些坐标随后用于计算它们在2D图像上的投影位置, 以及在高斯溅射过程中应用的各种变换和计算
        (2) 在PyTorch中, retain_grad() 方法用于指示保存给定张量的梯度。默认情况下, PyTorch只会为具有 requires_grad=True 的叶子节点张量
        （即直接与模型参数相关联的张量）保存梯度。非叶子节点张量（即由其他张量通过运算得到的张量）的梯度在使用后通常会被自动丢弃，以节省内存。如果
        你需要保留这些非叶子节点的梯度用于后续的计算或分析，就需要调用 .retain_grad() 方法
    '''
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 视场角（Field of View，FoV）是指从相机或眼睛位置在水平和垂直方向上可以看到的范围。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 根据相机视角和其它参数初始化高斯光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points    #与三维坐标同样维度的零张量
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 协方差矩阵此处用 scale 和 rotation 表示，Σ = R S S^T R^T
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None: # 不进行颜色覆盖
        if pipe.convert_SHs_python: # pipe 是一个对象或者变量，包含各种配置参数控制不同步骤
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 以相机中心点为起点的方向向量矩阵
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 球谐函数转成 RGB
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = sh2rgb + 0.5
            # colors_precomp = sh2rgb
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # dataloader 的 expstime 得通过外部接口输入到这里
    if colors_precomp is not None and render_mode == 'ldr':
        # numpy 转成 tensor 并且 cuda
        expstime = torch.from_numpy(viewpoint_camera.exps).cuda()
        # 在对数空间上做 embedding
        # colors_precomp 已经是一个cuda了
        # stx()
        # input_embeded_r = torch.log(colors_precomp[:,0]) + torch.log(expstime).float()
        # input_embeded_r = input_embeded_r.unsqueeze(1)
        # input_embeded_g = torch.log(colors_precomp[:,1]) + torch.log(expstime).float()
        # input_embeded_g = input_embeded_g.unsqueeze(1)
        # input_embeded_b = torch.log(colors_precomp[:,2]) + torch.log(expstime).float()
        # input_embeded_b = input_embeded_b.unsqueeze(1)

        # stx()

        # 换到线性空间上做
        # input_embeded_r = colors_precomp[:,0] * expstime
        # input_embeded_r = input_embeded_r.unsqueeze(1).float()
        # input_embeded_g = colors_precomp[:,1] * expstime
        # input_embeded_g = input_embeded_g.unsqueeze(1).float()
        # input_embeded_b = colors_precomp[:,2] * expstime
        # input_embeded_b = input_embeded_b.unsqueeze(1).float()

        # if iteration == 30000:
        #     stx()

        # if iteration == 7000:
        #     stx()

        # 假设已经是对数空间上的了
        # 换到线性空间上做
        '''
            No Clamp
            iter 7000                                                       iter 30000
            input_embeded_r: [-8.5863, 3.1335]                              input_embeded_r: [-15.3584, 0.4061]
            input_embeded_g: [-4.8316, 3.0815]                              input_embeded_g: [-10.4417, 0.2838]
            input_embeded_b: [-5.9671, 3.0009]                              input_embeded_b: [-13.5962, 0.2042]

            
            No clamp + exps_loss
            iter 7000                                                       iter 30000
            input_embeded_r: [-5.4577, 1.7769]                              input_embeded_r: [-11.3768, 0.4061]
            input_embeded_g: [-4.5743, 1.6082]                              input_embeded_g: [-10.4320, 0.7689]
            input_embeded_b: [-6.9358, 4.5107]                              input_embeded_b: [-10.0322, 2.9132]
        '''
        input_embeded_r = colors_precomp[:,0] + torch.log(expstime).float()
        input_embeded_r = input_embeded_r.unsqueeze(1)
        input_embeded_g = colors_precomp[:,1] + torch.log(expstime).float()
        input_embeded_g = input_embeded_g.unsqueeze(1)
        input_embeded_b = colors_precomp[:,2] + torch.log(expstime).float()
        input_embeded_b = input_embeded_b.unsqueeze(1)


        # 不能直接 tone_mapper_r, 需要循环地 forward
        for i in range(len(pc.tone_mapper_r)):
            if i == 0:
                colors_precomp_ldr_r = pc.tone_mapper_r[i](input_embeded_r)
            else:
                colors_precomp_ldr_r = pc.tone_mapper_r[i](colors_precomp_ldr_r)
        for i in range(len(pc.tone_mapper_g)):
            if i == 0:
                colors_precomp_ldr_g = pc.tone_mapper_g[i](input_embeded_g)
            else:
                colors_precomp_ldr_g = pc.tone_mapper_g[i](colors_precomp_ldr_g)
        for i in range(len(pc.tone_mapper_b)):
            if i == 0:
                colors_precomp_ldr_b = pc.tone_mapper_b[i](input_embeded_b)
            else:
                colors_precomp_ldr_b = pc.tone_mapper_b[i](colors_precomp_ldr_b)

        # cuda 转成 numpy
        colors_precomp_ldr = torch.cat([colors_precomp_ldr_r, colors_precomp_ldr_g, colors_precomp_ldr_b], dim=-1)

        # 做成 residual learning
        # stx()
        # colors_precomp_ldr = colors_precomp_ldr + tonemap(torch.sigmoid(torch.exp(colors_precomp)))
        


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 进行 rasterization 渲染
    if render_mode == 'hdr':
        rendered_image_hdr, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            # colors_precomp = colors_precomp,
            colors_precomp = torch.exp(colors_precomp),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image_hdr,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

    if render_mode == 'ldr':
        rendered_image_ldr, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp_ldr,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image_ldr,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
