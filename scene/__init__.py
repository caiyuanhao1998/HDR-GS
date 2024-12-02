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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, GenSpiralCameras
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from pdb import set_trace as stx
import numpy as np
import math

class Scene:

    gaussians : GaussianModel #类型注解

    def __init__(self, args : ModelParams, gaussians : GaussianModel, exp_logger, load_path="", shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}


        # 在此处通过 sceneLoadTypeCallbacks 函数来读取数据，sparse，blender 两个类
        '''
            此处是读数据的源头了,三种数据集类型:
            (1) Colmap 类
            (2) Blender 类
            (3) hdr_real 类
        '''
        print(args.source_path)
        # stx()
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # stx()
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, synthetic=args.syn)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds_exps.npy")):
            print("Found poses_bounds_exps.npy file, assuming HDR real data set!")
            scene_info = sceneLoadTypeCallbacks["hdr_real"](args.source_path, args.eval, exp_logger, args.llffhold, args.factor, args.recenter, args.bd_factor, args.spherify, args.path_zflat, args.max_exp, args.min_exp)
            # llffhold = 0, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, max_exp=1, min_exp=1
        else:
            assert False, "Could not recognize scene type!"
        # stx()

        if not self.loaded_iter:
            # scene_info 是上边这一部分 load 出来的这些数据
            # 使用, 分隔开, 将 src_file 读取出来并写入到 dest_file
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                # .extend() 函数将一个 list 的所有元素都添加到另一个 list 的末尾
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 将 camera_list 转成 json 文件
            for id, cam in enumerate(camlist):
                # stx()
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            '''
                把 scene_info 中的 train_cameras 给加载出来
                返回的就是一个 scene_info 类
            '''
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        #set render video camera trajectory
        if args.render_video:
            self.render_sp_cameras = {}
            render_camera_infos = GenSpiralCameras(scene_info.train_cameras, args=args)
            self.render_sp_cameras[resolution_scale] = cameraList_from_camInfos(render_camera_infos, resolution_scale, args)
        # 加载模型
        # 从指定 iteration 中加载或者从从 camera 数据中加载
        if load_path != "":
            self.gaussians.load_ply(os.path.join(load_path,"point_cloud.ply"))
            self.gaussians.load_tonemapper(os.path.join(load_path,"tone_mapper.pth"))
            print("Loading trained model at {}".format(load_path))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    # 定义存储函数
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_tone_mapper(os.path.join(point_cloud_path, "tone_mapper.pth"))

    # 从 train 或 test 里面取数据的函数
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    #self.train_cameras[1.0][0].R

    def getSpiralCameras(self, scale=1.0):
        return self.render_sp_cameras[scale]