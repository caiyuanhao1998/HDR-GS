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
import imageio
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from pdb import set_trace as stx
import copy
from utils.video_utils import get_spiral_render_path

tonemap = lambda x : np.log(x * 5000.0 + 1.0) / np.log(5000.0 + 1.0)


# 从命名元组的类中继承
'''
    CameraInfo: 存储相机的相关信息
'''
class CameraInfo(NamedTuple):
    uid: int            # 相机id
    R: np.array         # 旋转矩阵
    T: np.array         # 平移向量
    FovY: np.array      # 垂直视场角
    FovX: np.array      # 水平视场角
    image: np.array     # 图像对象
    image_path: str     # 图像路径
    image_name: str     # 图像文件名
    width: int          # 图像宽度
    height: int         # 图像高度


class CameraInfo_hdr(NamedTuple):
    uid: int            # 相机id
    R: np.array         # 旋转矩阵
    T: np.array         # 平移向量
    FovY: np.array      # 垂直视场角
    FovX: np.array      # 水平视场角
    image: np.array     # 图像对象
    image_path: str     # 图像路径
    image_name: str     # 图像文件名
    width: int          # 图像宽度
    height: int         # 图像高度
    exps: np.array      # 曝光值


class CameraInfo_hdr_syn(NamedTuple):
    uid: int                # 相机id
    R: np.array             # 旋转矩阵
    T: np.array             # 平移向量
    FovY: np.array          # 垂直视场角
    FovX: np.array          # 水平视场角
    image: np.array         # 图像对象
    image_path: str         # 图像路径
    image_name: str         # 图像文件名
    image_hdr: np.array     # 图像对象
    image_hdr_path: str     # 图像路径
    image_hdr_name: str     # 图像文件名
    width: int              # 图像宽度
    height: int             # 图像高度
    exps: np.array          # 曝光值


'''
    SceneInfo: 存储场景的相关信息
'''
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud    # 点云对象
    train_cameras: list             # 训练相机列表
    test_cameras: list              # 测试相机列表
    nerf_normalization: dict        # 归一化 nerf 信息字典
    ply_path: str                   # 点云文件路径


'''
    计算 NeRF++ 模型的归一化参数, 为何有一个 NeRF++ ?
    计算相机的平移和半径
'''
def getNerfppNorm(cam_info):

    # 计算相机中心点和对角线长度
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)                            # 将cam_centers数组展平为一维数组
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)    # 计算cam_centers数组的平均中心点坐标
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # 计算每个点距离中心点的距离
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    # 将相机中心在世界坐标系中的位置添加到cam_centers列表中
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def average_every_five(lst):
    # 计算列表长度
    n = len(lst)
    # 遍历列表，每隔5个元素计算均值并替换
    for i in range(0, n, 5):
        # 计算当前区间的均值
        avg = sum(lst[i:i+5]) / 5.0
        # 替换当前区间的值为均值
        lst[i:i+5] = [avg] * min(5, n-i)
    return lst

# 内外参矩阵每隔五个取一个均值
def average_camera_pose(cam_extrinsics, cam_intrinsics):
    # 遍历两个参数字典
    qvec_list = []
    tvec_list = []
    params_list = []
    for idx, key in enumerate(cam_extrinsics):

        extr_curr = cam_extrinsics[key]
        intr_curr = cam_intrinsics[extr_curr.camera_id]

        qvec_list.append(extr_curr.qvec)
        tvec_list.append(extr_curr.tvec)
        params_list.append(intr_curr.params)
    
    # 分成长度为 5 的 array
    qvec_list = average_every_five(qvec_list)
    tvec_list = average_every_five(tvec_list)
    params_list = average_every_five(params_list)
    
    return qvec_list, tvec_list, params_list



def copy_1_to_5_pose(cam_extrinsics, cam_intrinsics):
    # 构造两个新的字典，关键字为从 1 到 175
    # cam_extrinsics_new = {}
    # cam_intrinsics_new = {}

    # for idx, key in enumerate(cam_extrinsics):
    #     for i in range(5):
    #         cam_extrinsics_new[5*(key-1)+i+1] = cam_extrinsics[key]
    #         cam_intrinsics_new[5*(key-1)+i+1] = cam_intrinsics[key]

    #         # 0000_0 -> 0000_i
    #         cam_extrinsics_new[5*(key-1)+i+1].name = cam_extrinsics[key].name.replace('_0', f'_{i}')

    #         # id: key -> 5*(key-1)+i+1
    #         cam_extrinsics_new[5*(key-1)+i+1].id = 5*(key-1)+i+1
    #         cam_intrinsics_new[5*(key-1)+i+1].id = 5*(key-1)+i+1
    
    name_list = []
    id_list = []

    for idx, key in enumerate(cam_extrinsics):
        for i in range(5):
            # stx()
            name_list.append(cam_extrinsics[key].name.replace('_1', f'_{i}'))
            id_list.append(5*(key-1)+i+1)
    
    name_list.sort()

    return name_list, id_list

'''
    这段代码的作用是读取Colmap相机的参数, 并
    将其保存在cam_infos列表中。函数readColmapCameras
    的输入参数包括cam_extrinsics(相机外参)、
    cam_intrinsics (相机内参) 和 images_folder (图像文件夹路径)
'''
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, basedir):

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_exps.npy'))  # [175, 18]
    exps = poses_arr[:, -1:]

    cam_infos = []
    # qvec_list, tvec_list, params_list = average_camera_pose(cam_extrinsics, cam_intrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        '''
            cam_extrinsics 是一个 dict, key是 1 - 175
            value 是 image 类: cam_extrinsics[1]
            Image(id=1, qvec=array([ 0.99402673,  0.09105864, -0.0597705 , -0.00683187]), tvec=array([4.27259949, 2.31555504, 0.99870034]), camera_id=1, name='000_0.jpg', xys=array([[3174.69189453,  173.3223114 ],
                    [1263.27929688,  379.03912354],
                    [ 987.5970459 ,  400.98306274],
                    ...,
                    [ 867.94049072, 1638.10229492],
                    [1386.98913574, 1109.94641113],
                    [1130.25292969, 1624.7767334 ]]), point3D_ids=array([-1, -1, -1, ..., -1, -1, -1]))
            cam_intrinsics 也是一个dict, key也是一样
            value是 Camera 类: cam_intrinsics[1]
        '''

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        exps_curr = exps[idx]

        # stx()

        # 每5个pose取一次平均值

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo_hdr(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, exps=exps_curr)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    # stx()
    return cam_infos


def readColmapCameras_average(cam_extrinsics, cam_intrinsics, images_folder, basedir):
    qvec_list, tvec_list, params_list = average_camera_pose(cam_extrinsics, cam_intrinsics)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_exps.npy'))  # [175, 18]
    exps = poses_arr[:, -1:] # [175, 1]

    cam_infos = []
    # qvec_list, tvec_list, params_list = average_camera_pose(cam_extrinsics, cam_intrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        '''
            cam_extrinsics 是一个 dict, key是 1 - 175
            value 是 image 类: cam_extrinsics[1]
            Image(id=1, qvec=array([ 0.99402673,  0.09105864, -0.0597705 , -0.00683187]), tvec=array([4.27259949, 2.31555504, 0.99870034]), camera_id=1, name='000_0.jpg', xys=array([[3174.69189453,  173.3223114 ],
                    [1263.27929688,  379.03912354],
                    [ 987.5970459 ,  400.98306274],
                    ...,
                    [ 867.94049072, 1638.10229492],
                    [1386.98913574, 1109.94641113],
                    [1130.25292969, 1624.7767334 ]]), point3D_ids=array([-1, -1, -1, ..., -1, -1, -1]))
            cam_intrinsics 也是一个dict, key也是一样
            value是 Camera 类: cam_intrinsics[1]
        '''

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        exps_curr = exps[idx]   # idx 从0开始, key 从 1 开始

        # stx()

        # stx()

        # 每5个pose取一次平均值

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(qvec_list[idx]))
        T = np.array(tvec_list[idx])

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = params_list[idx][0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = params_list[idx][0]
            focal_length_y = params_list[idx][1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo_hdr(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, exps=exps_curr)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_single_exps(cam_extrinsics, cam_intrinsics, images_folder, basedir):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_exps.npy'))  # [175, 18]
    exps = poses_arr[:, -1:]
    # stx()
    cam_infos = []
    # qvec_list, tvec_list, params_list = average_camera_pose(cam_extrinsics, cam_intrinsics)
    for idx, key in enumerate(cam_extrinsics):
        for i in range(5):
            '''
                cam_extrinsics 是一个 dict, key是 1 - 35, 但是图片的序号打乱了, 只能每个序号当场复制五遍
                value 是 image 类: cam_extrinsics[1]
                Image(id=1, qvec=array([ 0.99402673,  0.09105864, -0.0597705 , -0.00683187]), tvec=array([4.27259949, 2.31555504, 0.99870034]), camera_id=1, name='000_0.jpg', xys=array([[3174.69189453,  173.3223114 ],
                        [1263.27929688,  379.03912354],
                        [ 987.5970459 ,  400.98306274],
                        ...,
                        [ 867.94049072, 1638.10229492],
                        [1386.98913574, 1109.94641113],
                        [1130.25292969, 1624.7767334 ]]), point3D_ids=array([-1, -1, -1, ..., -1, -1, -1]))
                cam_intrinsics 也是一个dict, key也是一样
                value是 Camera 类: cam_intrinsics[1]
            '''
            id_curr = 5*(key-1)+i+1

            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]  # camera_id 在后续没有再用到
            exps_curr = exps[id_curr-1]       # 5个一循环


            exps_id = extr.name.split("_")[1].split(".")[0]

            image_name_curr = os.path.basename(extr.name).replace(f'_{exps_id}', f'_{i}')

            # stx()

            # 每5个pose取一次平均值

            height = intr.height
            width = intr.width

            uid = id_curr
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(images_folder, os.path.basename(image_name_curr))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            cam_info = CameraInfo_hdr(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, exps=exps_curr)
            cam_infos.append(cam_info)
    # stx()
    return cam_infos


def readColmapCameras_single_exps_syn(cam_extrinsics, cam_intrinsics, images_folder, basedir):

    # stx()
    
    exps = np.array([0.125, 0.5, 2.0, 8.0, 32.0])
    exps = np.expand_dims(exps, axis=1)

    # stx()

    cam_infos = []
    # qvec_list, tvec_list, params_list = average_camera_pose(cam_extrinsics, cam_intrinsics)
    for idx, key in enumerate(cam_extrinsics):
        for i in range(5):
            '''
                cam_extrinsics 是一个 dict, key是 1 - 35, 但是图片的序号打乱了, 只能每个序号当场复制五遍
                value 是 image 类: cam_extrinsics[1]
                Image(id=1, qvec=array([ 0.99402673,  0.09105864, -0.0597705 , -0.00683187]), tvec=array([4.27259949, 2.31555504, 0.99870034]), camera_id=1, name='000_0.jpg', xys=array([[3174.69189453,  173.3223114 ],
                        [1263.27929688,  379.03912354],
                        [ 987.5970459 ,  400.98306274],
                        ...,
                        [ 867.94049072, 1638.10229492],
                        [1386.98913574, 1109.94641113],
                        [1130.25292969, 1624.7767334 ]]), point3D_ids=array([-1, -1, -1, ..., -1, -1, -1]))
                cam_intrinsics 也是一个dict, key也是一样
                value是 Camera 类: cam_intrinsics[1]
            '''
            id_curr = 5*(key-1)+i+1

            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]  # camera_id 在后续没有再用到
            exps_curr = exps[(id_curr-1)%5]       # 5个一循环

            # stx()

            exps_id = extr.name.split("_")[1].split(".")[0]

            image_name_curr = os.path.basename(extr.name).replace(f'_{exps_id}', f'_{i}')

            exr_name = os.path.basename(extr.name).split("_")[0] + ".exr"

            # stx()

            # 每5个pose取一次平均值

            height = intr.height
            width = intr.width

            uid = id_curr
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(images_folder, os.path.basename(image_name_curr))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            exr_folder = os.path.join(basedir, "exr")
            image_hdr_path = os.path.join(exr_folder, os.path.basename(exr_name))
            image_hdr_name = os.path.basename(image_hdr_path).split(".")[0]
            # stx()
            image_hdr_np = np.array(imageio.imread(image_hdr_path)).astype(np.float32)
            if image_hdr_np.shape[2] == 4:
                image_hdr_np = image_hdr_np[:, :, :3]
            image_hdr_np /= np.max(image_hdr_np)
            image_hdr_np = tonemap(image_hdr_np)
            image_hdr = Image.fromarray((image_hdr_np * 255).astype(np.uint8))

            # stx()

            # 在此处把 exr 文件读取进来然后传入 CameraInfo_hdr_synthetic 类中

            cam_info = CameraInfo_hdr_syn(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, image_hdr = image_hdr, image_hdr_path=image_hdr_path,
                                 image_hdr_name=image_hdr_name, width=width, height=height, exps=exps_curr)
            cam_infos.append(cam_info)
    # stx()
    return cam_infos


'''
    读取给定路径下的PLY文件, 并从中
    提取顶点数据，包括位置坐标、颜色和法向量。
    颜色将被转换为一个归一化的范围(0-1)。
    最后，代码使用提取到的数据 (位置、颜色和法向量) 创建一个BasicPointCloud对象并返回
'''
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # interval = 4
    # stx()
    # return BasicPointCloud(points=positions[::interval], colors=colors[::interval], normals=normals[::interval])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

'''
    将包含位置坐标和颜色信息的点云数据保存为PLY文件格式。
'''
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


'''
    读取 colmap 类型数据的函数返回的是相机信息列表,
    里面是 CameraInfo 类 - 元组
    返回一个 SceneInfo 类
'''
def readColmapSceneInfo(path, images, eval, llffhold=8, synthetic=False):
    try:
        cameras_extrinsic_file = os.path.join(path, f"sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        # stx()
    except:
        cameras_extrinsic_file = os.path.join(path, f"sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), basedir=path)
    # cam_infos_unsorted = readColmapCameras_average(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), basedir=path)
    if not synthetic:
        cam_infos_unsorted = readColmapCameras_single_exps(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), basedir=path)
    else:
        cam_infos_unsorted = readColmapCameras_single_exps_syn(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), basedir=path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # colmap 根据 测试的list来每8个抽一个
    if eval:
        i_train = []
        i_test = []
        exp_num = 5
        # [0, 2, 4, 10, 12, 14, 20, 22, 24, 30, 32, 34, 40, 42, 44, 50, 52, 54, 60, 62, 64, 70, 72, 74, 80, 82, 84, 90, 92, 94, 100, 102, 104, 110, 112, 114, 120, 122, 124, 130, 132, 134, 140, 142, 144, 150, 152, 154, 160, 162, 164, 170, 172, 174]
        for i in range(len(cam_infos) // (exp_num*2) + 1):         # 为什么是间隔为 2 的取呢？ 训练集和测试集
            step = i*exp_num*2
            i_train.append(step+0)
            i_train.append(step+2)
            i_train.append(step+4)
        
        #[5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 35, 36, 37, 38, 39, 45, 46, 47, 48, 49, 55, 56, 57, 58, 59, 65, 66, 67, 68, 69, 75, 76, 77, 78, 79, 85, 86, 87, 88, 89, 95, 96, 97, 98, 99, 105, 106, 107, 108, 109, 115, 116, 117, 118, 119, 125, 126, 127, 128, 129, 135, 136, 137, 138, 139, 145, 146, 147, 148, 149, 155, 156, 157, 158, 159, 165, 166, 167, 168, 169]
        for i in range(len(cam_infos) // (exp_num*2)):
            step = (2*i+1)*exp_num
            i_test.append(step+0)
            i_test.append(step+1)
            i_test.append(step+2)
            i_test.append(step+3)
            i_test.append(step+4)
        
        # stx()

        i_train = np.sort(np.array(i_train).reshape([-1]))
        # i_test = np.array([i for i in np.arange(int(len(cam_infos))) if (i not in i_train)])
        i_test = np.sort(np.array(i_test).reshape([-1]))

        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

        print("Traininig set:", i_train)
        print("Testing set:", i_test)

        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"sparse/0/points3D.ply")
    # ply_path = os.path.join(path, 'points3d_colmap_all.ply')
    bin_path = os.path.join(path, f"sparse/0/points3D.bin")
    txt_path = os.path.join(path, f"sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # stx()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# 在下边的 readNerfSyntheticInfo 函数中被调用
# 用来读取 blender 数据集 - 这边就是标准的 opencv 不需要修改
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info





# ----------------------------------------- hdr 数据处理篇 -----------------------------------------

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'input_images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'input_images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'input_images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'input_images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'input_images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]                     # [3, 1] - hwf 不变
    center = poses[:, :3, 3].mean(0)            # [175, 3] -> [3, 1] - 中心位置
    vec2 = normalize(poses[:, :3, 2].sum(0))    # [175, 3] -> [3, 1] - rotation 的第二列
    up = poses[:, :3, 1].sum(0)                 # [175, 3] -> [3, 1] - rotation 的第一列
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array(
            [np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))

    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_exps.npy'))  # [175, 18]
    # 按顺序 concate 起来
    '''
        poses: [175, 15] -> [175, 3, 5] -> [3, 5, 175], 175 是数目, [3, 5]的 poses 包含旋转矩阵(3x3), 平移向量(3x1), shape(2x1) + focal length(1x1) 即 hwf
        bds: [175, 2] -> [2, 175]
        exps: [175, 1] -> [1, 175]
    '''
    poses = poses_arr[:, :-3].reshape([-1, 3, 5]).transpose([1,2,0])    
    bds = poses_arr[:, -3:-1].transpose([1,0])
    exps = poses_arr[:, -1:].transpose([1,0])

    # stx()
    
    img0 = [os.path.join(basedir, 'input_images', f) for f in sorted(os.listdir(os.path.join(basedir, 'input_images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])      # 对图像进行缩小处理
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'input_images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # stx()

    '''
        sh: [2136, 3216, 3]
        sh[:2]: [2136, 3216] - spatial size, shape 是 (2,) 然后被 reshape 到 (2, 1)
        此时 poses 的 shape 是 [3, 5, 175], 仍旧没变
    '''
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])  # 为何要 reshape 一下呢？
    poses[2, 4, :] = poses[2, 4, :] * 1./factor         # 这一列存的是什么信息呢？ —— 调整深度或距离的比例
    # stx()
    
    if not load_imgs:
        return poses, bds, exps
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]    # 0 - 255 -> 0 - 1
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])

    return poses, bds, exps, imgs 


def load_real_llff_data(basedir, exp_logger, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, max_exp=1, min_exp=1):
    poses, bds, exp, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    '''
        poses: [3, 5, 175]
        bds: [2, 175]
        exp: [1, 175]
        imgs: [2136, 3216, 3, 175]
    '''

    exp_logger.info('Loaded %s %s %s', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    # 为何要将最后一个维度移动大第一个维度呢？数量移动到最前面来
    # stx()
    '''
        对 poses 的矫正：
        poses[:, 1:2, :] --- [3, 1, 175]
        -poses[:, 0:1, :] --- [3, 1, 175]
        poses[:, 2:, :] --- [3, 3, 175]
        把第二列和第一列对调, 第一列取相反数, 为什么这么操作呀？ -- 坐标轴调整, 需要扭一下
    '''
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)   # llff --> OpenGL
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    exp = np.moveaxis(exp, -1, 0).astype(np.float32) # [1, N]
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    # translation 和 bounds 都要乘以 sc
    poses[:,:3,3] *= sc
    bds *= sc
    
    # 将相机位姿重新定位到新的中心位置
    # 所有的相机位姿减去中心位置？
    if recenter:
        poses = recenter_poses(poses)
        
    # 为啥要球化呀？好像一般做成 false
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)              # 计算得到平均相机位姿, c2w: [3, 5]
        exp_logger.info('recentered %s', c2w.shape)      # 
        exp_logger.info(c2w[:3,:4])

        ## Get spiral  -- 生成螺旋路径 -- demo 用的
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 30, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses and exposures for spiral path
        # 此时的 render_poses 还是 [120, 3, 5]
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        render_exps = np.linspace(np.log2(min_exp), np.log2(max_exp), N_views//2) # the exposure denotes exposure time
        render_exps = 2 ** render_exps
        render_exps = np.concatenate([render_exps, render_exps[::-1]])

        
    render_poses = np.array(render_poses).astype(np.float32)
    render_exps = np.reshape(render_exps, [-1, 1]).astype(np.float32)

    # stx()

    c2w = poses_avg(poses)
    exp_logger.info('Data:')
    # exp_logger.info(poses.shape, images.shape, bds.shape)
    exp_logger.info('poses.shape: %s, images.shape: %s, bds.shape: %s', str(poses.shape), str(images.shape), str(bds.shape))
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    exp_logger.info('HOLDOUT view is %s', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    # stx()

    return images, poses, bds, exp, render_poses, render_exps, i_test


# 直接把 train 和 test 放在一块读取
def readCamerasFromTransforms_hdr_real(basedir, exp_logger, llffhold = 0, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, max_exp=1, min_exp=1):
    train_cam_infos = []
    test_cam_infos = []

    images, poses, bds, exps_source, render_poses, render_exps, i_test = load_real_llff_data(basedir=basedir, exp_logger=exp_logger, factor=factor,
                                                                                                 recenter=recenter, bd_factor=bd_factor, spherify=spherify,
                                                                                                 max_exp=max_exp, min_exp=min_exp)
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    exps = exps_source
    # 对读取的数据进行前处理
    exp_logger.info('Loaded real llff: %s %s %s %s', images.shape, render_poses.shape, hwf, basedir)
    if not isinstance(i_test, list):
        i_test = [i_test]
    
    # 如果 args.llffhold > 0 即手动设置了测试试图，则按照间隔 args.llffhold 选择测试视图
    if llffhold > 0:
        exp_logger.info('Auto LLFF holdout, %s', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]
    
    # randomly select an exposure from {t_1, t_3, t_5} for each input view
    elif llffhold == 0: 
        exp_logger.info('Random select images for training.')
        np.random.seed(100)
        i_train = []
        exp_num = 5
        for i in range(images.shape[0] // (exp_num*2) + 1):         # 为什么是间隔为 2 的取呢？
            step = i*exp_num*2
            i_train.append(np.random.choice([0+step, 2+step, 4+step], 1, replace=False))    # 从当前 step 的 0, 2, 4 中选择一个作为训练试图
        i_train = np.sort(np.array(i_train).reshape([-1]))
        i_test = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_train)])

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

    exp_logger.info('TRAIN views are: %s', i_train)
    exp_logger.info('TEST views are: %s', i_test)
    exp_logger.info('VAL views are: %s', i_val)
    
    # 按检索号提取 train_camera_info
    for idx in i_train:
        c2w = poses[idx]  # [3, 4], 正常的c2w是 4x4 的矩阵
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        # stx()
        c2w[:3, 1:3] *= -1 # blender -> colmap
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        FovX = np.float64(focal2fov(hwf[-1], hwf[0]))
        FovY = np.float64(focal2fov(hwf[-1], hwf[1]))
        image = Image.fromarray((images[idx]*255).astype(np.uint8), "RGB")
        exps_cur = exps[idx].astype(np.float64)
        train_cam_infos.append(CameraInfo_hdr(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=None, image_name=str(idx), width=int(hwf[0]), height=int(hwf[1]), exps=exps_cur))
    
    # 按检索号提取 test_camera_info
    for idx in i_test:
        c2w = poses[idx]
        c2w = np.vstack((c2w, np.array([0, 0, 0, 1])))
        c2w[:3, 1:3] *= -1 # blender -> colmap
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        FovX = np.float64(focal2fov(hwf[-1], hwf[0]))
        FovY = np.float64(focal2fov(hwf[-1], hwf[1]))
        image = Image.fromarray((images[idx]*255).astype(np.uint8), "RGB")
        exps_cur = exps[idx].astype(np.float64)
        test_cam_infos.append(CameraInfo_hdr(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=None, image_name=str(idx), width=int(hwf[0]), height=int(hwf[1]), exps=exps_cur))
            
    return train_cam_infos, test_cam_infos


def readNerfInfo_hdr_real(path, eval, exp_logger, llffhold = 0, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, max_exp=1, min_exp=1):
    print("Reading Training and Testing Transforms")
    train_cam_infos, test_cam_infos, bds = readCamerasFromTransforms_hdr_real(basedir = path, exp_logger=exp_logger, llffhold = llffhold, factor=factor, recenter=recenter, bd_factor=bd_factor, spherify=spherify, path_zflat=path_zflat, max_exp=max_exp, min_exp=min_exp)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, "points3d_colmap.ply")
    # ply_path = "/home/ycai51/hdr_gaussian_mlp/output/flower/2024_04_07_17_53_30/point_cloud/iteration_30000/point_cloud.ply"
    ply_path = os.path.join(path, "points3d_colmap_all.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        exp_logger.info(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
        exp_logger.info("Loading point cloud from: %s", ply_path)
        # stx()
    except:
        pcd = None
    # stx()

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def TransformPosesToCamera(render_poses,sample_cam,args):
    # random focal
    cam_infos = []
    sample_cam_temp = copy.deepcopy(sample_cam)
    render_exps_first = np.linspace(args.video_render_exps[0],args.video_render_exps[1],render_poses.shape[0]//2)
    render_exps_sec = np.linspace(args.video_render_exps[1],args.video_render_exps[0],render_poses.shape[0]//2)
    render_exps = np.concatenate([render_exps_first,render_exps_sec],axis = 0)
    for idx in range(render_poses.shape[0]):
        R = np.array(render_poses[idx][:3, :3], np.float32)#.transpose(1, 0)
        T = np.array(render_poses[idx][:3, 3], np.float32)
        if args.syn:
            cam_infos.append(CameraInfo_hdr_syn(uid=idx, R=R, T=T, FovY=sample_cam_temp.FovY, FovX=sample_cam_temp.FovX, image=sample_cam_temp.image,
                                image_path=sample_cam_temp.image_path, image_name=sample_cam_temp.image_name, image_hdr = sample_cam_temp.image_hdr, image_hdr_path=sample_cam_temp.image_hdr_path,
                                 image_hdr_name=sample_cam_temp.image_hdr_name, width=sample_cam_temp.width, height=sample_cam_temp.height, exps=np.asarray([render_exps[idx]])))
        else:
            cam_infos.append(CameraInfo_hdr(uid=idx, R=R, T=T, FovY=sample_cam_temp.FovY, FovX=sample_cam_temp.FovX, image=sample_cam_temp.image,
                                    image_path=None, image_name=sample_cam_temp.image_name, width=sample_cam_temp.width, height=sample_cam_temp.height, exps=np.asarray([render_exps[idx]])))        
    return cam_infos

def GenSpiralCameras(train_cameras, args):
    all_c2ws = []
    for cam in train_cameras:
        exts = np.concatenate((cam.R.transpose(1, 0),np.expand_dims(cam.T, axis=1)),axis = 1)
        temp = np.expand_dims(np.array((0,0,0,1)),axis=0)
        exts_homo = np.concatenate((exts,temp),axis = 0)
        focal = fov2focal(cam.FovX,cam.width)
        all_c2ws.append(np.linalg.inv(exts_homo))
    all_c2ws = np.stack(all_c2ws)
    render_poses = get_spiral_render_path(all_c2ws,focal=focal)
    render_camera_info = TransformPosesToCamera(render_poses,train_cameras[0],args)
    return render_camera_info
    
# 三种读取函数
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "hdr_real": readNerfInfo_hdr_real
}