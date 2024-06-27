import os
import sys

sys.path.append("./outside-code")
import time
import datetime
import random
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from os.path import exists, join
from os import listdir, makedirs

import statistics
from src.model_shape_aware import RetNet
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from src.utils import put_in_world_bvh
from src.utils import get_orient_start
from src.utils import put_in_world2, get_height, get_height_from_skel
from transforms import quat2euler

import scipy
import scipy.ndimage.filters as filters
from Pivots import Pivots


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='Inference BVH')
    parser.add_argument(
        '--config',
        default='./config/inference_bvh_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument(
        '--save_path', default='./', help='path to the configuration file'
    )
    parser.add_argument('--phase', default='test', help='must be train or test')
    parser.add_argument('--load_inp_data', type=dict, default=dict(), help='')
    parser.add_argument('--weights', default='', help='xxx.pt weights for generator')
    parser.add_argument(
        '--device', type=int, default=0, nargs='+', help='only 0 avaliable'
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--k', type=float, default=0.8, help='adjustable k for balacing gate'
    )

    return parser


def get_skel(joints, parents):
    """获得joint相对 parent-joint之间的offset"""
    """
    parents: joint的parent关系 [-1  0  1  2  3  4  0  6  7  8  0 10 11 12  3 14 15 16  3 18 19 20]
    resturn: joint-offset [22, 3]
    """
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:# 非root joint
            # 计算该joint在运动状态，和parent-position的offset
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else: # 如果该joint是root关节
            c_offsets.append(joints[j, :])
    # print(np.stack(c_offsets, axis=0).shape)  # [22,3]
    return np.stack(c_offsets, axis=0)


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    """某种操作，原理未知"""
    return -softmax(-x, **kw)


def process(positions): # joint的全局position
    """Put on Floor"""
    # 定义左脚[8,9两个joint]（左脚跟，左脚尖），右脚[12,13两个joint] 索引
    fid_l, fid_r = np.array([8, 9]), np.array([12, 13])
    # 找出抬角最高的“脚高度”--XYZ顺序-axis--难道是Y朝上?
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    # print(foot_heights.shape) # （743，）
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)  # 1.0126096469286376
    # position-Y 全部减去floor_height，BVH平移到floor上
    positions[:, :, 1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    # 每帧的root关节position
    reference = positions[:, 0]
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    """ Get Foot Contacts """
    # 速度阈值因子： 小于它，则认为 脚部 不移动
    # 高度因子： 小于它，则认为 脚 接地
    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])

    # list[:-1] 代码除最后一个元素外的切片
    #计算再X 轴上的移动距离 [计算连续两帧之间的x轴距离（t帧）-(t-1帧)]
    # list[1,2,3,4]  list[1:] = [2,3,4] list[:-1] = [1,2,3]
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    # print(feet_l_x.shape)  # (742, 2)
    feet_l_h = positions[:-1, fid_l, 1] # (742, 2)
    print(feet_l_h)
    # 判断 left foot是否接地
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(float)

    # 右脚
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (
        ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    ).astype(float)

    """ Get Root Velocity """
    # 差分，计算2帧之间 root-position的偏移量 。 (根关节在每一帧相对于前一帧的移动量)
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

    """ Remove Translation """
    #所有joint 在x的平移 归零
    positions[:, :, 0] = positions[:, :, 0] - positions[:, :1, 0]
    #所有joint 在y的平移 归零
    positions[1:, 1:, 1] = positions[1:, 1:, 1] - (
        positions[1:, :1, 1] - positions[:1, :1, 1]
    )
    #所有joint 在z的平移 归零
    positions[:, :, 2] = positions[:, :, 2] - positions[:, :1, 2]

    """ Get Forward Direction """
    # Original indices + 1 for added reference joint
    # hip_l, hip_r--对应->LeftLeg, rightLeg  
    # sdr_l sdf_r -->对应 --> LeftArm, RightArm  
    sdr_l, sdr_r, hip_l, hip_r = 15, 19, 7, 11
    across1 = positions[:, hip_l] - positions[:, hip_r] # 对应  左腿-右腿 之间向量
    across0 = positions[:, sdr_l] - positions[:, sdr_r] # 对应 左臂右臂之间的向量
    across = across0 + across1
    # 转换成 单位向量
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # 定义滤波器窗口
    direction_filterwidth = 20
    # 获得每帧 body的朝向方向
    forward = np.cross(across, np.array([[0, 1, 0]]))
    
    # 沿着帧的维度，对body朝向 做 （附近20帧）的滤波平滑
    forward = scipy.ndimage.gaussian_filter1d(
        forward, 
        direction_filterwidth, # 窗口大小
        axis=0, # 沿着第0轴
        mode="nearest" # 边界地方 采用“最近邻”处理
    )
    # 对朝向向量做 标准化
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    # 把z轴作为目标方向
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    # 计算朝向&z轴的夹角（四元数）
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    # 反向旋转回去
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 1]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    return positions, rotation


def get_inp_from_bvh(bvh_path):
    """解析input BVH
    return (data_quat, # [frame, joint_num(22), (4)四元数]
        data_seq, # [frame, (74=22*3(postion)+6(速度信息)+2（左右角接地flag）)]
        data_skel) #[frame, 22, 3(parent-offset-position)]
    """
    joints_list = [
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg", 
        "RightLeg",
        "RightFoot",
        "RightToeBase",  
        "LeftShoulder", 
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    # 读取input bvh,然后提取 anim
    anim, name_, _ = BVH.load(bvh_path)  # animation, joint_names, frametime
    # print(anim.shape) # (742, 65)
    bvh_file = open(bvh_path).read().split("JOINT")

    # [' mixamorig:Spine', ' mixamorig:Spine1', ' mixamorig:Spine2', ' mixamorig:Neck'..]
    # mixamo的骨架上，有64个joint,只提取特定的，我们关注的骨架
    # 相当于进行了[骨架的简化]
    bvh_joints = [f.split("\n")[0] for f in bvh_file[1:]]
    to_keep = [0]  # 标记要保留的joint名称
    for jname in joints_list: 
        for k in range(len(bvh_joints)):
            # 如果该joint-name包含指定 字符串
            if jname == bvh_joints[k][-len(jname) :]:
                to_keep.append(k + 1)
                break
    #对简化后的骨架，重新关联“父joint”
    anim.parents = anim.parents[to_keep]
    for i in range(1, len(anim.parents)):
        """If joint not needed, connect to the previous joint"""
        if anim.parents[i] not in to_keep:
            anim.parents[i] = anim.parents[i] - 1
        anim.parents[i] = to_keep.index(anim.parents[i])

    # 新骨架的position,rotation,orient数据准备
    print("new骨架映射：")
    for item_index in range(len(to_keep)):
        print("当前index:{i}, joint-num:{a}, joint-name:{n}".format(
            i=item_index, a=to_keep[item_index], n=name_[to_keep[item_index]]))
    anim.positions = anim.positions[:, to_keep, :]
    anim.rotations.qs = anim.rotations.qs[:, to_keep, :]
    anim.orients.qs = anim.orients.qs[to_keep, :]
    if anim.positions.shape[0] > 1:
        # 计算 joint的全局position
        joints = Animation.positions_global(anim)
        # print(joints.shape) # (742, 22, 3)
        # 拼接：   joints[-1:] ： 最后一帧
        joints = np.concatenate([joints, joints[-1:]], axis=0)
        # joint-position+速度+左右角接地 信息  joint朝向与z轴的夹角
        new_joints, rotation = process(joints)
        new_joints = new_joints[:, 3:]

        rotation = rotation[:-1]
        anim.rotations[:, 0, :] = rotation[:, 0, :] * anim.rotations[:, 0, :]
        angle = anim.rotations.qs
        pose = np.reshape(new_joints[:, :-8], (new_joints.shape[0], -1, 3))
        tgtanim = anim.copy()
        tgtanim.positions[:, 0, :] = new_joints[:, :3]
        poseR = Animation.positions_global(tgtanim)
        data_quat = angle.copy()
        data_seq = new_joints

        anim.rotations.qs[...] = anim.orients.qs[None]
        tjoints = Animation.positions_global(anim)
        # print(anim.parents)
        
        # joint-offset
        anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
        anim.positions[:, 0, :] = new_joints[:, :3]
        data_skel = anim.positions
        print("Load Success.", bvh_path)
        return (data_quat, # [frame, joint_num(22), (4)四元数]
                data_seq, # [frame, (74=22*3(postion)+6(速度信息)+2（左右角接地flag）)]
                data_skel) #[frame, 22, 3(parent-offset-position)]

    print("bvh 提取 Error!")
    return None


def load_from_bvh(
    device, 
    inp_shape_path, # input shape
    tgt_shape_path, # target shape
    stats_path, 
    inp_bvh_path, # input bvh
    tgt_bvh_path # tgt bvh
    ):
    #四元数  帧数据(posiont+速度6+左右脚2)  parent-joint-offset-position
    inpquat, inseq, inpskel = get_inp_from_bvh(inp_bvh_path)
    __, _, tgtskel = get_inp_from_bvh(tgt_bvh_path)
    offset = inseq[:, -8:-4]  # 尾部4列
    print(offset.shape) # [742, 4]
    # joint-position[frame, 22, 3]
    inseq = np.reshape(inseq[:, :-8], [inseq.shape[0], -1, 3])

    T = inpskel.shape[0]
    print("Sequence(帧) length:", T)

    """
    # 解析 input shape文件(FBX-mesh文件的解析结果), 有如下key:
    skinning_weights
    joint_names: 关节列表
    root_orient
    rest_vertices
    rest_faces
    skeleton
    subject
    vertex_part
    rest_body_vertices: body部分的顶点 shape(5452, 3)
    rest_arm_vertices： arm部分 包含的顶点， shape:[5792（顶点数）, 3(position)]
    body_width: body躯干的包围box [68.70365524, 98.38961029, 69.76597977]
    arm_width: arm的包围box [ 38.06436157, 108.1993351 ,  46.32997704]
    full_width: body的包围box [234.76997375, 185.918993  ,  69.76597977]
    joint_shape: size [22, 3（包围box?）]
    crude_verts_ids
    """
    inp_fbx_file = np.load(inp_shape_path)
    inp_full_width = inp_fbx_file['full_width'].astype(np.single)
    inp_joint_shape = inp_fbx_file['joint_shape'].astype(np.single)
    # 转换成比例，joint的长度(绝对值) 转 比例值
    inp_shape = np.divide(inp_joint_shape, inp_full_width[None, :]).reshape(-1)

    # 解析 tgt shape文件
    tgt_fbx_file = np.load(tgt_shape_path)
    tgt_full_width = tgt_fbx_file['full_width'].astype(np.single)
    tgt_joint_shape = tgt_fbx_file['joint_shape'].astype(np.single)
    tgt_shape = np.divide(tgt_joint_shape, tgt_full_width[None, :]).reshape(-1)

    # 方差，均值
    local_mean = np.load(join(stats_path, "mixamo_local_motion_mean.npy"))
    local_std = np.load(join(stats_path, "mixamo_local_motion_std.npy"))
    global_mean = np.load(join(stats_path, "mixamo_global_motion_mean.npy"))
    global_std = np.load(join(stats_path, "mixamo_global_motion_std.npy"))
    quat_mean = np.load(join(stats_path, "mixamo_quat_mean.npy"))
    quat_std = np.load(join(stats_path, "mixamo_quat_std.npy"))
    local_std[local_std == 0] = 1

    inp_skel = inpskel[0, :].reshape([22, 3])
    out_skel = tgtskel[0, :].reshape([22, 3])
    # get skeleton height（单位是米）
    inp_height = get_height(inp_skel) / 100
    tgt_height = get_height(out_skel) / 100

    # 执行z-normal
    inseq = (inseq - local_mean) / local_std  # 对joint-position进行标准化
    inpskel = (inpskel - local_mean) / local_std  # 对源 parent-joint-offset进行标准化
    tgtskel = (tgtskel - local_mean) / local_std  # 对target parent-joint-offset进行标准化
    inpquat = (inpquat - quat_mean) / quat_std # 对源 joint-ratation进行标准化

    inseq = inseq.reshape([inseq.shape[0], -1])
    inpskel = inpskel.reshape([inpskel.shape[0], -1])
    tgtskel = tgtskel.reshape([tgtskel.shape[0], -1])

    inp_seq = np.concatenate((inseq, offset), axis=-1)

    # 为什么再读取一遍呢？
    tgtanim, tgtnames, tgtftime = BVH.load(tgt_bvh_path)
    inpanim, inpnames, inpftime = BVH.load(inp_bvh_path)

    #############获取目标BVH中要 提取的joint-name
    tbvh_file = open(tgt_bvh_path).read().split("JOINT")
    tbvh_joints = [
        f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in tbvh_file[1:]
    ]
    tto_keep = [0]
    # 20个joint
    joints_list = [
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    for jname in joints_list:
        for k in range(len(tbvh_joints)):
            if jname == tbvh_joints[k][-len(jname) :]:
                tto_keep.append(k + 1)
                break

    #############获取源BVH中要 提取的joint-name
    ibvh_file = open(inp_bvh_path).read().split("JOINT")
    ibvh_joints = [
        f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in ibvh_file[1:]
    ]
    ito_keep = [0]
    # 20个joint
    joints_list = [
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    for jname in joints_list:
        for k in range(len(ibvh_joints)):
            if jname == ibvh_joints[k][-len(jname) :]:
                ito_keep.append(k + 1)
                break
    #######################数据加工结束########################################
    # 转移计算设备
    inp_seq = torch.from_numpy(inp_seq.astype(np.single))[None, :].cuda(device)
    inpskel = torch.from_numpy(inpskel.astype(np.single))[None, :].cuda(device)
    tgtskel = torch.from_numpy(tgtskel.astype(np.single))[None, :].cuda(device)
    inp_shape = torch.from_numpy(inp_shape.astype(np.single))[None, :].cuda(device)
    tgt_shape = torch.from_numpy(tgt_shape.astype(np.single))[None, :].cuda(device)
    inpquat = torch.from_numpy(inpquat.astype(np.single))[None, :].cuda(device)
    inp_height_ = torch.zeros((1, 1)).cuda(device)
    tgt_height_ = torch.zeros((1, 1)).cuda(device)
    inp_height_[0, 0] = inp_height
    tgt_height_[0, 0] = tgt_height

    return (
        inp_seq, # source joint-position进行标准化
        inpskel, # source parent-joint-offset进行标准化
        tgtskel, # target parent-joint-offset进行标准化
        inp_shape, # 归一化后的source joint shape
        tgt_shape, # 归一化后的target joint shape
        inpquat, # 归一化后的source joint rotation
    
        inp_height_, # source 骨架高度
        tgt_height_, # target 骨架高度
        ###############stats统计信息
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        # 全局motiona的均值，方差
        global_mean,
        global_std,
        
        #target BVH中 提取的ani
        tgtanim,
        tgtnames,
        tgtftime,
        #source BVH中 提取的ani
        inpanim, # ani
        inpnames, # joint-name
        inpftime, # 帧时间
        
        ito_keep, # source BVH中，要保存的joint-name
        tto_keep, # target BVH中，要保存的joint-name
    )


def get_height(joints):
    return (
        np.sqrt(((joints[5, :] - joints[4, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[4, :] - joints[3, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[3, :] - joints[2, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[2, :] - joints[1, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[1, :] - joints[0, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[6, :] - joints[7, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[7, :] - joints[8, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[8, :] - joints[9, :]) ** 2).sum(axis=-1))
    )


def getmodel(weight_path, arg):
    """加载模型"""
    model = RetNet(**arg.ret_model_args).cuda(arg.device[0])
    model = nn.DataParallel(model, device_ids=arg.device)

    print("load weight from: " + weight_path)
    weights = torch.load(weight_path)
    model.load_state_dict(weights)
    model.eval()
    return model


def inference(
        ret_model:RetNet, # module本身
        parents, # 父 joint 映射
        arg):
    ret_model.eval()  # set 推理模式 
    ret_model.requires_grad_(False) # 关闭梯度更新

    # load data
    (
        inp_seq,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        global_mean,
        global_std,
        tgtanim,
        tgtname,
        tgtftime,
        inpanim,
        inpname,
        inpftime,
        inpjoints,
        tgtjoints,
    ) = load_from_bvh(arg.device[0], **arg.load_inp_data)

    '''
    oursL: local position, should be un-normalized
    oursG: global movement
    quatsB: local rotation
    delta_q: semantics delta
    delta_s: geometry delta
    '''
    # 执行 推理pipeline
    oursL, oursG, quatsB, delta_q, delta_s = ret_model(  
        inp_seq, # source joint-position （z-norml后）
        None,
        inpskel, # source parent-joint-offset（z-norml后）
        tgtskel, # target parent-joint-offset（z-norml后）
        inp_shape, # source shape
        tgt_shape, # target shape
        inpquat, # source rotation

        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        parents,
        arg.k,
    )

    # 从cuda截取到cpu上
    localB = oursL.clone()
    oursL = oursL.cpu().numpy()
    oursG = oursG.cpu().numpy()
    quatsB = quatsB.cpu().numpy()
    delta_q = delta_q.cpu().numpy()
    inp_seq_gpu = inp_seq.clone()
    inp_seq = inp_seq.cpu().numpy()
    tgt_height_gpu = tgt_height.clone()
    inp_height_gpu = inp_height.clone()
    tgt_height = tgt_height.cpu().numpy()
    inp_height = inp_height.cpu().numpy()
    tgtskel = tgtskel.cpu().numpy()

    oursL = oursL.reshape([oursL.shape[0], oursL.shape[1], -1])
    local_mean_rshp = local_mean.reshape((1, 1, -1))
    local_std_rshp = local_std.reshape((1, 1, -1))
    oursL[:, :, :] = oursL[:, :, :] * local_std_rshp + local_mean_rshp
    oursG[:, :, :] = oursG[:, :, :]
    
    # 还原z-normal
    localA = (
        inp_seq_gpu[:, :, :-4].view(localB.shape)
        * torch.from_numpy(local_std)[None, :].cuda()
        + torch.from_numpy(local_mean)[None, :].cuda()
    )
    localB = (
        localB * torch.from_numpy(local_std)[None, :].cuda()
        + torch.from_numpy(local_mean)[None, :].cuda()
    )

    if not exists(arg.save_path):
        makedirs(arg.save_path)

    ours_total = np.concatenate((oursL, oursG), axis=-1)

    """ VIDEO BVH """
    i = 0
    max_steps = tgtskel.shape[1]
    tjoints = np.reshape(tgtskel[i] * local_std_rshp + local_mean_rshp, [max_steps, -1, 3])

    tmp_gt = Animation.positions_global(
        tgtanim
    ) 
    # Given an animation compute the global joint positions at at every frame
    start_rots = get_orient_start(
        tmp_gt,
        tgtjoints[14],  # left shoulder
        tgtjoints[18],  # right shoulder
        tgtjoints[6],  # left upleg
        tgtjoints[10],
    )  # right upleg

    """Exclude angles in exclude_list as they will rotate non-existent children During training."""
    exclude_list = [
        5,
        17,
        21,
        9,
        13,
    ]  # head, left hand, right hand, left toe base, right toe base
    canim_joints = []
    cquat_joints = []
    for l in range(len(tgtjoints)):
        if l not in exclude_list:
            canim_joints.append(tgtjoints[l])
            cquat_joints.append(l)

    outputB_bvh = ours_total[i].copy()

    """Follow the same motion direction as the input and zero speeds
                    that are zero in the input."""
    outputB_bvh[:, -4:] = outputB_bvh[:, -4:] * (
        np.sign(inp_seq[i, :, -4:]) * np.sign(ours_total[i, :, -4:])
    )
    outputB_bvh[:, -3][np.abs(inp_seq[i, :, -3]) <= 1e-2] = 0.0

    outputB_bvh[:, :3] = tgtanim.positions[:1, 0, :].copy()
    wjs, rots = put_in_world_bvh(outputB_bvh.copy(), start_rots)
    tjoints[:, 0, :] = wjs[0, :, 0].copy()

    """ Quaternion results """
    cquat = quatsB[i][:, cquat_joints].copy()

    from_name = arg.load_inp_data["inp_bvh_path"].split('/')[-2]
    to_name = arg.load_inp_data["tgt_bvh_path"].split('/')[-2]
    bvh_name = arg.load_inp_data["inp_bvh_path"].split('/')[-1]

    # 把source,和target BVH重新恢复和保存（用于验证pipeline中的函数正确性）
    BVH.save(join(arg.save_path, from_name + '_inp_' + bvh_name), inpanim, inpname, inpftime)
    BVH.save(join(arg.save_path, to_name + '_gt_' + bvh_name), tgtanim, tgtname, tgtftime)

    """保存 Ours bvh file """
    bvh_path = join(arg.save_path, from_name + '_to_' + to_name + '_' + bvh_name)
    tgtanim.positions[:, tgtjoints] = tjoints
    tgtanim.offsets[tgtjoints[1:]] = tjoints[0, 1:]
    cquat[:, 0:1, :] = (rots * Quaternions(cquat[:, 0:1, :])).qs
    tgtanim.rotations.qs[:, canim_joints] = cquat

    BVH.save(bvh_path, tgtanim, tgtname, tgtftime)


def main(arg):
    retarget_net: RetNet = getmodel(arg.weights, arg)
    parents = np.array(
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
    )
    inference(retarget_net, parents, arg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_parser()
    p = parser.parse_args()

    # p.config对应 一个config文件
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    """
    Namespace(
    config='./config/inference_bvh_cfg.yaml', 
    save_path='./save/inference/', 
    phase='test', 
    load_inp_data={
        # 源shape 文件
        'inp_shape_path': './datasets/mixamo/test_shape/Ortiz.npz', 
        # target(tgt) shape文件
        'tgt_shape_path': './datasets/mixamo/test_shape/Sporty_Granny.npz', 
        # 
        'stats_path': './datasets/mixamo/stats', 
        ####同一段动作，应用到了两个body上
        # 输入bvh(用了anim数据)
        'inp_bvh_path': './datasets/mixamo/demo/Ortiz/Gangnam Style.bvh', # 江南style
        # 目标bvh(只解析了骨架信息，他的BVH是我们要预测的[这里它本身其实是GT，作为label,train用])
        'tgt_bvh_path': './datasets/mixamo/demo/Sporty_Granny/Gangnam Style.bvh'}, 
    weights='./pretrain/shape_aware.pt',  # modele bin路径
    device=[0], 
    num_joint=22, 
    ret_model_args={
        'num_joint': 22, 
        'token_channels': 64, 
        'hidden_channels_p': 256, 
        'embed_channels_p': 128, 
        'kp': 1.0}, 
    k=0.8)
    """
    main(arg)
