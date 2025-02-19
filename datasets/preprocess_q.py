import os
import shutil
import sys

sys.path.append("../outside-code")
sys.path.append("./")
os.chdir(sys.path[0])
import BVH as BVH
import numpy as np
import scipy.ndimage.filters as filters
import Animation
from Quaternions import Quaternions
from Pivots import Pivots
from os import listdir, makedirs, system
from os.path import exists
from inference_bvh import process

def get_skel(joints, parents):
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else:
            c_offsets.append(joints[j, :])
    return np.stack(c_offsets, axis=0)


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


""" This script generated the local/global motion decoupled data and
    stores it for later training """

home_dir = "/home/wxk/project_linux/dongbu/R2ET"
#待处理的BVH目录
data_paths = ["{a}/datasets/mixamo/train_char/".format(a=home_dir)]
# 生成的结果目录
save_path = "{a}/datasets/mixamo/train_q/".format(a=home_dir)

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

for data_path in data_paths:
    print("Processing " + data_path)
    print(listdir(data_path))
    folders = sorted(
        [
            f
            for f in listdir(data_path)
            if not f.startswith(".") and not f.endswith("py") and not f.endswith("npz")
        ]
    )
    # folders = ['']  # for one folder
    for folder in folders:
        # os.mkdir(save_path+folder)
        files = sorted([f for f in listdir(data_path + folder) if f.endswith(".bvh")])
        for cfile in files:
            #######开始逐个处理BVH文件
            in_bvh_file = data_path + folder + "/" + cfile
            out_npy_file_prefix = save_path + folder + "/" + cfile[:-4]
            out_npy_quat_file = out_npy_file_prefix + "_quat.npy"
            # [frame, joint, posiont(global)]
            out_npy_seq_file = out_npy_file_prefix + "_seq.npy"
            # joint-offset-position
            out_npy_skel_file = out_npy_file_prefix + "_skel.npy"
            # print(in_bvh_file)
            # print(out_npy_quat_file)
            # print(out_npy_seq_file)
            # print(out_npy_skel_file)
            # 1/0
            # 加载BVH数据
            anim, _, _ = BVH.load(in_bvh_file)

            # 获得要保留的joint
            bvh_file = open(data_path + folder + "/" + cfile).read().split("JOINT")
            bvh_joints = [f.split("\n")[0] for f in bvh_file[1:]]
            to_keep = [0]
            for jname in joints_list:
                for k in range(len(bvh_joints)):
                    if jname == bvh_joints[k][-len(jname) :]:
                        to_keep.append(k + 1)
                        break
            # 重新修改骨架数据(保留 保留的joint)
            anim.parents = anim.parents[to_keep] # 修改joint-parent关系
            for i in range(1, len(anim.parents)):
                """If joint not needed, connect to the previous joint"""
                if anim.parents[i] not in to_keep:
                    anim.parents[i] = anim.parents[i] - 1
                anim.parents[i] = to_keep.index(anim.parents[i])
            # 提取指定joint-offset-psition, rotation，全局rotation等数据
            anim.positions = anim.positions[:, to_keep, :]
            anim.rotations.qs = anim.rotations.qs[:, to_keep, :]
            anim.orients.qs = anim.orients.qs[to_keep, :]
            
            if anim.positions.shape[0] > 1:
                # 获得joint的全局position
                joints = Animation.positions_global(anim)
                # 尾部，拼接最后一帧
                joints = np.concatenate([joints, joints[-1:]], axis=0)
                new_joints, rotation = process(joints)
                new_joints = new_joints[:, 3:]

                rotation = rotation[:-1]
                anim.rotations[:, 0, :] = rotation[:, 0, :] * anim.rotations[:, 0, :]
                angle = anim.rotations.qs
                pose = np.reshape(new_joints[:, :-8], (new_joints.shape[0], -1, 3))
                
                tgtanim = anim.copy()
                tgtanim.positions[:, 0, :] = new_joints[:, :3]
                poseR = Animation.positions_global(tgtanim)
                #再次执行,看看滤波平滑 造成的最大position变化是多少===发现很小===>3.3880986904932797e-10
                print((poseR - pose).max())
                if not exists(save_path + folder):
                    makedirs(save_path + folder)
                np.save(out_npy_quat_file, angle)
                np.save(out_npy_seq_file, new_joints)
                anim.rotations.qs[...] = anim.orients.qs[None]
                tjoints = Animation.positions_global(anim)
                anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
                anim.positions[:, 0, :] = new_joints[:, :3]  # root position
                np.save(out_npy_skel_file, anim.positions)
                print(anim.parents)
                print("Success.")

print("Done.")
