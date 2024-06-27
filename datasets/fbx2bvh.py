import pdb
import bpy
import numpy as np

from os import listdir, makedirs, system
from os.path import exists
"""
利用blender把 fbx转换成bvh
"""
data_path = "./datasets/mixamo/train_char"
directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
for d in directories:
    files = sorted([f for f in listdir(data_path + d) if f.endswith(".fbx")])

    for f in files:
        #输入文件：./datasets/mixamo/train_char/AJ/180 Turn W_ Briefcase.fbx
        sourcepath = data_path + d + "/" + f
        #输出文件 ./datasets/mixamo/train_char/AJ/180 Turn W_ Briefcase.bvh
        dumppath = data_path + d + "/" + f.split(".fbx")[0] + ".bvh"

        if exists(dumppath):
            continue
        
        #导入fbx
        """fbx中可能包括mesh,材质，贴图，camera,animation等"""
        bpy.ops.import_scene.fbx(filepath=sourcepath)

        # 获得开始帧，和结束帧
        frame_start = int(9999)
        frame_end = int(-9999)
        action = bpy.data.actions[-1]
        if action.frame_range[1] > frame_end:
            frame_end = int(action.frame_range[1])
        if action.frame_range[0] < frame_start:
            frame_start = int(action.frame_range[0])

        # 最大60帧
        frame_end = np.max([60, frame_end])
        # 输出为BVH（只保留BVH动画）
        bpy.ops.export_anim.bvh(
            filepath=dumppath,
            frame_start=frame_start,
            frame_end=frame_end,
            root_transform_only=True,
        )
        bpy.data.actions.remove(bpy.data.actions[-1])
        print(data_path + d + "/" + f + " processed.")
