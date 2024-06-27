import os
import bmesh
import numpy as np
import sys
# import tqdm
import bpy

sys.path.append("./outside-code")
sys.path.append("./datasets")


from bvh import BVHData
from os import listdir, makedirs
from os.path import exists, join


def rm_prefix(str):
    if ':' in str:
        return str[str.index(':') + 1 :]
    else:
        return str


def get_width(vertices):
    # vertices: (num_joint, 3[x,y,z])
    """获得整个joint点集的 包围box"""
    box = np.zeros((2, 3))
    box[0, :] = vertices.min(axis=0) # 竖向看-所有joint的xyz的最小值
    box[1, :] = vertices.max(axis=0) # 竖向看+所有joint的xyz的最大值
    width = box[1, :] - box[0, :]  # 获得整个joint点集的 包围box
    return width


'''22 joints mixamo'''
body_joint_lst = [0, 1, 2, 3]
right_arm_joint_lst = [14, 15, 16, 17]  # 左臂
left_arm_joint_lst = [18, 19, 20, 21]  # 右臂
arms_joint_lst = right_arm_joint_lst + left_arm_joint_lst


def extract_data(fbx_path, subject_name, save_path):
    # 导入场景，并设置使用anim
    bpy.ops.import_scene.fbx(filepath=fbx_path, use_anim=True)
    context = bpy.context 
    scene = context.scene  # 获取场景信息

    # obtain mesh under rest pose
    bpy.context.object.data.pose_position = 'REST' # 所有对象恢复到"默认姿势"

    # 在Blender的object中查找名为'Armature'的对象
    # 在FBX文件中，Armature对象通常代表 角色骨架 
    source_arm = bpy.data.objects['Armature']
    # 获得 第一个骨骼（根骨骼-hips） 的position
    rest_x, rest_y, rest_z = source_arm.data.bones[0].head_local
    # rest_x, rest_y, rest_z = 0, 0, 0
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            # 将该mesh对象 复制一份放到bmesh中
            bme_rest = bmesh.new() # BMesh是Blender中用于处理网格数据的库
            bme_rest.from_mesh(obj.data)

            bm_rest_verts = bme_rest.verts # 顶点集合
            bm_rest_faces_ori = bme_rest.faces # 面集合(可能有多边形面)
            # 将面集合，进行三角化，形成新的面集合
            bm_rest_faces_tri = bmesh.ops.triangulate(
                bme_rest,
                faces=bm_rest_faces_ori,
                quad_method='BEAUTY', # 三角化参数
                ngon_method='BEAUTY', #三角化参数
            )['faces']

            rest_verts_lst = [] # [（每个顶点的相对xyz坐标）]
            for v in bm_rest_verts:
                # 获得每个顶点的 相对（root-hips关节）的position
                rest_verts_lst.append((v.co.x - rest_x, v.co.y - rest_y, v.co.z - rest_z))

            # 将三角面的顶点索引收集起来
            rest_faces_lst = ([])   # 【三角面数，顶点3坐标】
            for face in bm_rest_faces_tri:
                f_verts = face.verts
                rest_faces_lst.append(
                    (f_verts[0].index, f_verts[1].index, f_verts[2].index)
                )
            # 全部顶点集合
            np_rest_verts = np.array(rest_verts_lst) # # [顶点，（每个顶点的相对xyz坐标）]
            # 三角面中的顶点索引
            np_rest_faces = np.array(rest_faces_lst)

    # obtain the frame indices of begining and ending
    bpy.context.object.data.pose_position = 'POSE'
    a = bpy.context.object.animation_data.action
    frame_start, frame_end = map(int, a.frame_range)
    seq_length = frame_end - frame_start + 1

    # 到处fbx-->bvh
    output_bvh_path = fbx_path.replace('fbx', 'bvh')
    bpy.ops.export_anim.bvh(
        filepath=output_bvh_path,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=True,
    )

    bvh_data = BVHData(output_bvh_path)

    # ====== extract data block ======
    # extract skinning weight and simplify it
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            # 顶点
            verts = obj.data.vertices # []
            # 顶点组(每个顶点组对应一个骨骼)
            vgrps = obj.vertex_groups  # vertex groups correspond to the joints.
            # print(vgrps.keys())  {'Hips': VertexGroup("Hips"), 'Spine':...}
            # 注意矩阵的size [定点数，顶点组数]
            np_skinning_weights = np.zeros((len(verts), len(vgrps)))
            mask = np.zeros(np_skinning_weights.shape, dtype=np.int)
            vgrp_label = vgrps.keys() # [骨骼名称列表]

            for i, vert in enumerate(verts):
                for g in vert.groups: # 能影响 该顶点的组
                    j = g.group # 一个组 对应一个bone
                    # 组j 对顶点i的 影响权重
                    np_skinning_weights[i, j] = g.weight  # 一个数值：1/0
                    # 设置影响关系
                    mask[i, j] = 1

        if obj.type == 'ARMATURE':
            source_arm = bpy.data.objects[obj.name]
    # 顶点---joint-name 的 影响映射
    #print(bvh_data.simplified_joint_names) [精简后的骨架-骨骼]
    # ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']

    # 精简骨架的 [顶点，bone] 的权重映射
    np_simplified_skinning_weights = np.zeros(
        (len(verts), len(bvh_data.simplified_joint_names))
    )
    for j, name in enumerate(vgrp_label):
        # 原始骨架中的骨骼name
        bone = source_arm.data.bones[name]
        # 把原始骨架中的骨骼 逐步线上查找 到“根骨骼”or"新的精简骨骼"点
        while (
            bone.parent is not None
            and rm_prefix(bone.name) not in bvh_data.simplified_joint_names
        ):
            bone = bone.parent

        idx = bvh_data.simplified_joint_names.index(rm_prefix(bone.name))
        # 原始骨骼被精简成了“新骨骼”，这时还要把 原始骨骼的蒙皮映射，等效到“新骨骼”上
        # 比如原本的spine1,spine2 ---新骨架--->spine
        # 这时把 spine1,spine2 的蒙皮映射，等效累加到 spine骨骼点上
        np_simplified_skinning_weights[:, idx] += np_skinning_weights[:, j]

    vertex_part = np.argmax(np_simplified_skinning_weights, axis=1)
    # print(vertex_part) [n(顶点), 0(影响最大的joint索引)]  # 一维列表
    num_face = np_rest_faces.shape[0]
    face_part = [] # 【面数，该面（第一顶点）影响较大的bone】
    for i in range(num_face):
        # np_rest_faces[i][0]：该三角面的第一个顶点
        # 把 对 该顶点 影响最大 bone添加到 face_part中
        face_part.append(vertex_part[ np_rest_faces[i][0] ])

    face_part = np.array(face_part)
    body_vid_lst = []
    arm_vid_lst = []
    for i in range(vertex_part.shape[0]):
        # vertex_part[i] 对该顶点影响 最大的bone
        if vertex_part[i] in body_joint_lst: # 如果这个Bone在身体骨骼中
            body_vid_lst.append(i)
        if vertex_part[i] in arms_joint_lst: # 如果这个Bone在手臂骨骼中
            arm_vid_lst.append(i)
    # 受身体骨骼点影响较大的点
    rest_body_vertices = np_rest_verts[body_vid_lst, :]
    # 受arm骨骼点影响较大的点
    rest_arm_vertices = np_rest_verts[arm_vid_lst, :]

    # 获得body躯干的包围box
    body_width = get_width(rest_body_vertices)
    # 获得全身的包围box
    full_width = get_width(np_rest_verts)

    # detail shape
    joint_shape_lst = [] # [[该joint影响到的顶点]]  # 数组index 代表关节号
    for i in range(22): # 遍历joint
        joint_i = [] # 该joint影响到的顶点
        for j in range(vertex_part.shape[0]): # 遍历顶点
            if vertex_part[j] == i:
                joint_i.append(j)
        joint_shape_lst.append(joint_i)

    shape_lst = [] # [[该joint影响到的点集的包围box]]  # 数组index 代表关节号
    for joint_i in joint_shape_lst:
        # joint_i：list: 该joint影响到的点集
        if len(joint_i) == 0:
            shape_lst.append(np.array([0, 0, 0]))
        else:
            # 获取这些点集的 具体position坐标  --->这些点集形成一朵“点云”
            joint_i_vertices = np_rest_verts[joint_i, :]
            # 计算这个“点云”的包围box
            joint_i_width = get_width(joint_i_vertices)
            shape_lst.append(joint_i_width)

    #### 汇总，准备返回
    shape_lst_array = np.stack(shape_lst, axis=0)

    skinning_weights_data = np_simplified_skinning_weights.astype(np.single)
    joint_names_data = bvh_data.simplified_joint_names
    root_orient_data = bvh_data.simplified_axis_angle[:, :3].astype(np.single)
    rest_vertices_data = np_rest_verts.astype(np.single)
    rest_faces_data = np_rest_faces
    subject_data = subject_name
    skeleton_data = bvh_data.simplified_joint_offsets.astype(np.single)
    rest_body_vertices_data = rest_body_vertices
    rest_arm_vertices_data = rest_arm_vertices
    output_path = os.path.join(save_path, '%s.npz' % (subject_name))

    np.savez(
        output_path,
        # 精简骨架的 [顶点，bone] 的权重映射
        skinning_weights=skinning_weights_data,
        # [精简后的骨架-骨骼]  [hips, spines...]
        joint_names=joint_names_data,
        # root的朝向
        root_orient=root_orient_data,
        #     # [顶点，（每个顶点的相对xyz坐标）]
        rest_vertices=rest_vertices_data,
        # 单脚面的顶点索引 [三角面数， 3（3个顶点索引）]
        rest_faces=rest_faces_data,
        # 精简骨架的joint-offset-position数据
        skeleton=skeleton_data,
        # character名称
        subject=subject_data,
        # [n(顶点), 0(影响最大的joint索引)]  # 一维列表
        vertex_part=vertex_part,
        # # 受身体骨骼点影响较大的点
        rest_body_vertices=rest_body_vertices_data,
        # # 受arm骨骼点影响较大的点
        rest_arm_vertices=rest_arm_vertices_data,
        # body躯干的包围box
        body_width=body_width,
        # arm臂的包围box
        full_width=full_width,
        # # [[该joint影响到的点集的包围box]]  # 数组index 代表关节号
        joint_shape=shape_lst_array,
    )

    # for obj in bpy.context.scene.objects:
    #     obj.select = True
    bpy.ops.object.delete()


if __name__ == '__main__':
    root_path = "/home/wxk/project_linux/dongbu/R2ET/datasets/mixamo/train_char"
    save_path = "./datasets/mixamo/train_shape"
    for subject_name in listdir(root_path):
        fbx_path = [join(root_path, subject_name, item_file)  for item_file in  
                    listdir(join(root_path, subject_name)) if item_file.endswith(".fbx") ]
        # print(fbx_path)
        # print(subject_name)
        # print(save_path)
        # print("+++++++++++++++++++++")
        print("当前处理：{a}".format(a=fbx_path[0]))
        try:
            extract_data(fbx_path[0], subject_name, save_path)
        except Exception as e:
            print(e)
