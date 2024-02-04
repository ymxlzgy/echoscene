import glob
import numpy as np
import os
import json
import trimesh
import pyrender
import cv2
import h5py
import torch
import seaborn as sns
from model.diff_utils.util_3d import render_sdf, render_mesh, sdf_to_mesh
from helpers.util import pytorch3d_to_trimesh, fit_shapes_to_box_v2

## Need to download 3D-FUTURE-SDF
# files from SG-FRONT
obj_info_path_test = "/media/ymxlzgy/Data/Dataset/3D-FRONT/obj_boxes_all_test.json"
obj_info_path_trainval = "/media/ymxlzgy/Data/Dataset/3D-FRONT/obj_boxes_all_trainval.json"
rel_trainval_file = "/media/ymxlzgy/Data/Dataset/3D-FRONT/relationships_all_trainval.json"
rel_test_file = "/media/ymxlzgy/Data/Dataset/3D-FRONT/relationships_all_test.json"
class_file = "/media/ymxlzgy/Data/Dataset/3D-FRONT/classes_all.txt"

bath = '/media/ymxlzgy/Data/graphto3d_v2_test/sdf_fov90_h8'
cat = {}
large = False
without_lamp = False
no_stool = True
mapping_file = "/media/ymxlzgy/Data/Dataset/3D-FRONT/mapping.json" if not no_stool else "/media/ymxlzgy/Data/Dataset/3D-FRONT/mapping_no_stool.json"
if without_lamp:
    bath += '_wo_lamp'
if no_stool:
    bath += '_no_stool'

if large:
    img_path = os.path.join(bath, 'large')
    with open(class_file) as f:
        for line in f:
            category = line.rstrip()
            cat[category] = category
    classes = dict(zip(sorted(cat), range(len(cat))))
else:
    img_path = os.path.join(bath, 'small')
    mapping_full2simple = json.load(open(mapping_file, "r"))
    classes = dict(zip(sorted(list(set(mapping_full2simple.values()))), range(len(list(set(mapping_full2simple.values()))))))

classes_r = dict(zip(classes.values(), classes.keys()))

def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for tri_mesh in trimesh_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi/2)

    # set up positions and the origin
    camera_location = np.array([0.0, 8.0, 0.0])  # y axis
    look_at_point = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, -1.0])  # -z axis

    camera_direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    right_vector = np.cross(camera_direction, up_vector)
    up_vector = np.cross(right_vector, camera_direction)

    camera_pose = np.identity(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = up_vector
    camera_pose[:3, 2] = -camera_direction
    camera_pose[:3, 3] = camera_location
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    # 添加一个点光源，更改颜色和强度
    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=camera_pose)
    color, depth = renderer.render(scene)
    return color


num_classes = len(classes_r.values())
color_palette = np.array(sns.color_palette('hls', num_classes))

##################################################################################

# with open(rel_trainval_file) as f:
#     rel = json.load(f)
# img_path_trainval = os.path.join(img_path,'trainval')
# if not os.path.exists(img_path_trainval):
#     os.makedirs(img_path_trainval)
#
# for root , direct, files in os.walk(img_path_trainval):
#     existed_files = files
#
# for info in rel['scans']:
#     obj_mesh_list = []
#     scan_id = info['scan']
#     cat_names = list(info['objects'].values())
#     if scan_id+'.png' in existed_files:
#         continue
#     print(scan_id)
#     obj_list = sorted(glob.glob(os.path.join(image_path,scan_id,'*.obj')))
#     for obj, cat_name in zip(obj_list[:-1], cat_names[:-1]):
#         obj_mesh = trimesh.load(obj)
#         obj_mesh = trimesh.Trimesh(vertices=obj_mesh.vertices,faces=obj_mesh.faces)
#         color = color_palette[classes[cat_name]] if large else color_palette[classes[mapping_full2simple[cat_name]]]
#         obj_mesh.visual.vertex_colors = color
#         obj_mesh.visual.face_colors = color
#         obj_mesh_list.append(obj_mesh)
#     color_img = render_img(obj_mesh_list)
#     color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
#     cv2.imwrite(os.path.join(img_path_trainval, '{}.png'.format(scan_id)), color_bgr)

##################################################################################

with open(obj_info_path_test) as f:
    obj_info = json.load(f)
with open(rel_test_file) as f:
    rel = json.load(f)
for info in rel['scans']:
    obj_mesh_list = []
    scan_id = info['scan']
    for k, v in info["objects"].items():
        # floor
        if obj_info[scan_id][k]['model_path'] is None:
            continue
        bbox = obj_info[scan_id][k]['param7']
        bbox[3:6] -= np.array(obj_info[scan_id]['scene_center'])  # centered in the scene
        class_ = mapping_full2simple[v] if not large else v
        if without_lamp and (class_ == 'lamp' or class_ == 'ceiling_lamp' or class_ == 'pendant_lamp'):
            continue
        class_id = classes[class_]
        color = color_palette[class_id]

        # the base of the model path should be changed to your own path
        sdf_path = obj_info[scan_id][k]['model_path'].replace('3D-FUTURE-model','3D-FUTURE-SDF').replace('raw_model.obj', 'ori_sample_grid.h5')
        h5_f = h5py.File(sdf_path, 'r')
        obj_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(obj_sdf).view(1, 64, 64, 64)
        sdf = torch.clamp(sdf, min=-0.2, max=0.2)
        pyorch3d_mesh = sdf_to_mesh(sdf.view(1, 1, 64, 64, 64), render_all=True)
        trimesh_mesh = pytorch3d_to_trimesh(pyorch3d_mesh)
        trimesh_mesh.visual.vertex_colors = color
        trimesh_mesh.visual.face_colors = color
        box_points, obj = fit_shapes_to_box_v2(trimesh_mesh, bbox, degrees=False)
        obj_mesh_list.append(obj)

    # scene = trimesh.Scene(obj_mesh_list)
    # scene.show()

    img_path_test = os.path.join(img_path, 'test')
    if not os.path.exists(img_path_test):
        os.makedirs(img_path_test)

    color_img = render_img(obj_mesh_list)
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(os.path.join(img_path_test, '{}.png'.format(scan_id)), color_bgr)
