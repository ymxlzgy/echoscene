import itertools
import sys
import numpy as np
import random
import os
import trimesh
import yaml
import json
import csv
import argparse
from PIL import Image
import pickle
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.datasets.threed_front import ThreedFront
from simple_3dviz import Scene
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render
# from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from utils import floor_plan_from_scene, export_scene, get_textured_objects
from graphviz import Digraph

object_ids_csv_path = '/media/ymxlzgy/Data/Dataset/3D-FRONT/object_ids.csv'


def get_obj_name_color_from_csv(node_name):
    with open(object_ids_csv_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # ['object_name','obj_id', 'color', 'class_id', 'class']

        for row in reader:  # 不包括header
            if row[0] == node_name.rsplit('_',1)[0] or row[0] == node_name:
                return row[0], row[1]
        raise ValueError(f"{node_name} not found {type(node_name)}")

def visualize_scene_graph(obj_dict, relationships, filename):
    from collections import defaultdict
    g = Digraph(format='png')
    nodes=[]
    node_name_cache = {} # "1": "table"
    num_cache = {}
    edges=defaultdict(list)
    obj_dict_new = {}
    for key,value in obj_dict.items():
        if value not in obj_dict_new.values():
            num_cache[value] = 0
            obj_dict_new[key] = value
        else:
            num_cache[value] += 1
            obj_dict_new[key] = value+'_'+str(num_cache[value])
    for relationship in relationships:
        entity_A = relationship[0]
        entity_B = relationship[1]
        rela_text = relationship[3]
        if rela_text == "same style as" or rela_text == "same super category as" or rela_text == "same material as":
            continue

        if entity_A not in nodes:
            nodes.append(obj_dict_new[str(entity_A)])
        if entity_B not in nodes:
            nodes.append(obj_dict_new[str(entity_B)])


        edges[(entity_A, entity_B)].append(rela_text)

    for node in nodes:
        node_name, node_color = get_obj_name_color_from_csv(node)

        g.node(node, node, fontname='helvetica', color=node_color, style='filled')
    for edge in edges:
        edges[edge] = sorted(edges[edge])
        rel_all_text = ", ".join(edges[edge])

        if 'close' in rel_all_text or "symmetrical to" in rel_all_text:
            g.edge(obj_dict_new[str(edge[0])], obj_dict_new[str(edge[1])], label=rel_all_text, color='red', style='dotted') #color='red', style='dotted'
        else:
            g.edge(obj_dict_new[str(edge[0])], obj_dict_new[str(edge[1])], label=rel_all_text, color='grey') #color='red', style='dotted'
    g.render(filename, view=False)

def cal_l2_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def close_dis(corners1,corners2):
    dist = -2 * np.matmul(corners1, corners2.transpose())
    dist += np.sum(corners1 ** 2, axis=-1)[:, None]
    dist += np.sum(corners2 ** 2, axis=-1)[None, :]
    dist = np.sqrt(dist)
    return np.min(dist)

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

# def render_scene(save_path, path_to_floor_plan_textures, scene,trimesh_visual=False, s=None, with_floor_layout=True,with_walls=True, with_door_and_windows=True, without_screen=False):
#     renderables = scene.furniture_renderables(
#         with_floor_plan_offset=True, with_texture=True
#     )
#     trimesh_meshes = []
#     behaviours = []
#     for furniture in scene.bboxes:
#         # Load the furniture and scale it as it is given in the dataset
#         raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
#         raw_mesh.scale(furniture.scale)
#
#         # Create a trimesh object for the same mesh in order to save
#         # everything as a single scene
#         tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
#         tr_mesh.visual.material.image = Image.open(
#             furniture.texture_image_path
#         )
#         tr_mesh.vertices *= furniture.scale
#         theta = furniture.z_angle
#         R = np.zeros((3, 3))
#         R[0, 0] = np.cos(theta)
#         R[0, 2] = -np.sin(theta)
#         R[2, 0] = np.sin(theta)
#         R[2, 2] = np.cos(theta)
#         R[1, 1] = 1.
#         tr_mesh.vertices[...] = \
#             tr_mesh.vertices.dot(R) + furniture.position
#         tr_mesh.vertices[...] = tr_mesh.vertices - scene.centroid
#         trimesh_meshes.append(tr_mesh)
#
#     if with_floor_layout:
#         # Get a floor plan
#         floor_plan, tr_floor, _ = floor_plan_from_scene(
#             scene, path_to_floor_plan_textures, without_room_mask=True
#         )
#         renderables += floor_plan
#         trimesh_meshes += tr_floor
#
#     if with_walls:
#         for ei in scene.extras:
#             if "WallInner" in ei.model_type:
#                 renderables = renderables + [
#                     ei.mesh_renderable(
#                         offset=-scene.centroid,
#                         colors=(0.8, 0.8, 0.8, 0.6)
#                     )
#                 ]
#
#     if with_door_and_windows:
#         for ei in scene.extras:
#             if "Window" in ei.model_type or "Door" in ei.model_type:
#                 renderables = renderables + [
#                     ei.mesh_renderable(
#                         offset=-scene.centroid,
#                         colors=(0.8, 0.8, 0.8, 0.6)
#                     )
#                 ]
#
#     if without_screen:
#         path_to_image = "{}/{}_".format("/media/ymxlzgy/Data/Dataset/3D-FRONT/tmp", scene.uid)
#         behaviours += [SaveFrames(path_to_image + "{:03d}.png", 1)]
#         render(
#             renderables,
#             size=(512,512),
#             camera_position=(-1.5,1.5,1.5),
#             camera_target=(0,0,0),
#             up_vector=(0,1,0),
#             background=[1,1,1,1],
#             behaviours=behaviours,
#             n_frames=1,
#             scene=s
#         )
#     else:
#         show(
#             renderables,
#             behaviours=behaviours + [SnapshotOnKey()],
#             size=(512, 512),
#             camera_position=(-2.0, -2.0, -2.0),
#             camera_target=(0, 0, 0),
#             up_vector=(0, 0, 1),
#             background=[1,1,1,1],
#             light=(-2.0, -2.0, -2.0),
#         )
#     # Create a trimesh scene and export it
#     export_scene(save_path, trimesh_meshes)
#     if trimesh_visual:
#         trimesh.Scene(trimesh_meshes).show()
#     trimesh.Scene(trimesh_meshes).export(os.path.join(save_path,"scene.glb"))


def main(argv):
    parser = argparse.ArgumentParser(
            description="Prepare the 3D-FRONT scene graph"
        )
    parser.add_argument(
            "option",
            help="['bedroom', 'livingroom', 'diningroom', 'library', 'all']"
        )
    parser.add_argument(
        "split",
        help="['train', 'val', 'test', 'all']"
    )
    args = parser.parse_args(argv)
    option = args.option

    path_to_3d_front_dataset_directory="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FRONT"
    path_to_3d_future_dataset_directory="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model"
    path_to_model_info="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model/model_info.json"
    path_to_pickled_3d_futute_models = "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE_pickle/threed_future_model_{}.pkl".format(option)
    d_list = []

    if option == 'all':
        for room_type in ['bedroom', 'livingroom', 'diningroom', 'library']:
            config_file = "/home/ymxlzgy/code/graphto3d_v2/config/{}_sg_config.yaml".format(room_type)
            config = load_config(config_file)
            d = ThreedFront.from_dataset_directory(
                path_to_3d_front_dataset_directory,
                path_to_model_info,
                path_to_3d_future_dataset_directory,
                path_to_room_masks_dir=None,
                path_to_bounds=None,
                filter_fn=filter_function(config["data"],
                split=config[str(args.split)].get("splits", ["train", "val", "test"])))
            d_list.append(d)
            print("Loaded {0} {1}s".format(len(d.scenes), room_type))
    else:
        config_file = "/home/ymxlzgy/code/graphto3d_v2/config/{}_sg_config.yaml".format(option)
        config = load_config(config_file)
        d_list = [ThreedFront.from_dataset_directory(
            path_to_3d_front_dataset_directory,
            path_to_model_info,
            path_to_3d_future_dataset_directory,
            path_to_room_masks_dir=None,
            path_to_bounds=None,
            filter_fn=filter_function(
                config["data"],
                split=config[str(args.split)].get("splits", ["train", "val", "test"])
            )
        )]
        print("Loaded {0} {1}s".format(len(d_list[0].scenes), option))

    # Build the dataset of 3D models
    # objects_dataset = ThreedFutureDataset.from_pickled_dataset(path_to_pickled_3d_futute_models)
    # print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))
    consistency_dict_room = []
    s=Scene(size=(512,512))
    scene_id_cache = []
    for d in d_list:
        for scene in d.scenes:
            obj_dict = {}
            rel = []
            path_to_objs = os.path.join(
                "/media/ymxlzgy/Data/Dataset/3D-FRONT/visualization",
                "{}".format(scene.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            elif os.path.exists(os.path.join(path_to_objs,"consistency.json")):
                if scene.scene_id in scene_id_cache:
                    continue
                scene_id_cache.append(scene.scene_id)
                with open(os.path.join(path_to_objs,"consistency.json")) as file:
                    consistency_dict = file.read()
                    consistency_dict = json.loads(consistency_dict)
                    consistency_dict_room.append(consistency_dict)
                print("already  ", scene.scene_id)
                continue
            for i in range(len(scene.bboxes)):
                obj_dict[str(i+1)] = scene.bboxes[i].label
            obj_pair_list = list(itertools.combinations(range(len(scene.bboxes)),2))

            for obj_pair in obj_pair_list:
                sub = scene.bboxes[obj_pair[0]]
                sub_id = obj_pair[0] + 1
                obj = scene.bboxes[obj_pair[1]]
                obj_id = obj_pair[1] + 1

                # consistency check
                if sub.model_jid == obj.model_jid:
                    rel.append([sub_id, obj_id, 16, "same as"])


            consistency_dict = {
                "scan": scene.scene_id,
                "objects": obj_dict,
                "consistency": rel
            }

            with open(os.path.join(path_to_objs, "consistency.json"), "w") as f:
                json.dump(consistency_dict, f)

            consistency_dict_room.append(consistency_dict)

            scene_vis_filename = os.path.join(path_to_objs,'consistent_graph')
            visualize_scene_graph(consistency_dict["objects"], consistency_dict['consistency'], scene_vis_filename)
            print("generated ", scene.scene_id)
    with open("../GT/3dfront/consistencies_{0}_{1}.json".format(option,args.split), "w") as f:
        graph_dict = {"scans": consistency_dict_room}
        json.dump(graph_dict, f)

if __name__ == "__main__":
    main(sys.argv[1:])