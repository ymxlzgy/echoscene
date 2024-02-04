from __future__ import print_function
import sys
sys.path.append("..")
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
from . import util
from tqdm import tqdm
import json
from helpers.psutil import FreeMemLinux
from helpers.util import normalize_box_params, denormalize_box_params, get_rotation
from omegaconf import OmegaConf
import clip
import random
import pickle
import open3d as o3d


def load_ckpt(ckpt):
    map_fn = lambda storage, loc: storage
    if type(ckpt) == str:
        state_dict = torch.load(ckpt, map_location=map_fn)
    else:
        state_dict = ckpt
    return state_dict

class RIODatasetSceneGraph(data.Dataset):
    def __init__(self, root, root_3rscan,
                 label_file, npoints=2500, class_choice=None,
                 split='train', data_augmentation=True, shuffle_objs=False,
                 pass_scan_id=False, use_points=True, use_SDF=False,
                 use_scene_rels=False, data_len=None,
                 with_changes=True, vae_baseline=False,
                 scale_func='diag', eval=False, eval_type='addition',
                 atlas=None, path2atlas=None, with_feats=False, with_CLIP=False,
                 seed=True, use_splits=False, large=False,
                 use_rio27=False, recompute_feats=False, recompute_clip=False, use_canonical=False,
                 crop_floor=False, center_scene_to_floor=False):

        # options currently not used in the experiments
        # for partial scenes (use_splits), it crops the floor around the objects that are part of that scene fraction
        self.crop_floor = crop_floor
        self.center_scene_to_floor = center_scene_to_floor

        self.seed = seed
        self.with_feats = with_feats
        self.with_CLIP = with_CLIP
        self.atlas = atlas
        self.cond_model = None
        self.path2atlas = path2atlas
        self.large = large
        self.recompute_feats = recompute_feats
        self.recompute_clip = recompute_clip

        self.use_canonical = use_canonical

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        self.scale_func = scale_func
        self.with_changes = with_changes
        self.npoints = npoints
        self.use_points = use_points
        self.use_SDF = use_SDF
        self.root = root
        # list of class categories
        self.catfile = os.path.join(self.root, 'classes.txt')
        self.cat = {}
        self.scans = []
        self.data_augmentation = data_augmentation
        self.data_len = data_len
        self.vae_baseline = vae_baseline
        self.use_scene_rels = use_scene_rels

        self.fm = FreeMemLinux('GB')
        self.vocab = {}
        with open(os.path.join(self.root, 'classes.txt'), "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = f.readlines()

        splitfile = os.path.join(self.root, '{}.txt'.format(split))

        filelist = open(splitfile, "r").read().splitlines()
        self.filelist = [file.rstrip() for file in filelist] # training list
        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))

        # uses scene sections of up to 9 objects (from 3DSSG) if true, and full scenes otherwise
        self.use_splits = use_splits
        if split == 'train_scans': # training set
            splits_fname = 'relationships_train_clean' if self.use_splits else 'relationships_merged_train_clean'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_train_refined.json')
            self.floor_json_file = os.path.join(self.root, 'floor_boxes_split_train.json')
        else: # validation set
            splits_fname = 'relationships_validation_clean' if self.use_splits else 'relationships_merged_validation_clean'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_val_refined.json')
            self.floor_json_file = os.path.join(self.root, 'floor_boxes_split_val.json')

        if self.crop_floor:
            with open(self.floor_json_file, "r") as read_file:
                self.floor_data = json.load(read_file)

        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_json_file, self.box_json_file)

        self.label_file = label_file

        self.padding = 0.2
        self.eval = eval

        self.pass_scan_id = pass_scan_id

        self.shuffle_objs = shuffle_objs

        self.root_3rscan = root_3rscan
        if self.root_3rscan == '':
            self.root_3rscan = os.path.join(self.root, "data")

        # option to map object classes to a smaller class set from 3RScan
        # not used in the current paper results
        self.use_rio27 = use_rio27
        if self.use_rio27:
            self.vocab_rio27 = json.load(open(os.path.join(self.root, "classes_rio27.json"), "r"))
            self.vocab['object_idx_to_name'] = self.vocab_rio27['rio27_idx_to_name']
            self.vocab['object_name_to_idx'] = self.vocab_rio27['rio27_name_to_idx']
        self.mapping_full2rio27 = json.load(open(os.path.join(self.root, "mapping_full2rio27.json"), "r"))

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # "_scene_"-"windowsill": 0-160
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))

        # we had to discard some underrepresented classes for the shape generation
        # either not part of shapenet, or limited and low quality samples in 3rscan
        if not large:
            points_classes = ['bed', 'chair', 'armchair', 'desk', 'door', 'floor', 'picture', 'sofa', 'couch',
                              'stool', 'table']
        else: # larger set of shape classes
            points_classes = ['ball', 'basket', 'bench', 'bed', 'box', 'cabinet', 'chair', 'armchair',
                              'desk', 'door', 'floor', 'picture', 'sofa', 'couch', 'commode', 'monitor',
                              'stool', 'tv', 'table']
        if self.use_rio27:
            points_classes = list(self.vocab_rio27['rio27_name_to_idx'].keys())
            points_classes.remove("_scene_")
            points_classes.remove("wall")
            points_classes.remove("ceiling")

        points_classes_idx = []
        for pc in points_classes:
            if class_choice is not None:
                if pc in self.classes:
                    points_classes_idx.append(self.classes[pc])
                else:
                    points_classes_idx.append(0)
            else:
                if not use_rio27:
                    points_classes_idx.append(self.classes[pc])
                else:
                    points_classes_idx.append(int(self.vocab_rio27['rio27_name_to_idx'][pc]))

        self.point_classes_idx = points_classes_idx + [0]
        self.sorted_cat_list = sorted(self.cat)
        self.files = {}
        self.eval_type = eval_type

        # check if all shape features exist. If not they get generated here (once)
        if with_feats:
            print('Checking for missing feats. This can be slow the first time.\nThis process needs to be only run once!')
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_feats = False

        # check if all clip/bert features exist. If not they get generated here (once)
        if self.with_BERT:
            print(
                'Checking for missing bert feats. This can be slow the first time.\nThis process needs to be only run once!')
            df_conf = OmegaConf.load('../model/pretrained_model/config/bert.yaml')
            bert_params = df_conf.bert.params
            self.text_embed_dim = bert_params.n_embed
            self.cond_model = BERTTextEncoder(**bert_params)
            self.cond_model.to('cuda')
            print('loading BERT')
            state_dict = load_ckpt('../model/pretrained_model/bert.pth')
            self.cond_model.load_state_dict(state_dict['cond_model'])
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_bert = False
        # check if all clip/bert features exist. If not they get generated here (once)
        if self.with_CLIP:
            self.cond_model, preprocess = clip.load("ViT-B/32", device='cuda')
            print('loading CLIP')
            print(
                'Checking for missing clip feats. This can be slow the first time.\nThis process needs to be only run once!')
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_clip = False

    def read_relationship_json(self, json_file, box_json_file):
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scan in data['scans']:

                relationships = []
                for relationship in scan["relationships"]:
                    relationship[2] -= 1
                    relationships.append(relationship)

                # for every scan in rel json, we append the scan id
                rel[scan["scan"] + "_" + str(scan["split"])] = relationships
                self.scans.append(scan["scan"] + "_" + str(scan["split"]))

                objects = {}
                boxes = {}
                for k, v in scan["objects"].items():
                    objects[int(k)] = v
                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param7'] = box_data[scan["scan"]][k]["param7"]
                        boxes[int(k)]['param7'][6] = np.deg2rad(boxes[int(k)]['param7'][6])
                        if self.use_canonical:
                            if "direction" in box_data[scan["scan"]][k].keys():
                                boxes[int(k)]['direction'] = box_data[scan["scan"]][k]["direction"]
                            else:
                                boxes[int(k)]['direction'] = 0
                    except:
                        # probably box was not saved because there were 0 points in the instance!
                        continue
                objs[scan["scan"] + "_" + str(scan["split"])] = objects
                tight_boxes[scan["scan"] + "_" + str(scan["split"])] = boxes
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    def load_points(self, filename, factor=1, filter_mask=False):
        point_set, _, _, mask = util.read_ply(filename)
        if filter_mask:
            point_set = point_set[np.where(mask > 0)[0], :]
        choice = np.random.choice(len(point_set), self.npoints * factor, replace=True)
        point_set = point_set[choice, :]
        if len(mask) > 0:
            mask.shape = (mask.shape[0], 1)  # in place reshape
            mask = mask[choice, :]
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        point_set = torch.from_numpy(point_set)
        return point_set, mask

    import open3d as o3d
    import numpy as np

    def point_cloud_to_sdf(self, point_cloud, voxel_size, normals_estimation_radius, sdf_truncation_distance):
        # Load point cloud from file or create from numpy array
        if isinstance(point_cloud, str):
            pcd = o3d.io.read_point_cloud(point_cloud)
        elif isinstance(point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            raise ValueError("Invalid input type. Expected a file path or numpy array.")

        # Estimate surface normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normals_estimation_radius, max_nn=30))

        # Create voxel grid from point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        # Compute SDF from point cloud
        sdf = o3d.pipelines.color_map.ComputeTSDFVolume(voxel_grid, pcd, sdf_truncation_distance)

        return sdf

    def visualize_sdf(self, sdf, iso_value=0.0):
        # Extract surface mesh from SDF using Marching Cubes algorithm
        mesh = sdf.extract_surface_mesh(iso_value)

        # Visualize the mesh
        o3d.visualization.draw_geometries([mesh])


    def norm_tensor(self, p, params7=None, scale=False, center=True, rotation=False, scale_func='diag'):
        """ Given a set of points of an object and (optionally) a oriented 3D box, normalize points

        :param p: tensor of 3D pointset
        :param params7: bounding box parameters [W, L, H, Cx, Cy, Cy, Z]
        :param scale: boolean, if true apply scaling to pointset p according to scale_func
        :param center: boolean, if true normalize the center of points to 0,0,0
        :param rotation: boolean, if true rotate points based on the box rotation in param7
        :param scale_func: string specifying the function used for scaling. 'diag' normalizes the diagonal to length 1.
        'whl' sets each dimension to range [-1,1].
        :return: the normalized tensor of 3D pointset
        """
        if center:
            if params7 is None:
                # this is center of mass
                mean = torch.mean(p, dim=0)
            else:
                # get center from box center if available
                mean = torch.from_numpy(params7[3:6].astype("float32"))
            p -= mean.unsqueeze(0)
        if rotation and params7 is not None:
            p = (torch.from_numpy(get_rotation(-params7[-1], degree=False).astype("float32")) @ p.T).T
        if scale and params7 is not None:
            # first if needed rotate to canonical rotation
            # apply scaling
            # if needed rotate back
            if not rotation:
                p = (torch.from_numpy(get_rotation(-params7[-1], degree=False).astype("float32")) @ p.T).T
            if scale_func == 'diag':
                # OPTION 1: normalize diagonal = 1
                norm2 = np.linalg.norm(params7[:3].astype("float32"))
                p /= norm2
            elif scale_func == 'whl':
                # OPTION 2: normalize each axis by H, W, L
                norm2 = torch.from_numpy(params7[:3].astype("float32")).reshape(1, 3)
                min_p = p.min(0)[0]
                p = ((p - min_p) / norm2) * 2. - 1.  # between -1 and 1 in all directions
            elif scale_func == 'whl_after':
                norm2 = p.max(0)[0] - p.min(0)[0]
                min_p = p.min(0)[0]
                p = ((p - min_p) / norm2) * 2. - 1.  # between -1 and 1 in all directions
            else:
                raise NotImplementedError
            if not rotation:
                p = (torch.from_numpy(get_rotation(params7[-1], degree=False).astype("float32")) @ p.T).T
        return p

    def load_semseg(self, json_file):
        """ Loads semantic segmentation from json file

        :param json_file: path to file
        :return: dict, that maps instance label to text semantic label
        """
        instance2label = {}
        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for segGroups in data['segGroups']:
                instance2label[segGroups["id"]] = segGroups["label"].lower()
        return instance2label

    def get_key(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return None

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.split('_')[0]
        split = scan_id.split('_')[1]

        if self.crop_floor:
            scene_floor = self.floor_data[scan_id_no_split][split]
            floor_idx = list(scene_floor.keys())[0]
            if self.center_scene_to_floor:
                scene_center = np.asarray(scene_floor[floor_idx]['params7'][3:6])
            else:
                scene_center = np.array([0, 0, 0])

            min_box = np.asarray(scene_floor[floor_idx]['min_box']) - scene_center
            max_box = np.asarray(scene_floor[floor_idx]['max_box']) - scene_center

        file = os.path.join(self.root_3rscan, scan_id_no_split, self.label_file)
        if os.path.exists(os.path.join(self.root_3rscan, scan_id_no_split, "semseg.v2.json")):
            semseg_file = os.path.join(self.root_3rscan, scan_id_no_split, "semseg.v2.json")
        elif os.path.exists(os.path.join(self.root_3rscan, scan_id_no_split, "semseg.json")):
            semseg_file = os.path.join(self.root_3rscan, scan_id_no_split, "semseg.json")
        else:
            raise FileNotFoundError("Cannot find semseg.json file.")

        # instance2label, the whole instance ids in this scene e.g. {1: 'floor', 2: 'wall', 3: 'picture', 4: 'picture'}
        instance2label = self.load_semseg(semseg_file)
        # real needed classes
        selected_instances = list(self.objs_json[scan_id].keys())
        keys = list(instance2label.keys())

        if self.shuffle_objs:
            random.shuffle(keys)

        feats_in = None
        bert_feats_ins = None
        bert_feats_rel = None
        clip_feats_ins = None
        clip_feats_rel = None
        # If true, expected paths to saved bert features will be set here
        if self.with_BERT:
            self.bert_feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                      'BERT_{}_{}.pkl'.format('splits' if self.use_splits else 'merged',
                                                                  scan_id.split('_')[1]))
            if self.crop_floor:
                self.bert_feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                          'BERT_{}_{}_floor.pkl'.format('splits' if self.use_splits else 'merged',
                                                                            scan_id.split('_')[1]))
            if self.recompute_bert:
                self.bert_feats_path += 'tmp'

        if self.with_CLIP:
            self.clip_feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                                'CLIP_{}_{}.pkl'.format('splits' if self.use_splits else 'merged',
                                                                        scan_id.split('_')[1]))
            if self.crop_floor:
                self.clip_feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                                    'CLIP_{}_{}_floor.pkl'.format(
                                                        'splits' if self.use_splits else 'merged',
                                                        scan_id.split('_')[1]))
            if self.recompute_clip:
                self.clip_feats_path += 'tmp'


        # If true, expected paths to saved atlasnet features will be set here
        if self.with_feats and self.path2atlas is not None:
            _, atlasname = os.path.split(self.path2atlas)
            atlasname = atlasname.split('.')[0]

            if not self.large:
                feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                          '{}_small_{}_{}.pkl'.format(atlasname,
                                                                      'splits' if self.use_splits else 'merged',
                                                                      scan_id.split('_')[1]))
            else:
                feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                          '{}_large_{}_{}.pkl'.format(atlasname,
                                                                      'splits' if self.use_splits else 'merged',
                                                                      scan_id.split('_')[1]))
                if self.crop_floor:
                    feats_path = os.path.join(self.root_3rscan, scan_id.split('_')[0],
                                          '{}_large_{}_{}_floor.pkl'.format(atlasname,
                                                                      'splits' if self.use_splits else 'merged',
                                                                      scan_id.split('_')[1]))
            if self.recompute_feats:
                feats_path += 'tmp'

        # Load points if with features but features cannot be found or are forced to be recomputed
        # Loads points if use_points is set to true
        if (self.with_feats and (not os.path.exists(feats_path) or self.recompute_feats)) or self.use_points:
            if file in self.files: # Caching
                (points, instances) = self.files[file]
            else:
                points, instances, _, _ = util.read_ply(file)

                if self.fm.user_free > 5:
                    self.files[file] = (points, instances)

            if self.crop_floor and self.center_scene_to_floor:
                print("shifting points")
                points = points - scene_center.reshape(1, -1)

        instance2mask = {}
        instance2mask[0] = 0

        cat = []
        tight_boxes = []

        counter = 0

        instances_order = []
        selected_shapes = []

        # key: 1 of 1: 'floor' instance_id              keys: whole instance ids
        for key in keys:
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key
            scene_instance_class = instance2label[key]
            scene_class_id = -1
            if scene_instance_class in self.classes and \
                    (not self.use_rio27 or self.mapping_full2rio27[scene_instance_class] != '-'):
                if self.use_rio27:
                    scene_instance_class = self.mapping_full2rio27[scene_instance_class]
                    scene_class_id = int(self.vocab_rio27['rio27_name_to_idx'][scene_instance_class])
                else:
                    scene_class_id = self.classes[scene_instance_class] # clasee id in the entire dataset ids
            if scene_class_id != -1 and key in selected_instances:
                instance2mask[scene_instance_id] = counter + 1
                counter += 1
            else:
                instance2mask[scene_instance_id] = 0

            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0) and (key in selected_instances):
                if self.use_canonical:
                    direction = self.tight_boxes_json[scan_id][key]['direction']
                    if direction in [-1, 0, 6]:
                        # skip invalid point clouds with ambiguous direction annotation
                        selected_shapes.append(False)
                    else:
                        selected_shapes.append(True)
                cat.append(scene_class_id)
                bbox = self.tight_boxes_json[scan_id][key]['param7'].copy()
                if self.crop_floor and key in self.floor_data[scan_id_no_split][split].keys():
                    bbox = self.floor_data[scan_id_no_split][split][key]['params7'].copy()
                    bbox[6] = np.deg2rad(bbox[6])
                    direction = self.floor_data[scan_id_no_split][split][key]['direction']

                if self.crop_floor and self.center_scene_to_floor:
                    bbox[3:6] -= scene_center

                if self.use_canonical:
                    if direction > 1 and direction < 5:
                        # update direction-less angle with direction data (shifts it by 90 degree
                        # for every added direction value
                        bbox[6] += (direction - 1) * np.deg2rad(90)
                        if direction == 2 or direction == 4:
                            temp = bbox[0]
                            bbox[0] = bbox[1]
                            bbox[1] = temp
                    # for other options, do not change the box
                instances_order.append(key)
                if not self.vae_baseline:
                    bins = np.linspace(0, np.deg2rad(360), 24)
                    angle = np.digitize(bbox[6], bins)
                    bbox = normalize_box_params(bbox)
                    bbox[6] = angle
                else:
                    bins = np.linspace(0, np.deg2rad(360), 24)
                    bbox[6] = np.digitize(bbox[6], bins)
                tight_boxes.append(bbox)

        if self.with_BERT:
            # If precomputed features exist, we simply load them
            if os.path.exists(self.bert_feats_path):
                bert_feats_dic = pickle.load(open(self.bert_feats_path, 'rb'))

                bert_feats_ins = bert_feats_dic['instance_feats']
                bert_feats_order = np.asarray(bert_feats_dic['instance_order'])
                ordered_feats = []
                for inst in instances_order:
                    bert_feats_in_instance = inst == bert_feats_order
                    ordered_feats.append(bert_feats_ins[:-1][bert_feats_in_instance])
                ordered_feats.append(bert_feats_ins[-1][np.newaxis,:])
                bert_feats_ins = list(np.concatenate(ordered_feats, axis=0))
                bert_feats_rel = bert_feats_dic['rel_feats']

        if self.with_CLIP:
            # If precomputed features exist, we simply load them
            if os.path.exists(self.clip_feats_path):
                clip_feats_dic = pickle.load(open(self.clip_feats_path, 'rb'))

                clip_feats_ins = clip_feats_dic['instance_feats']
                clip_feats_order = np.asarray(clip_feats_dic['instance_order'])
                ordered_feats = []
                for inst in instances_order:
                    clip_feats_in_instance = inst == clip_feats_order
                    ordered_feats.append(clip_feats_ins[:-1][clip_feats_in_instance])
                ordered_feats.append(clip_feats_ins[-1][np.newaxis,:])
                clip_feats_ins = list(np.concatenate(ordered_feats, axis=0))
                clip_feats_rel = clip_feats_dic['rel_feats']

        if self.with_feats:
            # If precomputed features exist, we simply load them
            if os.path.exists(feats_path):
                feats_dic = pickle.load(open(feats_path, 'rb'))

                feats_in = feats_dic['feats']
                feats_order = np.asarray(feats_dic['instance_order'])
                ordered_feats = []
                for inst in instances_order:
                    feats_in_instance = inst == feats_order
                    ordered_feats.append(feats_in[:-1][feats_in_instance])
                ordered_feats.append(np.zeros([1, feats_in.shape[1]]))
                feats_in = list(np.concatenate(ordered_feats, axis=0))

        # Sampling of points from object if they are loaded
        if self.use_points:
            masks = np.array(list(map(lambda l: instance2mask[l] if l in instance2mask.keys() else 0, instances)),
                             dtype=np.int32)
            num_pointsets = len(cat) + int(self.use_scene_rels)  # add zeros for the scene node
            obj_points = torch.zeros([num_pointsets, self.npoints, 3])

            for i in range(len(cat)):
                obj_pointset = points[np.where(masks == i + 1)[0], :]

                if self.crop_floor and self.vocab['object_idx_to_name'][cat[i]].split('\n')[0] == 'floor':
                    filter_mask = (obj_pointset[:,0] > min_box[0]) * (obj_pointset[:,0] < max_box[0]) \
                      * (obj_pointset[:,1] > min_box[1]) * (obj_pointset[:,1] < max_box[1])

                    obj_pointset = obj_pointset[np.where(filter_mask > 0)[0], :]

                    print(self.vocab['object_idx_to_name'][cat[i]].split('\n')[0], len(obj_pointset))
                if len(obj_pointset) >= self.npoints:
                    choice = np.random.choice(len(obj_pointset), self.npoints, replace=False)
                else:
                    choice = np.arange(len(obj_pointset))
                    # use repetitions to fill some more points
                    choice2 = np.random.choice(len(obj_pointset), self.npoints - choice.shape[0], replace=True)
                    choice = np.concatenate([choice, choice2], 0)
                    random.shuffle(choice)

                obj_pointset = obj_pointset[choice, :]
                obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))

                if not self.vae_baseline:
                    obj_pointset = self.norm_tensor(obj_pointset, denormalize_box_params(tight_boxes[i]),
                                           scale=True, rotation=self.use_canonical, scale_func=self.scale_func)
                else:
                    obj_pointset = self.norm_tensor(obj_pointset, np.asarray(tight_boxes[i]),
                                                    scale=True, rotation=self.use_canonical, scale_func=self.scale_func)
                if 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(obj_pointset.cpu().numpy())
                    o3d.visualization.draw_geometries([pcd], '{}'.format(self.classes_r[cat[i]]), width=500, height=500)
                obj_points[i] = obj_pointset

            #TODO
            if self.use_SDF:
                obj_sdfs = []
                for obj_point in obj_points:
                    # Example usage
                    point_cloud_data = obj_point.detach().cpu().numpy()
                    voxel_size = 0.05
                    normals_estimation_radius = 0.1
                    sdf_truncation_distance = 2 * voxel_size

                    obj_sdf = self.point_cloud_to_sdf(point_cloud_data, voxel_size, normals_estimation_radius,
                                             sdf_truncation_distance)
                    if 1:
                        # Example usage
                        self.visualize_sdf(obj_sdf)
                    obj_sdfs.append(obj_sdf)
        else:
            obj_points = None



        triples = []
        words = []
        rel_json = self.relationship_json[scan_id]
        for r in rel_json: # create relationship triplets from data
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = instance2mask[r[0]] - 1
                object = instance2mask[r[1]] - 1
                predicate = r[2] + 1
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
                    words.append(instance2label[r[0]]+' '+r[3]+' '+instance2label[r[1]])
            else:
                continue

        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat)
            for i, ob in enumerate(cat):
                triples.append([i, 0, scene_idx])
                words.append(self.get_key(self.classes, ob) + ' ' + 'in' + ' ' + 'room')
            cat.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1, -1])

        output = {}
        if self.use_points:
            output['scene'] = points

        # if features are requested but the files don't exist, we run all loaded pointclouds through atlasnet
        # to compute them and then save them for future usage
        if self.with_BERT and (not os.path.exists(self.bert_feats_path) or bert_feats_ins is None) and self.cond_model is not None:
            num_cat = len(cat)
            feats_ins = []
            feats_rel = {}
            with torch.no_grad():
                for i in range(num_cat-1):
                    # print(self.get_key(self.classes,cat[i]))
                    feats_ins.append(self.cond_model(self.get_key(self.classes,cat[i])).detach().cpu().numpy()) # 1*77*1280
                feats_ins.append(self.cond_model('room').detach().cpu().numpy())
                feats_ins = np.vstack(feats_ins)
                for i in range(len(words)):
                    feats_rel[words[i]] = self.cond_model(words[i]).detach().cpu().numpy()
            bert_feats_in = {}
            bert_feats_in['instance_feats'] = feats_ins
            bert_feats_in['instance_order'] = instances_order
            bert_feats_in['rel_feats'] = feats_rel
            # feats_in = list(feats)
            path = os.path.join(self.bert_feats_path)
            if self.recompute_bert:
                path = path[:-3]

            pickle.dump(bert_feats_in, open(path, 'wb'))

        # if features are requested but the files don't exist, we run all loaded pointclouds through atlasnet
        # to compute them and then save them for future usage
        if self.with_CLIP and (not os.path.exists(self.clip_feats_path) or clip_feats_ins is None) and self.cond_model is not None:
            num_cat = len(cat)
            feats_rel = {}
            obj_cat = []
            with torch.no_grad():
                for i in range(num_cat - 1):
                    obj_cat.append(self.get_key(self.classes, cat[i]))
                obj_cat.append('room')
                text_obj = clip.tokenize(obj_cat).to('cuda')

                feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
                text_rel = clip.tokenize(words).to('cuda')
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                for i in range(len(words)):
                    feats_rel[words[i]] = rel[i]

            clip_feats_in = {}
            clip_feats_in['instance_feats'] = feats_ins
            clip_feats_in['instance_order'] = instances_order
            clip_feats_in['rel_feats'] = feats_rel
            # feats_in = list(feats)
            path = os.path.join(self.clip_feats_path)
            if self.recompute_clip:
                path = path[:-3]

            pickle.dump(clip_feats_in, open(path, 'wb'))

        # if features are requested but the files don't exist, we run all loaded pointclouds through atlasnet
        # to compute them and then save them for future usage
        if self.with_feats and (not os.path.exists(feats_path) or feats_in is None) and self.atlas is not None:
            pf = torch.from_numpy(np.array(list(obj_points.numpy()), dtype=np.float32)).float().cuda().transpose(1,2)
            with torch.no_grad():
                feats = self.atlas.encoder(pf).detach().cpu().numpy()

            feats_out = {}
            feats_out['feats'] = feats
            feats_out['instance_order'] = instances_order
            feats_in = list(feats)

            assert self.path2atlas is not None
            path = os.path.join(feats_path)
            if self.recompute_feats:
                path = path[:-3]

            pickle.dump(feats_out, open(path, 'wb'))

        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = cat
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = tight_boxes
        output['encoder']['words'] = words
        if self.use_points:
            output['encoder']['points'] = list(obj_points.numpy())

        if self.with_BERT:
            output['encoder']['text_feats'] = bert_feats_ins
            bert_feats_rel_new = []
            if bert_feats_rel != None:
                for word in words:
                    bert_feats_rel_new.append(bert_feats_rel[word])
                output['encoder']['rel_feats'] = bert_feats_rel_new

        elif self.with_CLIP:
            output['encoder']['text_feats'] = clip_feats_ins
            clip_feats_rel_new = []
            if clip_feats_rel != None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                output['encoder']['rel_feats'] = clip_feats_rel_new

        if self.with_feats:
            output['encoder']['feats'] = feats_in

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval:
                if self.with_changes:
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][
                        np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'])
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        output['manipulate']['type'] = 'none'
            else:
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        return -1
        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64)) # this is changed
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.use_points:
            output['encoder']['points'] = torch.from_numpy(np.array(output['encoder']['points'], dtype=np.float32))
        if self.with_feats:
            output['encoder']['feats'] = torch.from_numpy(np.array(output['encoder']['feats'], dtype=np.float32))
        if self.with_BERT or self.with_CLIP:
            output['encoder']['text_feats'] = torch.from_numpy(np.array(output['encoder']['text_feats'], dtype=np.float32)) # this is changed
            output['encoder']['rel_feats'] = torch.from_numpy(np.array(output['encoder']['rel_feats'], dtype=np.float32))


        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))
        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64)) # this is changed
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.use_points:
            output['decoder']['points'] = torch.from_numpy(np.array(output['decoder']['points'], dtype=np.float32))
        if self.with_feats:
            output['decoder']['feats'] = torch.from_numpy(np.array(output['decoder']['feats'], dtype=np.float32))
        if self.with_BERT or self.with_CLIP:
            output['decoder']['text_feats'] = torch.from_numpy(np.array(output['decoder']['text_feats'], dtype=np.float32))
            output['decoder']['rel_feats'] = torch.from_numpy(np.array(output['decoder']['rel_feats'], dtype=np.float32)) # this is changed

        output['scan_id'] = scan_id_no_split
        output['split_id'] = scan_id.split('_')[1]
        output['instance_id'] = instances_order

        ## encoder:
        # objs: class id;
        # triples: [mask_sub, relation, mask_obj] mask is the instance id in the selected split started from 1 to number_objs rather than the one in the whole instances.
        # boxes: param7
        return output


    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        if not self.use_rio27:
            excluded = [27, 58, 155]
        else:
            excluded = [1, 2, 15]
        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        graph['objs'].pop(node_id)
        if self.use_points:
            graph['points'].pop(node_id)
        if self.with_feats:
            graph['feats'].pop(node_id)
        if self.with_BERT or self.with_CLIP:
            graph['text_feats'].pop(node_id)

        graph['boxes'].pop(node_id)

        to_rm = []
        for i,x in reversed(list(enumerate(graph['triples']))):
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)
                if self.with_BERT or self.with_CLIP:
                    graph['rel_feats'].pop(i)
                    graph['words'].pop(i)

        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        return node_id

    def modify_relship(self, graph, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''26 hanging in' '25 lying in' '24 cover' '23 build in' '22 standing in' '21 belonging to'
         '20 part of' '19 leaning against' '18 connected to' '17 hanging on' '16 lying on' '15 standing on'
         '14 attached to' '13 same as' '12 same symmetry as' '11 lower than' '10 higher than' '9 smaller than'
         '8 bigger than' '7 inside' '6 close by' '5 behind' '4 front' '3 right' '2 left' '1 supported by'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [2, 3, 4, 5, 8, 9, 10, 11]
        rel_dict = {2: 'left', 3: 'right', 4: 'front', 5: 'behind', 8: 'bigger than', 9: 'smaller than', 10: 'higher than', 11: 'lower than'}
        inside_rel = [7, 22, 23, 24, 25, 26]
        # arm chair, basked, bathtub, bidet, bookshelf, box, chair,  commode, cupboard, trash bin, kettle, shelf,
        # wardrobe, sink
        inside_obj = [1, 7, 9, 13, 20, 22, 28, 37, 43, 67, 74, 119, 129, 156]

        hanging_rel = [14, 17]
        # backpack, bag, blinds, clock, curtain
        hanging_sub_allowed = [2, 3, 16, 30, 44]
        # wall
        hanging_obj_allowed = [155]

        did_change = False
        trials = 0
        excluded = [27]
        eval_excluded = [27, 58, 155]

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if pred == 0:
                continue
            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            # sub_cl, obj_cl = graph['objs'][sub], graph['objs'][obj]
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                new_pred = interpretable_rels[np.random.randint(1, len(interpretable_rels))]
            else:
                new_pred = np.random.randint(1, 27)

            graph['words'][idx] = graph['words'][idx].replace(self.relationships[graph['triples'][idx][1]],self.relationships[new_pred])
            if self.with_BERT:
                rel = self.cond_model(graph['words'][idx]).detach().cpu().numpy()
                graph['rel_feats'][idx] = np.squeeze(rel)
            elif self.with_CLIP:
                text_rel = clip.tokenize(graph['words'][idx]).to('cuda')
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                graph['rel_feats'][idx] = np.squeeze(rel)
            graph['triples'][idx][1] = new_pred

            did_change = True
        return idx, (sub, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)


def collate_fn_vaegan(batch, use_points=False):
    """
    Collate function to be used when wrapping a RIODatasetSceneGraph in a
    DataLoader. Returns a dictionary
    """

    out = {}

    out['scene_points'] = []
    out['scan_id'] = []
    out['instance_id'] = []
    out['split_id'] = []

    out['missing_nodes'] = []
    out['missing_nodes_decoder'] = []
    out['manipulated_nodes'] = []
    global_node_id = 0
    global_dec_id = 0

    for i in range(len(batch)):
        if batch[i] == -1:
            return -1
        # notice only works with single batches
        out['scan_id'].append(batch[i]['scan_id'])
        out['instance_id'].append(batch[i]['instance_id'])
        out['split_id'].append(batch[i]['split_id'])

        if batch[i]['manipulate']['type'] == 'addition':
            out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added'])
            out['missing_nodes_decoder'].append(global_dec_id + batch[i]['manipulate']['added'])
        elif batch[i]['manipulate']['type'] == 'relationship':
            rel, (sub, obj) = batch[i]['manipulate']['relship']
            out['manipulated_nodes'].append(global_dec_id + sub)
            out['manipulated_nodes'].append(global_dec_id + obj)

        if 'scene' in batch[i]:
            out['scene_points'].append(batch[i]['scene'])

        global_node_id += len(batch[i]['encoder']['objs'])
        global_dec_id += len(batch[i]['decoder']['objs'])

    for key in ['encoder', 'decoder']:
        all_objs, all_boxes, all_triples = [], [], []
        all_obj_to_scene, all_triple_to_scene = [], []
        all_points = []
        all_feats = []
        all_text_feats = []
        all_rel_feats = []

        obj_offset = 0

        for i in range(len(batch)):
            if batch[i] == -1:
                print('this should not happen')
                continue
            (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

            if 'points' in batch[i][key]:
                all_points.append(batch[i][key]['points'])
            if 'feats' in batch[i][key]:
                all_feats.append(batch[i][key]['feats'])
            if 'text_feats' in batch[i][key]:
                all_text_feats.append(batch[i][key]['text_feats'])
            if 'rel_feats' in batch[i][key]:
                all_rel_feats.append(batch[i][key]['rel_feats'])

            num_objs, num_triples = objs.size(0), triples.size(0)

            all_objs.append(objs)
            all_boxes.append(boxes)

            if triples.dim() > 1:
                triples = triples.clone()
                triples[:, 0] += obj_offset
                triples[:, 2] += obj_offset

                all_triples.append(triples)
                all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

            all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

            obj_offset += num_objs

        all_objs = torch.cat(all_objs)
        all_boxes = torch.cat(all_boxes)

        all_obj_to_scene = torch.cat(all_obj_to_scene)

        if len(all_triples) > 0:
            all_triples = torch.cat(all_triples)
            all_triple_to_scene = torch.cat(all_triple_to_scene)
        else:
            return -1

        outputs = {'objs': all_objs,
                   'tripltes': all_triples,
                   'boxes': all_boxes,
                   'obj_to_scene': all_obj_to_scene,
                   'triple_to_scene': all_triple_to_scene}

        if len(all_points) > 0:
            all_points = torch.cat(all_points)
            outputs['points'] = all_points

        if len(all_feats) > 0:
            all_feats = torch.cat(all_feats)
            outputs['feats'] = all_feats
        if len(all_text_feats) > 0:
            all_text_feats = torch.cat(all_text_feats)
            outputs['text_feats'] = all_text_feats
        if len(all_rel_feats) > 0:
            all_rel_feats = torch.cat(all_rel_feats)
            outputs['rel_feats'] = all_rel_feats
        out[key] = outputs

    return out


def collate_fn_vaegan_points(batch):
    """ Wrapper of the function collate_fn_vaegan to make it also return points
    """
    return collate_fn_vaegan(batch, use_points=True)

if __name__ == "__main__":
    from model.atlasnet import AE_AtlasNet

    saved_atlasnet_model = torch.load("../experiments/atlasnet/model_70.pth")
    point_ae = AE_AtlasNet(num_points=1024, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_atlasnet_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()
    point_ae.eval()
    dataset = RIODatasetSceneGraph(
        root="../GT",
        root_3rscan="/media/ymxlzgy/DATA/3RScan",
        label_file='labels.instances.align.annotated.ply',
        npoints=1024,
        path2atlas="./experiments/atlasnet/model_70.pth",
        split='train_scans',
        shuffle_objs=True,
        use_points=False,
        use_scene_rels=True,
        with_changes=True,
        vae_baseline=False,
        with_feats=True,
        large=True,
        atlas=point_ae,
        seed=False,
        use_splits=True,
        use_rio27=False,
        use_canonical=True,
        crop_floor=False,
        center_scene_to_floor=False,
        recompute_feats=False)
    a = dataset[10]
    print(a)