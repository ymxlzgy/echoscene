import numpy as np
import os
import json

import torch

import dataset.util as helpers
from helpers.util import get_rotation

# obj_category -> list of objects (scan_id, instance_id, boxes)
retrieval_cache = dict()


def rio_retrieve(objs, bboxes, vocab, cat2objs, root_3rscan, skip_scene_node=True, return_retrieval_id=False):
    input_boxes = [box.cpu().detach().numpy() for box in bboxes]
    O = len(objs)
    shapes = []
    if return_retrieval_id:
        retrieval_ids = []

    num_nodes = O - 1 if skip_scene_node else O
    for i in range(num_nodes):
        obj_idx = objs[i]
        obj_type = vocab['object_idx_to_name'][obj_idx].split('\n')[0]
        box = input_boxes[i]
        deltax = box[0]
        deltay = box[1]
        deltaz = box[2]
        ratio = np.array([deltay / deltax, deltaz / deltax])
        try:
            obj_data = cat2objs[obj_type]
        except:
            obj_data = cat2objs['object']
        ratio_data = []
        for obj in obj_data:
            obj_size = np.array(obj["param7"][:3])
            obj_ratio = np.array([obj_size[1] / obj_size[0], obj_size[2] / obj_size[0]])
            ratio_data.append(np.sum(np.abs(obj_ratio - ratio)))

        obj_choose = np.argmin(ratio_data)
        obj_id = obj_data[obj_choose]["id"]
        scan = obj_data[obj_choose]["scan"]
        box_choosen = obj_data[obj_choose]["param7"]
        dir_choosen = obj_data[obj_choose]["direction"]
        shape = load_retrieved_obj(root_3rscan, scan, obj_id, box_choosen, dir_choosen)
        shapes.append(shape)

        retrieval_ids.append(str(scan) + '_' + str(obj_id))

    if return_retrieval_id:
      return (torch.from_numpy(np.asarray(shapes).astype('float32')),
              np.asarray(retrieval_ids, dtype=np.str))

    return torch.from_numpy(np.asarray(shapes).astype('float32'))


def load_retrieved_obj(root_3rscan, scan_id, id, box, direction):
    global retrieval_cache
    retrieval_id = "{}_{}".format(scan_id, id)
    if retrieval_id in retrieval_cache:
        return retrieval_cache[retrieval_id]

    label_file = 'binary_labels.instances.align.annotated.ply'
    file = os.path.join(root_3rscan, scan_id, label_file)
    if not os.path.exists(file):
        file = os.path.join(root_3rscan, scan_id, 'labels.instances.align.annotated.ply')
    points_all, instances_all, _, _ = helpers.read_ply(file)
    points = points_all[instances_all==id]
    bbox = box.copy()
    if direction in [1, 2, 3, 4]:
      bbox[6] = bbox[6] + (direction - 1) * 90
    if direction in [2, 4]:
      temp = bbox[0]
      bbox[0] = bbox[1]
      bbox[1] = temp
    points = points - bbox[3:6]
    points = (get_rotation(-bbox[6]) @ points.T).T
    points = points / np.linalg.norm(bbox[:3])
    choice = np.random.choice(len(points), 1024, replace=True)
    points = points[choice, :]

    retrieval_cache[retrieval_id] = points
    return points


def read_box_json(json_file, box_json_file):
    cat2objs = {}
    with open(box_json_file, "r") as read_file:
        box_data = json.load(read_file)
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for scan in data['scans']:
            for k, v in scan["objects"].items():
                if v not in cat2objs.keys():
                    cat2objs[v] = []
                try:
                    if 'direction' not in box_data[scan["scan"]][k].keys():
                      direction = 0
                    else:
                      direction = box_data[scan["scan"]][k]["direction"]
                    obj_ins = {'scan': scan['scan'],
                               'id': int(k),
                               'param7': box_data[scan["scan"]][k]["param7"],
                               'direction': direction}
                    cat2objs[v].append(obj_ins)
                except:
                    # probably not saved because there were 0 points!
                    continue
    return cat2objs
