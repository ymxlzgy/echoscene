from __future__ import print_function

import time

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from model.SGDiff import SGDiff
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag, preprocess_angle2sincos, batch_torch_destandardize_box_params, descale_box_params, postprocess_sincos2arctan, sample_points
from helpers.metrics_3dfront import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_scene import render, render_full, render_box
from omegaconf import OmegaConf
# import extension.dist_chamfer as ext
# chamfer = ext.chamferDist()
import json

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=3, help='for diversity')

parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/FRONT", help="dataset path")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, full')
parser.add_argument('--evaluate_diversity', type=bool_flag, default=False, help='Computes diversity based on multiple predictions')
parser.add_argument('--gen_shape', default=False, type=bool_flag, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool_flag)
parser.add_argument('--export_3d', default=False, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')

args = parser.parse_args()

room_type = ['all', 'bedroom', 'livingroom', 'diningroom', 'library']


def reseed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    random.seed(num)

def normalize(vertices, scale=1):
    xmin, xmax = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
    ymin, ymax = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
    zmin, zmax = np.amin(vertices[:, 2]), np.amax(vertices[:, 2])

    vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
    vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
    vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

    scalars = np.max(vertices, axis=0)
    scale = scale

    vertices = vertices / scalars * scale
    return vertices

def validate_constrains_loop_w_changes(modelArgs, testdataset, model, normalized_file=None, with_diversity=True, bin_angles=False, num_samples=3, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_changes = torch.utils.data.DataLoader(
        testdataset,
        batch_size=1,
        collate_fn=testdataset.collate_fn,
        shuffle=False,
        num_workers=0)

    if with_diversity and num_samples < 2:
        raise ValueError('Diversity requires at least two runs (i.e. num_samples > 1).')

    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy_unchanged[k] = []
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []
    all_diversity_chamfer = []
    bbox_file = "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_trainval.json" if datasize == 'large' else "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_all_small.json"
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
        box_data['chair'].update(box_data['stool'])

    for i, data in enumerate(test_dataloader_changes, 0):
        try:
            enc_objs, enc_triples, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                              data['encoder']['tripltes'], \
                                                                                              data['encoder']['obj_to_scene'], \
                                                                                              data['encoder']['triple_to_scene']

            dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                              data['decoder']['tripltes'], \
                                                                                              data['decoder']['boxes'], \
                                                                                              data['decoder']['obj_to_scene'], \
                                                                                              data['decoder']['triple_to_scene']
            dec_sdfs = None
            if modelArgs['with_SDF']:
                dec_sdfs = data['decoder']['sdfs']

            missing_nodes = data['missing_nodes']
            manipulated_subs = data['manipulated_subs']
            manipulated_objs = data['manipulated_objs']
            manipulated_preds = data['manipulated_preds']

        except Exception as e:
            print("Exception: skipping scene", e)
            continue

        enc_objs, enc_triples = enc_objs.cuda(), enc_triples.cuda()
        dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
        encoded_enc_rel_feat, encoded_enc_text_feat, encoded_dec_text_feat, encoded_dec_rel_feat = None, None, None, None
        if modelArgs['with_CLIP']:
            encoded_enc_text_feat, encoded_enc_rel_feat = data['encoder']['text_feats'].cuda(), data['encoder']['rel_feats'].cuda()
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()

        all_pred_boxes = []
        all_pred_angles = []

        with torch.no_grad():

            if len(manipulated_subs) and len(manipulated_objs):
                manipulated_nodes = manipulated_subs + manipulated_objs
                keep, data_dict = model.sample_boxes_and_shape_with_changes(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                            encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                            dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                            manipulated_nodes, gen_shape=gen_shape)
            else:
                keep, data_dict = model.sample_boxes_and_shape_with_additions(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                              encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                              dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                              missing_nodes, gen_shape=gen_shape)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'], data_dict['translations']), dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')

            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file) # mean, std
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max

            # if with_diversity:
            #     assert model.type_ == 'cs++' or model.type_ == 'cs++_l'
            #     print('calculating diversity...')
            #     # Run multiple times to obtain diversity
            #     # Only when a node was added or manipulated we run the diversity computation
            #     if len(missing_nodes) > 0 or len(manipulated_nodes) > 0:
            #         # Diversity results for this dataset sample
            #         boxes_diversity_sample, shapes_sample, angle_diversity_sample, diversity_retrieval_ids_sample = [], [], [], []
            #
            #         for sample in range(num_samples):
            #             # Generated changes
            #             diversity_angles = None
            #             if args.manipulate:
            #                 #TODO modify v1
            #                 diversity_boxes, diversity_points, diversity_keep = model.decoder_with_changes_boxes_and_shape(
            #                     z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes,
            #                     box_data=box_data, gen_shape=gen_shape)
            #             else:
            #                 diversity_boxes, diversity_angles, diversity_points, diversity_keep = model.decoder_with_additions_boxes_and_shape(
            #                     z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes)
            #
            #             if model.type_ == 'v2_full':
            #                 from model.diff_utils.util_3d import sdf_to_mesh
            #                 diversity_points = sdf_to_mesh(diversity_points)
            #                 diversity_points = diversity_points.verts_list()
            #                 diversity_points = sample_points(diversity_points, 5000) #TODO adjust number
            #             elif model.type_ == 'v1_full':
            #                 from pytorch3d.structures import Meshes
            #                 verts_list = []
            #                 faces_list = []
            #                 for mesh in diversity_points:
            #                     verts = torch.tensor(mesh.vertices, dtype=torch.float32)
            #                     faces = torch.tensor(mesh.faces, dtype=torch.int64)
            #
            #                     verts_list.append(verts)
            #                     faces_list.append(faces)
            #                 diversity_points = Meshes(verts=verts_list, faces=faces_list)
            #                 diversity_points = diversity_points.verts_list()
            #                 diversity_points = sample_points(diversity_points, 5000) #TODO adjust number
            #
            #             if with_angles and diversity_angles is None:
            #                 diversity_boxes, diversity_angles = diversity_boxes
            #
            #             # Computing shape diversity on canonical and normalized shapes
            #             normalized_points = []
            #             filtered_diversity_retrieval_ids = []
            #             for ins_id, obj_id in enumerate(dec_objs):
            #                 if obj_id.item() != 0 and testdataloader.dataset.classes_r[obj_id.item()] != 'floor':
            #                     # We only care for manipulated nodes
            #                     if diversity_keep[ins_id, 0] == 1:
            #                         continue
            #                     points = diversity_points[ins_id]
            #                     if type(points) is torch.Tensor:
            #                         points = points.cpu().numpy()
            #                     if points is None:
            #                         continue
            #                     # Normalizing shapes
            #                     points = torch.from_numpy(normalize(points))
            #                     if torch.cuda.is_available():
            #                         points = points.cuda()
            #                     normalized_points.append(points)
            #
            #             # We use keep to filter changed nodes
            #             boxes_diversity_sample.append(diversity_boxes[diversity_keep[:, 0] == 0])
            #
            #             if with_angles:
            #                 # We use keep to filter changed nodes
            #                 angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles[diversity_keep[:, 0] == 0].cpu().numpy(), 1), 1) / 24. * 360.) # TODO change this maybe
            #
            #             if len(normalized_points) > 0:
            #                 shapes_sample.append(torch.stack(normalized_points)) # keep has already been applied for points
            #
            #         # Compute standard deviation for box for this sample
            #         if len(boxes_diversity_sample) > 0:
            #             boxes_diversity_sample = torch.stack(boxes_diversity_sample, 1)
            #             bs = boxes_diversity_sample.shape[0]
            #             boxes_diversity_sample = batch_torch_denormalize_box_params(boxes_diversity_sample.reshape([-1, 6]),file=normalized_file).reshape([bs, -1, 6])
            #             all_diversity_boxes += torch.std(boxes_diversity_sample, dim=1).cpu().numpy().tolist()
            #
            #         # Compute standard deviation for angle for this sample
            #         if len(angle_diversity_sample) > 0:
            #             angle_diversity_sample = np.stack(angle_diversity_sample, 1)
            #             all_diversity_angles += [estimate_angular_std(d[:,0]) for d in angle_diversity_sample]
            #
            #         # Compute chamfer distances for shapes for this sample
            #         if len(shapes_sample) > 0:
            #             if len(diversity_retrieval_ids_sample) > 0:
            #                 diversity_retrieval_ids_sample = np.stack(diversity_retrieval_ids_sample, 1)
            #             shapes_sample = torch.stack(shapes_sample, 1)
            #
            #             for shapes_id in range(len(shapes_sample)):
            #                 # Taking a single predicted shape
            #                 shapes = shapes_sample[shapes_id]
            #                 if len(diversity_retrieval_ids_sample) > 0:
            #                     # To avoid that retrieval the object ids like 0,1,0,1,0 gives high error
            #                     # We sort them to measure how often different objects are retrieved 0,0,0,1,1
            #                     diversity_retrieval_ids = diversity_retrieval_ids_sample[shapes_id]
            #                     sorted_idx = diversity_retrieval_ids.argsort()
            #                     shapes = shapes[sorted_idx]
            #                 sequence_diversity = []
            #                 # Iterating through its multiple runs
            #                 for shape_sequence_id in range(len(shapes) - 1):
            #                     # Compute chamfer with the next shape in its sequences
            #                     dist1, dist2 = chamfer(shapes[shape_sequence_id:shape_sequence_id + 1].float(),
            #                                            shapes[shape_sequence_id + 1:shape_sequence_id + 2].float())
            #                     chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
            #                     # Save the distance
            #                     sequence_diversity += [chamfer_dist.cpu().numpy().tolist()]
            #                 all_diversity_chamfer.append(np.mean(sequence_diversity))
        bp_box, bp_angle = [], []
        for i in range(len(keep)):
            if keep[i] == 0:
                # manipulated / added node
                bp_box.append(boxes_pred_den[i:i+1].cpu().detach())
                bp_angle.append(angles_pred[i:i+1].cpu().detach())
            else:
                # original node
                dec_tight_boxes[i:i+1,:6] = descale_box_params(dec_tight_boxes[i:i+1,:6], file=normalized_file)  # min, max
                bp_box.append(dec_tight_boxes[i:i+1,:6].cpu().detach())
                angle = dec_tight_boxes[i:i+1, 6:7] / np.pi * 180
                bp_angle.append(angle.cpu().detach())

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())

        # compute relationship constraints accuracy through simple geometric rules
        # TODO boxes_pred_den with angle
        accuracy = validate_constrains_changes(dec_triples, boxes_pred_den, angles_pred, keep, model.vocab, accuracy)
        accuracy_in_orig_graph = validate_constrains_changes(dec_triples, torch.stack(bp_box, dim=0).squeeze(1), torch.stack(bp_angle, dim=0).squeeze(1), keep, model.vocab, accuracy_in_orig_graph)
        accuracy_unchanged = validate_constrains(dec_triples, boxes_pred_den, angles_pred, keep, model.vocab, accuracy_unchanged)

    # if with_diversity:
    #     print("DIVERSITY:")
    #     print("\tShape (Avg. Chamfer Distance) = %f" % (np.mean(all_diversity_chamfer)))
    #     print("\tBox (Std. metric size and location) = %f, %f" % (
    #         np.mean(np.mean(all_diversity_boxes, axis=0)[:3]),
    #         np.mean(np.mean(all_diversity_boxes, axis=0)[3:])))
    #     print("\tAngle (Std.) %s = %f" % (k, np.mean(all_diversity_angles)))

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{testdataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "changed nodes"), (accuracy_unchanged, 'unchanged nodes'),
                         (accuracy_in_orig_graph, 'changed nodes placed in original graph')]:
            # NOTE 'changed nodes placed in original graph' are the results reported in the paper!
            # The unchanged nodes are kept from the original scene, and the accuracy in the new nodes is computed with
            # respect to these original nodes
            # Calculating means
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print('{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(typ, lr_mean,
                                        fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))


def validate_constrains_loop(modelArgs, test_dataset, model, epoch=None, normalized_file=None, with_diversity=True, export_3d=False, cat2objs=None, datasize='large', num_samples=3, gen_shape=False):

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=0)

    point_classes_idx = test_dataset.point_classes_idx
    vocab = test_dataset.vocab
    if with_diversity and num_samples < 2:
        raise ValueError('Diversity requires at least two runs (i.e. num_samples > 1).')

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []
    all_diversity_chamfer = []

    bed_diversity_chamfer = []
    night_diversity_chamfer = []
    wardrobe_diversity_chamfer = []
    chair_diversity_chamfer = []
    table_diversity_chamfer = []
    cabinet_diversity_chamfer = []
    sofa_diversity_chamfer = []
    lamp_diversity_chamfer = []
    shelf_diversity_chamfer = []
    tvstand_diversity_chamfer = []


    all_pred_shapes_exp = {} # for export
    all_pred_boxes_exp = {}
    bbox_file = "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_trainval.json" if datasize == 'large' else "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_all_small.json"

    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
        box_data['chair'].update(box_data['stool'])

    for i, data in enumerate(test_dataloader_no_changes, 0):
        # print(data['scan_id'])
        # if data['scan_id'][0] != 'LivingDiningRoom-13945': # problematic room
        #     continue
        # if data['scan_id'][0].split('-')[0] not in ['DiningRoom', "LivingDiningRoom"]:
        #     continue
        # if data['scan_id'][0] not in ['MasterBedroom-58086','MasterBedroom-109561','Bedroom-11202', 'DiningRoom-2432', 'DiningRoom-451', 'DiningRoom-20718', 'LivingRoom-2050', 'LivingRoom-3540', 'LivingRoom-29294']:
            # continue
        if data['scan_id'][0] != 'SecondBedroom-19119': # problematic room
            continue

        try:
            dec_objs, dec_triples = data['decoder']['objs'], data['decoder']['tripltes']
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
        except Exception as e:
            print(e)
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat = None, None
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()
        dec_sdfs = None
        if modelArgs['with_SDF']:
            dec_sdfs = data['decoder']['sdfs']

        all_pred_boxes = []
        all_pred_angles = []

        with torch.no_grad():

            data_dict = model.sample_box_and_shape(dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=gen_shape)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')
            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file) # mean, std
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max


        if args.visualize:
            colors = None
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            print("rendering", [classes[i].strip('\n') for i in dec_objs])
            if model.type_ == 'cs++_l':
                render_box(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, store_img=True, render_boxes=False, visual=True, demo=False, without_lamp=True, store_path=modelArgs['store_path'])
            elif model.type_ == 'cs++':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=True, store_path=modelArgs['store_path'])
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())
        if with_diversity:
            print('calculating diversity...')
            assert model.type_ == 'cs++_l' or model.type_ == 'cs++'
            # Run multiple times to obtain diversities
            # Diversity results for this dataset sample
            boxes_diversity_sample, shapes_sample, angle_diversity_sample, diversity_retrieval_ids_sample = [], [], [], []
            for sample in range(num_samples):
                # reseed(int(time.time()))
                diversity_boxes, diversity_points = model.sample_box_and_shape(point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                               attributes=None, gen_shape=True)
                if model.type_ == 'cs++':
                    from model.diff_utils.util_3d import sdf_to_mesh
                    diversity_points = sdf_to_mesh(diversity_points, render_all=True)
                    diversity_points = diversity_points.verts_list()
                    diversity_points = sample_points(diversity_points, 5000) #TODO adjust number

                diversity_boxes, diversity_angles = diversity_boxes

                # Computing shape diversity on canonical and normalized shapes
                normalized_points = []
                filtered_diversity_retrieval_ids = []
                bed_ins_id_list, night_ins_id_list, wardrobe_ins_id_list, chair_ins_id_list, table_ins_id_list, cabinet_ins_id_list, lamp_ins_id_list, sofa_ins_id_list, shelf_ins_id_list, tvstand_ins_id_list = [], [], [], [], [], [], [], [], [], []
                for ins_id, obj_id in enumerate(dec_objs):
                    if testdataloader.dataset.classes['bed'] == obj_id.item():
                        bed_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['nightstand'] == obj_id.item():
                        night_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['wardrobe'] == obj_id.item():
                        wardrobe_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['chair'] == obj_id.item():
                        chair_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['table'] == obj_id.item():
                        table_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['cabinet'] == obj_id.item():
                        cabinet_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['lamp'] == obj_id.item():
                        lamp_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['sofa'] == obj_id.item():
                        sofa_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['shelf'] == obj_id.item():
                        shelf_ins_id_list.append(ins_id)
                    if testdataloader.dataset.classes['tv_stand'] == obj_id.item():
                        tvstand_ins_id_list.append(ins_id)
                    if obj_id.item() != 0 and testdataloader.dataset.classes_r[obj_id.item()] != 'floor':
                        points = diversity_points[ins_id]
                        if type(points) is torch.Tensor:
                            points = points.cpu().numpy()
                        if points is None:
                            continue
                        # Normalizing shapes
                        points = torch.from_numpy(normalize(points))
                        if torch.cuda.is_available():
                            points = points.cuda()
                        normalized_points.append(points)
                        # if model.type_ == 'sln':
                        #     filtered_diversity_retrieval_ids.append(diversity_retrieval_ids[ins_id])

                # We use keep to filter changed nodes
                boxes_diversity_sample.append(diversity_boxes)

                # We use keep to filter changed nodes
                if modelArgs['bin_angle']:
                    angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles.cpu().numpy(), 1), 1) / 24. * 360.)
                else:
                    angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles.cpu().numpy(), 1), 1) / np.pi * 180.)  # TODO change this maybe

                if len(normalized_points) > 0:
                    shapes_sample.append(torch.stack(normalized_points)) # keep has already been aplied for points
                    # if model.type_ == 'sln':
                    #     diversity_retrieval_ids_sample.append(np.stack(filtered_diversity_retrieval_ids))


            # Compute standard deviation for box for this sample
            if len(boxes_diversity_sample) > 0:
                boxes_diversity_sample = torch.stack(boxes_diversity_sample, 1)
                bs = boxes_diversity_sample.shape[0]
                boxes_diversity_sample = descale_box_params(boxes_diversity_sample.reshape([-1, 6]),file=normalized_file).reshape([bs, -1, 6])
                all_diversity_boxes += torch.std(boxes_diversity_sample, dim=1).cpu().numpy().tolist()

            # Compute standard deviation for angle for this sample
            if len(angle_diversity_sample) > 0:
                angle_diversity_sample = np.stack(angle_diversity_sample, 1)
                all_diversity_angles += [estimate_angular_std(d[:,0]) for d in angle_diversity_sample]

            # Compute chamfer distances for shapes for this sample
            if len(shapes_sample) > 0:
                shapes_sample = torch.stack(shapes_sample, 1)

                for shapes_id in range(len(shapes_sample)):
                    # Taking a single predicted shape
                    shapes = shapes_sample[shapes_id]
                    if len(diversity_retrieval_ids_sample) > 0:
                        # To avoid that retrieval the object ids like 0,1,0,1,0 gives high error
                        # We sort them to measure how often different objects are retrieved 0,0,0,1,1
                        diversity_retrieval_ids = diversity_retrieval_ids_sample[shapes_id]
                        sorted_idx = diversity_retrieval_ids.argsort()
                        shapes = shapes[sorted_idx]
                    sequence_diversity = []
                    # Iterating through its multiple runs
                    for shape_sequence_id in range(len(shapes) - 1):
                        # Compute chamfer with the next shape in its sequences
                        dist1, dist2 = chamfer(shapes[shape_sequence_id:shape_sequence_id + 1].float(),
                                               shapes[shape_sequence_id + 1:shape_sequence_id + 2].float())
                        chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
                        # Save the distance
                        sequence_diversity += [chamfer_dist.cpu().numpy().tolist()]

                    if len(sequence_diversity) > 0:  # check if sequence has shapes
                        all_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in bed_ins_id_list:
                            bed_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in night_ins_id_list:
                            night_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in wardrobe_ins_id_list:
                            wardrobe_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in chair_ins_id_list:
                            chair_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in table_ins_id_list:
                            table_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in cabinet_ins_id_list:
                            cabinet_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in lamp_ins_id_list:
                            lamp_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in sofa_ins_id_list:
                            sofa_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in shelf_ins_id_list:
                            shelf_diversity_chamfer.append(np.mean(sequence_diversity))
                        if shapes_id in tvstand_ins_id_list:
                            tvstand_diversity_chamfer.append(np.mean(sequence_diversity))

        # compute constraints accuracy through simple geometric rules
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_pred, None, model.vocab, accuracy)

    if export_3d:
        # export box and shape predictions for future evaluation
        result_path = os.path.join(modelArgs['store_path'], 'results')
        # Create a new directory for results
        os.makedirs(result_path, exist_ok=True)
        shape_filename = os.path.join(result_path, 'shapes_' + ('large' if datasize else 'small') + '.json')
        box_filename = os.path.join(result_path, 'boxes_' + ('large' if datasize else 'small') + '.json')
        json.dump(all_pred_boxes_exp, open(box_filename, 'w')) # 'dis_nomani_boxes_large.json'
        json.dump(all_pred_shapes_exp, open(shape_filename, 'w'))

    if with_diversity:
        print("DIVERSITY:")
        print("{} objects: ".format(str(len(all_diversity_chamfer))))
        print("\tShape (Avg. Chamfer Distance) = %f" % (np.mean(all_diversity_chamfer)))
        print("{} beds: ".format(len(bed_diversity_chamfer)))
        print("\tBed Shape (Avg. Chamfer Distance) = %f" % (np.mean(bed_diversity_chamfer)))
        print("{} nightstands: ".format(len(night_diversity_chamfer)))
        print("\tNightstand Shape (Avg. Chamfer Distance) = %f" % (np.mean(night_diversity_chamfer)))
        print("{} wardrobes: ".format(len(wardrobe_diversity_chamfer)))
        print("\tWardrobe Shape (Avg. Chamfer Distance) = %f" % (np.mean(wardrobe_diversity_chamfer)))
        print("{} chairs: ".format(len(chair_diversity_chamfer)))
        print("\tChair Shape (Avg. Chamfer Distance) = %f" % (np.mean(chair_diversity_chamfer)))
        print("{} tables: ".format(len(table_diversity_chamfer)))
        print("\tTable Shape (Avg. Chamfer Distance) = %f" % (np.mean(table_diversity_chamfer)))
        print("{} cabinet: ".format(len(cabinet_diversity_chamfer)))
        print("\tCabinet Shape (Avg. Chamfer Distance) = %f" % (np.mean(cabinet_diversity_chamfer)))
        print("{} lamps: ".format(len(lamp_diversity_chamfer)))
        print("\tLamp Shape (Avg. Chamfer Distance) = %f" % (np.mean(lamp_diversity_chamfer)))
        print("{} shelfs: ".format(len(shelf_diversity_chamfer)))
        print("\tShelf Shape (Avg. Chamfer Distance) = %f" % (np.mean(shelf_diversity_chamfer)))
        print("{} sofas: ".format(len(sofa_diversity_chamfer)))
        print("\tSofa Shape (Avg. Chamfer Distance) = %f" % (np.mean(sofa_diversity_chamfer)))
        print("{} tvstands: ".format(len(tvstand_diversity_chamfer)))
        print("\tTV stand Shape (Avg. Chamfer Distance) = %f" % (np.mean(tvstand_diversity_chamfer)))
        print("\tBox (Std. metric size and location) = %f, %f" % (
            np.mean(np.mean(all_diversity_boxes, axis=0)[:3]),
            np.mean(np.mean(all_diversity_boxes, axis=0)[3:])))
        print("\tAngle (Std.) %s = %f" % (k, np.mean(all_diversity_angles)))

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{test_dataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "acc")]:
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(
                    typ, lr_mean,
                    fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))

def evaluate():
    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)
    normalized_file = os.path.join(args.dataset, 'centered_bounds_{}_trainval.txt').format(modelArgs['room_type'])
    test_dataset_rels_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='relationship',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type,
        recompute_clip=False)

    test_dataset_addition_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='addition',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=True,
        eval_type='none',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    modeltype_ = modelArgs['network_type']
    modelArgs['store_path'] = os.path.join(args.exp, "vis", args.epoch)
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    # args.visualize = False if args.gen_shape == False else args.visualize

    # instantiate the model
    diff_opt = modelArgs['diff_yaml']
    diff_cfg = OmegaConf.load(diff_opt)
    model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=False)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    cat2objs = None

    print('\nEditing Mode - Additions')
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_addition_changes, model, normalized_file=normalized_file, with_diversity=args.evaluate_diversity, bin_angles=modelArgs['bin_angle'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_rels_changes, model,  normalized_file=normalized_file, with_diversity=args.evaluate_diversity, bin_angles=modelArgs['bin_angle'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataset_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, with_diversity=args.evaluate_diversity, num_samples=args.num_samples,
                             export_3d=args.export_3d, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

if __name__ == "__main__":
    print(torch.__version__)
    evaluate()
