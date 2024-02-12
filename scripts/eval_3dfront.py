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
from helpers.util import bool_flag, batch_torch_destandardize_box_params, descale_box_params, postprocess_sincos2arctan, sample_points
from helpers.metrics_3dfront import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_scene import render, render_full, render_box
from omegaconf import OmegaConf
import extension.dist_chamfer as ext
chamfer = ext.chamferDist()
import json

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=3, help='for diversity')

parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/FRONT", help="dataset path")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
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

def evaluate():
    print(torch.__version__)

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
        recompute_clip=True)

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
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    collate_fn1 = test_dataset_rels_changes.collate_fn
    collate_fn2 = test_dataset_addition_changes.collate_fn
    collate_fn3 = test_dataset_no_changes.collate_fn

    test_dataloader_rels_changes = torch.utils.data.DataLoader(
        test_dataset_rels_changes,
        batch_size=1,
        collate_fn=collate_fn1,
        shuffle=False,
        num_workers=0)

    test_dataloader_add_changes = torch.utils.data.DataLoader(
        test_dataset_addition_changes,
        batch_size=1,
        collate_fn=collate_fn2,
        shuffle=False,
        num_workers=0)

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset_no_changes,
        batch_size=1,
        collate_fn=collate_fn3,
        shuffle=True,
        num_workers=0)

    modeltype_ = modelArgs['network_type']
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    args.visualize = False if args.gen_shape == False else args.visualize

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
    # validate_constrains_loop_w_changes(modelArgs, test_dataloader_add_changes, model, normalized_file=normalized_file, with_diversity=args.evaluate_diversity, bin_angles=modelArgs['bin_angle'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataloader_rels_changes, model,  normalized_file=normalized_file, with_diversity=args.evaluate_diversity, bin_angles=modelArgs['bin_angle'], num_samples=args.num_samples, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataloader_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, with_diversity=args.evaluate_diversity,
                             bin_angles=modelArgs['bin_angle'], num_samples=args.num_samples, vocab=test_dataset_no_changes.vocab,
                             point_classes_idx=test_dataset_no_changes.point_classes_idx,
                             export_3d=args.export_3d, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)


def validate_constrains_loop_w_changes(modelArgs, testdataloader, model, normalized_file=None, with_diversity=True, bin_angles=False, num_samples=3, cat2objs=None, datasize='large', gen_shape=False):
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
    bbox_file = "/home/ymxlzgy/code/graphto3d_v2/GT/3dfront/cat_jid_trainval.json" if datasize == 'large' else "/home/ymxlzgy/code/graphto3d_v2/GT/3dfront/cat_jid_trainval_small.json"
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
        if modelArgs['no_stool']:
            box_data['chair'].update(box_data['stool'])

    for i, data in enumerate(testdataloader, 0):
        try:
            enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                              data['encoder']['tripltes'], \
                                                                                              data['encoder']['boxes'], \
                                                                                              data['encoder']['obj_to_scene'], \
                                                                                              data['encoder']['triple_to_scene']
            if 'feats' in data['encoder']:
                encoded_enc_points = data['encoder']['feats']
                encoded_enc_points = encoded_enc_points.float().cuda()

            dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                              data['decoder']['tripltes'], \
                                                                                              data['decoder']['boxes'], \
                                                                                              data['decoder']['obj_to_scene'], \
                                                                                              data['decoder']['triple_to_scene']
            dec_sdfs = None
            if modelArgs['with_SDF']:
                dec_sdfs = data['decoder']['sdfs']

            missing_nodes = data['missing_nodes']
            manipulated_nodes = data['manipulated_nodes']

        except Exception as e:
            print("Exception: skipping scene", e)
            continue

        enc_objs, enc_triples, enc_tight_boxes = enc_objs.cuda(), enc_triples.cuda(), enc_tight_boxes.cuda()
        dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
        encoded_enc_rel_feat, encoded_enc_text_feat, encoded_dec_text_feat, encoded_dec_rel_feat = None, None, None, None
        if modelArgs['with_CLIP']:
            encoded_enc_text_feat, encoded_enc_rel_feat = data['encoder']['text_feats'].cuda(), data['encoder']['rel_feats'].cuda()
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()


        model = model.eval()

        all_pred_boxes = []

        enc_boxes = enc_tight_boxes[:, :6]
        enc_angles = enc_tight_boxes[:, 6].long() - 1
        enc_angles = torch.where(enc_angles > 0, enc_angles, torch.zeros_like(enc_angles))
        enc_angles = torch.where(enc_angles < 24, enc_angles, torch.zeros_like(enc_angles))

        attributes = None

        with torch.no_grad():
            if model.type_ == "v1_box" or model.type_ == "v2_box" or model.type_ == "v2_full":
                encoded_enc_points = None
            (z_box, _), (z_shape, _) = model.encode_box_and_shape(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, encoded_enc_points, enc_boxes, enc_angles, attributes)

            if args.manipulate:
                boxes_pred, shapes_pred, keep = model.decoder_with_changes_boxes_and_shape(z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat,
                                                                                           encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes,
                                                                                           box_data=box_data, gen_shape=gen_shape)
                if with_angles:
                    boxes_pred, angles_pred = boxes_pred
            else:
                boxes_pred, angles_pred, shapes_pred, keep = model.decoder_with_additions_boxes_and_shape(z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat,
                                                                                                          encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,  manipulated_nodes,
                                                                                                          gen_shape=gen_shape)
                if with_angles and angles_pred is None:
                    boxes_pred, angles_pred = boxes_pred

            if with_diversity:
                assert model.type_ == 'v2_full' or model.type_ == 'v1_full'
                print('calculating diversity...')
                # Run multiple times to obtain diversity
                # Only when a node was added or manipulated we run the diversity computation
                if len(missing_nodes) > 0 or len(manipulated_nodes) > 0:
                    # Diversity results for this dataset sample
                    boxes_diversity_sample, shapes_sample, angle_diversity_sample, diversity_retrieval_ids_sample = [], [], [], []

                    for sample in range(num_samples):
                        # Generated changes
                        diversity_angles = None
                        if args.manipulate:
                            #TODO modify v1
                            diversity_boxes, diversity_points, diversity_keep = model.decoder_with_changes_boxes_and_shape(
                                z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes,
                                box_data=box_data, gen_shape=gen_shape)
                        else:
                            diversity_boxes, diversity_angles, diversity_points, diversity_keep = model.decoder_with_additions_boxes_and_shape(
                                z_box, z_shape, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes)

                        if model.type_ == 'v2_full':
                            from model.diff_utils.util_3d import sdf_to_mesh
                            diversity_points = sdf_to_mesh(diversity_points)
                            diversity_points = diversity_points.verts_list()
                            diversity_points = sample_points(diversity_points, 5000) #TODO adjust number
                        elif model.type_ == 'v1_full':
                            from pytorch3d.structures import Meshes
                            verts_list = []
                            faces_list = []
                            for mesh in diversity_points:
                                verts = torch.tensor(mesh.vertices, dtype=torch.float32)
                                faces = torch.tensor(mesh.faces, dtype=torch.int64)

                                verts_list.append(verts)
                                faces_list.append(faces)
                            diversity_points = Meshes(verts=verts_list, faces=faces_list)
                            diversity_points = diversity_points.verts_list()
                            diversity_points = sample_points(diversity_points, 5000) #TODO adjust number

                        if with_angles and diversity_angles is None:
                            diversity_boxes, diversity_angles = diversity_boxes

                        # Computing shape diversity on canonical and normalized shapes
                        normalized_points = []
                        filtered_diversity_retrieval_ids = []
                        for ins_id, obj_id in enumerate(dec_objs):
                            if obj_id.item() != 0 and testdataloader.dataset.classes_r[obj_id.item()] != 'floor':
                                # We only care for manipulated nodes
                                if diversity_keep[ins_id, 0] == 1:
                                    continue
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

                        # We use keep to filter changed nodes
                        boxes_diversity_sample.append(diversity_boxes[diversity_keep[:, 0] == 0])

                        if with_angles:
                            # We use keep to filter changed nodes
                            angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles[diversity_keep[:, 0] == 0].cpu().numpy(), 1), 1) / 24. * 360.) # TODO change this maybe

                        if len(normalized_points) > 0:
                            shapes_sample.append(torch.stack(normalized_points)) # keep has already been applied for points

                    # Compute standard deviation for box for this sample
                    if len(boxes_diversity_sample) > 0:
                        boxes_diversity_sample = torch.stack(boxes_diversity_sample, 1)
                        bs = boxes_diversity_sample.shape[0]
                        boxes_diversity_sample = batch_torch_denormalize_box_params(boxes_diversity_sample.reshape([-1, 6]),file=normalized_file).reshape([bs, -1, 6])
                        all_diversity_boxes += torch.std(boxes_diversity_sample, dim=1).cpu().numpy().tolist()

                    # Compute standard deviation for angle for this sample
                    if len(angle_diversity_sample) > 0:
                        angle_diversity_sample = np.stack(angle_diversity_sample, 1)
                        all_diversity_angles += [estimate_angular_std(d[:,0]) for d in angle_diversity_sample]

                    # Compute chamfer distances for shapes for this sample
                    if len(shapes_sample) > 0:
                        if len(diversity_retrieval_ids_sample) > 0:
                            diversity_retrieval_ids_sample = np.stack(diversity_retrieval_ids_sample, 1)
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
                            all_diversity_chamfer.append(np.mean(sequence_diversity))
        bp = []
        for i in range(len(keep)):
            if keep[i] == 0:
                bp.append(boxes_pred[i].cpu().detach())
            else:
                bp.append(dec_tight_boxes[i,:6].cpu().detach())

        all_pred_boxes.append(boxes_pred.cpu().detach())

        # compute relationship constraints accuracy through simple geometric rules
        accuracy = validate_constrains_changes(dec_triples, boxes_pred, dec_tight_boxes, keep, model.vocab, accuracy, file_dist=normalized_file,
                                               with_norm=True)
        accuracy_in_orig_graph = validate_constrains_changes(dec_triples, torch.stack(bp, 0), dec_tight_boxes, keep,
                                                             model.vocab, accuracy_in_orig_graph, file_dist=normalized_file, with_norm=True)
        accuracy_unchanged = validate_constrains(dec_triples, boxes_pred, dec_tight_boxes, keep, model.vocab,
                                                 accuracy_unchanged, with_norm=True)

    if with_diversity:
        print("DIVERSITY:")
        print("\tShape (Avg. Chamfer Distance) = %f" % (np.mean(all_diversity_chamfer)))
        print("\tBox (Std. metric size and location) = %f, %f" % (
            np.mean(np.mean(all_diversity_boxes, axis=0)[:3]),
            np.mean(np.mean(all_diversity_boxes, axis=0)[3:])))
        print("\tAngle (Std.) %s = %f" % (k, np.mean(all_diversity_angles)))

    keys = list(accuracy.keys())
    for dic, typ in [(accuracy, "changed nodes"), (accuracy_unchanged, 'unchanged nodes'),
                     (accuracy_in_orig_graph, 'changed nodes placed in original graph')]:
        # NOTE 'changed nodes placed in original graph' are the results reported in the paper!
        # The unchanged nodes are kept from the original scene, and the accuracy in the new nodes is computed with
        # respect to these original nodes
        print('{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(typ, np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                                                    np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]),
                                                                                    np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
                                                                                    np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]),
                                                                                    np.mean(dic[keys[8]]), np.mean(dic[keys[9]]), np.mean(dic[keys[10]]), np.mean(dic[keys[11]])))
        print('means of mean: {:.2f}'.format(np.mean([np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                      np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]),
                                                      np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
                                                      np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]),
                                                      np.mean(dic[keys[8]]), np.mean(dic[keys[9]]), np.mean(dic[keys[10]])])))


def validate_constrains_loop(modelArgs, testdataloader, model, epoch=None, normalized_file=None, with_diversity=True, bin_angles=False, vocab=None,
                             point_classes_idx=None, export_3d=False, cat2objs=None, datasize='large',
                             num_samples=3, gen_shape=False):

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

    # from model.diff_utils.util_3d import init_mesh_renderer
    # dist, elev, azim = 1.7, 20, 20
    # sdf_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim,device='cuda')
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
        box_data['chair'].update(box_data['stool'])

    for i, data in enumerate(testdataloader, 0):
        # print(data['scan_id'])
        # if data['scan_id'][0] != 'LivingDiningRoom-13945':
        #     continue
        # if data['scan_id'][0].split('-')[0] not in ['DiningRoom', "LivingDiningRoom"]:
        #     continue
        # if data['scan_id'][0] not in ['MasterBedroom-58086','MasterBedroom-109561','Bedroom-11202', 'DiningRoom-2432', 'DiningRoom-451', 'DiningRoom-20718', 'LivingRoom-2050', 'LivingRoom-3540', 'LivingRoom-29294']:
            # continue

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

        with torch.no_grad():

            boxes_pred, shapes_pred = model.sample_box_and_shape(dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=gen_shape)

            boxes_pred, angles_pred = boxes_pred[:6], boxes_pred[6:]
            if bin_angles:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file) # mean, std
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred)
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max


        if args.visualize:
            colors = None
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            if model.type_ == 'cs++_l':
                render_box(model.type_, data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize, classes=classes, render_type='retrieval',
                       classed_idx=dec_objs, store_img=True, render_boxes=False, visual=True, demo=False, no_stool = args.no_stool, without_lamp=True)
            elif model.type_ == 'cs++':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_full(model.type_, data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, classes=classes, render_type='full', classed_idx=dec_objs,
                    shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=False)
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        if with_diversity:
            print('calculating diversity...')
            assert model.type_ == 'v2_full' or model.type_ == 'v1_full'
            # Run multiple times to obtain diversities
            # Diversity results for this dataset sample
            boxes_diversity_sample, shapes_sample, angle_diversity_sample, diversity_retrieval_ids_sample = [], [], [], []
            for sample in range(num_samples):
                # reseed(int(time.time()))
                diversity_boxes, diversity_points = model.sample_box_and_shape(point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                               attributes=None, gen_shape=True)
                if model.type_ == 'v2_full':
                    from model.diff_utils.util_3d import sdf_to_mesh
                    diversity_points = sdf_to_mesh(diversity_points, render_all=True)
                    diversity_points = diversity_points.verts_list()
                    diversity_points = sample_points(diversity_points, 5000) #TODO adjust number
                elif model.type_ == 'v1_full':
                    # TODO Complete shared shape decoding
                    diversity_points, _ = model.decode_g2sv1(dec_objs, diversity_points, box_data, retrieval=True)
                    from pytorch3d.structures import Meshes
                    verts_list = []
                    faces_list = []
                    for mesh in diversity_points:
                        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
                        faces = torch.tensor(mesh.faces, dtype=torch.int64)
                        verts_list.append(verts)
                        faces_list.append(faces)
                    diversity_points = Meshes(verts=verts_list, faces=faces_list)
                    diversity_points = diversity_points.verts_list()
                    diversity_points = sample_points(diversity_points, 5000)  # TODO adjust number

                if with_angles:
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

                if with_angles:
                    # We use keep to filter changed nodes
                    angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles.cpu().numpy(), 1), 1) / 24. * 360.) # TODO change this maybe

                if len(normalized_points) > 0:
                    shapes_sample.append(torch.stack(normalized_points)) # keep has already been aplied for points
                    # if model.type_ == 'sln':
                    #     diversity_retrieval_ids_sample.append(np.stack(filtered_diversity_retrieval_ids))


            # Compute standard deviation for box for this sample
            if len(boxes_diversity_sample) > 0:
                boxes_diversity_sample = torch.stack(boxes_diversity_sample, 1)
                bs = boxes_diversity_sample.shape[0]
                boxes_diversity_sample = batch_torch_denormalize_box_params(boxes_diversity_sample.reshape([-1, 6]),file=normalized_file).reshape([bs, -1, 6])
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
        accuracy = validate_constrains(dec_triples, boxes_pred, None, None, model.vocab, accuracy, file_dist=normalized_file, with_norm=model.type_ != 'sln')

    if export_3d:
        # export box and shape predictions for future evaluation
        result_path = os.path.join(args.exp, 'results')
        if not os.path.exists(result_path):
            # Create a new directory for results
            os.makedirs(result_path)
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
    for dic, typ in [(accuracy, "acc")]:

        print('{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(typ, np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                                           np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]),
                                                                           np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
                                                                           np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]),
                                                                           np.mean(dic[keys[8]]),
                                                                           np.mean(dic[keys[9]]),
                                                                           np.mean(dic[keys[10]]),np.mean(dic[keys[11]])))
        print('means of mean: {:.2f}'.format(np.mean([np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                      np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]),
                                                      np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
                                                      np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]),
                                                      np.mean(dic[keys[8]]),
                                                      np.mean(dic[keys[9]]),
                                                      np.mean(dic[keys[10]])])))


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


if __name__ == "__main__":
    evaluate()
