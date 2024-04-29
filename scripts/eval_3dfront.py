from __future__ import print_function

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
from helpers.visualize_scene import render_full, render_box
from omegaconf import OmegaConf
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/FRONT", help="dataset path")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, echoscene')
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

def validate_constrains_loop_w_changes(modelArgs, testdataset, model, normalized_file=None, bin_angles=False, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_changes = torch.utils.data.DataLoader(
        testdataset,
        batch_size=1,
        collate_fn=testdataset.collate_fn,
        shuffle=False,
        num_workers=0)
    vocab = testdataset.vocab
    obj_classes = sorted(list(set(vocab['object_idx_to_name'])))
    pred_classes = vocab['pred_idx_to_name']

    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy_unchanged[k] = []
        accuracy[k] = []

    for i, data in enumerate(test_dataloader_changes, 0):
        print(data['scan_id'][0])

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
            original = 0
            if original:
                # original graph
                print("***original graph***")
                original_data_dict = model.sample_box_and_shape(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                       gen_shape=gen_shape)
                original_boxes_pred, original_angles_pred = torch.concat((original_data_dict['sizes'], original_data_dict['translations']), dim=-1), original_data_dict['angles']
                original_shapes_pred = None
                try:
                    original_shapes_pred = original_data_dict['shapes']
                except:
                    print('no shape, only run layout branch.')
                original_angles_pred = postprocess_sincos2arctan(original_angles_pred) / np.pi * 180
                original_boxes_pred = descale_box_params(original_boxes_pred, file=normalized_file)  # min, max

            # manipulated graph
            print("***manipulated graph***")
            if len(manipulated_subs) and len(manipulated_objs):
                manipulated_nodes = manipulated_subs + manipulated_objs
                print('previous:' , obj_classes[enc_objs[manipulated_subs[0]]], pred_classes[manipulated_preds[0]], obj_classes[enc_objs[manipulated_objs[0]]])
                keep, data_dict = model.sample_boxes_and_shape_with_changes(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                            encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                            encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                            manipulated_nodes, gen_shape=gen_shape)
            else:
                keep, data_dict = model.sample_boxes_and_shape_with_additions(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                              encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                              encoded_dec_text_feat, encoded_dec_rel_feat,
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

            if args.visualize:
                # layout and shape visualization through open3d
                print("rendering", [obj_classes[i].strip('\n') for i in dec_objs])
                if model.type_ == 'echoscene':
                    # before manipulation
                    if original:
                        if original_shapes_pred is not None:
                            original_shapes_pred = original_shapes_pred.cpu().detach()
                        render_full(data['scan_id'], enc_objs.detach().cpu().numpy(), original_boxes_pred, original_angles_pred,
                                    datasize=datasize,
                                    classes=obj_classes, render_type=args.render_type, shapes_pred=original_shapes_pred,
                                    store_img=True,
                                    render_boxes=False, visual=True, demo=True, without_lamp=True,
                                    store_path=modelArgs['store_path']+"_before")

                    # after manipulation
                    if shapes_pred is not None:
                        shapes_pred = shapes_pred.cpu().detach()
                    render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred,
                                datasize=datasize,
                                classes=obj_classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True,
                                render_boxes=False, visual=True, demo=True, without_lamp=True,
                                store_path=modelArgs['store_path']+"_after")
                else:
                    raise NotImplementedError

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
        accuracy_unchanged = validate_constrains(dec_triples, boxes_pred_den, angles_pred, keep, model.vocab, accuracy_unchanged)

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{testdataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "changed nodes"), (accuracy_unchanged, 'unchanged nodes')]:
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


def validate_constrains_loop(modelArgs, test_dataset, model, epoch=None, normalized_file=None, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=0)

    vocab = test_dataset.vocab

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    for i, data in enumerate(test_dataloader_no_changes, 0):
        print(data['scan_id'])

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

        all_pred_boxes = []
        all_pred_angles = []

        with torch.no_grad():

            data_dict = model.sample_box_and_shape(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=gen_shape)

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
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            print("rendering", [classes[i].strip('\n') for i in dec_objs])
            if model.type_ == 'echolayout':
                render_box(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, store_img=False, render_boxes=False, visual=False, demo=False, without_lamp=True, store_path=modelArgs['store_path'])
            elif model.type_ == 'echoscene':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=True, store_path=modelArgs['store_path'])
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())
        # compute constraints accuracy through simple geometric rules
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_pred, None, model.vocab, accuracy)

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
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = test_dataset_no_changes.box_normalized_stats
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = modelArgs['with_CLIP']
    model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=True, load_shape_branch=args.gen_shape)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    cat2objs = None

    print('\nEditing Mode - Additions')
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_addition_changes, model, normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_rels_changes, model,  normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataset_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

if __name__ == "__main__":
    print(torch.__version__)
    evaluate()
