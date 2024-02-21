from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
import time
from tqdm import tqdm
from omegaconf import OmegaConf
sys.path.append('../')
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from model.SGDiff import SGDiff
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from model.losses import bce_loss
from helpers.util import bool_flag, _CustomDataParallel

from model.losses import calculate_model_losses

import torch.nn.functional as F
import json

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# standard hyperparameters, batch size, learning rate, etc
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')

# paths and filenames
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/3D-FRONT",
                    help="dataset path")
parser.add_argument('--logf', default='logs', help='folder to save tensorboard logs')
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--room_type', default='bedroom', help='room type [bedroom, livingroom, diningroom, library, all]')

# GCN parameters
parser.add_argument('--residual', type=bool_flag, default=False, help="residual in GCN")
parser.add_argument('--pooling', type=str, default='avg', help="pooling method in GCN")

# dataset related
parser.add_argument('--large', default=False, type=bool_flag,
                    help='large set of class labels. Use mapping.json when false')
parser.add_argument('--use_scene_rels', type=bool_flag, default=True, help="connect all nodes to a root scene node")

parser.add_argument('--separated', type=bool_flag, default=True, help="if use relation encoder in the diffusion branch")
parser.add_argument('--with_SDF', type=bool_flag, default=False)  # TODO
parser.add_argument('--with_feats', type=bool_flag, default=False,
                    help="if true reads latent point features instead of pointsets."
                         "If not existing, they get generated at the beginning.")  # TODO
parser.add_argument('--with_CLIP', type=bool_flag, default=True,
                    help="if use CLIP features. Set true for the full version")
parser.add_argument('--shuffle_objs', type=bool_flag, default=True, help="shuffle objs of a scene")
parser.add_argument('--use_canonical', default=True, type=bool_flag)  # TODO
parser.add_argument('--with_angles', default=True, type=bool_flag)
parser.add_argument('--bin_angle', default=False, type=bool_flag)
parser.add_argument('--num_box_params', default=6, type=int, help="number of the dimension of the bbox. [6,7]")

# training and architecture related
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--with_changes', default=True, type=bool_flag)
parser.add_argument('--loadmodel', default=False, type=bool_flag)
parser.add_argument('--loadepoch', default=90, type=int, help='only valid when loadmodel is true')
parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='cs++', choices=['cs++', 'cs++_l'], type=str)
parser.add_argument('--diff_yaml', default='../config/cs_full_mp.yaml', type=str,
                    help='config of the diffusion network [cross_attn/concat]')

parser.add_argument('--vis_num', type=int, default=8, help='for visualization in the training')

args = parser.parse_args()
print(args)


def parse_data(data):
    enc_objs, enc_triples, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                     data['encoder']['tripltes'], \
                                                                     data['encoder']['obj_to_scene'], \
                                                                     data['encoder']['triple_to_scene']
    if args.with_feats:
        encoded_enc_f = data['encoder']['feats']
        encoded_enc_f = encoded_enc_f.cuda()

    encoded_enc_text_feat = None
    encoded_enc_rel_feat = None
    encoded_dec_text_feat = None
    encoded_dec_rel_feat = None
    if args.with_CLIP:
        encoded_enc_text_feat = data['encoder']['text_feats'].cuda()
        encoded_enc_rel_feat = data['encoder']['rel_feats'].cuda()
        encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
        encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()

    dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                      data['decoder']['tripltes'], \
                                                                                      data['decoder']['boxes'], \
                                                                                      data['decoder']['obj_to_scene'], \
                                                                                      data['decoder']['triple_to_scene']
    dec_objs_grained = data['decoder']['objs_grained']
    dec_sdfs = None
    if 'sdfs' in data['decoder']:
        dec_sdfs = data['decoder']['sdfs']
    if 'feats' in data['decoder']:
        encoded_dec_f = data['decoder']['feats']
        encoded_dec_f = encoded_dec_f.cuda()

    # changed nodes
    missing_nodes = data['missing_nodes']
    changed_triples = (data['manipulated_subs'], data['manipulated_preds'], data['manipulated_objs']) # this is the real triple

    enc_objs, enc_triples = enc_objs.cuda(), enc_triples.cuda()
    dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
    dec_objs_grained = dec_objs_grained.cuda()

    enc_scene_nodes = enc_objs == 0
    dec_scene_nodes = dec_objs == 0
    if not args.with_feats:
        with torch.no_grad():
            encoded_enc_f = None  # TODO
            encoded_dec_f = None  # TODO

    # set all scene (dummy) node encodings to zero
    try:
        encoded_enc_f[enc_scene_nodes] = torch.zeros(
            [torch.sum(enc_scene_nodes), encoded_enc_f.shape[1]]).float().cuda()
        encoded_dec_f[dec_scene_nodes] = torch.zeros(
            [torch.sum(dec_scene_nodes), encoded_dec_f.shape[1]]).float().cuda()
    except:
        if args.network_type == 'cs++_box':
            encoded_enc_f = None
            encoded_dec_f = None

    if args.num_box_params == 7:
        # all parameters, including angle, procesed by the box_net
        dec_boxes = dec_tight_boxes
    elif args.num_box_params == 6:
        # no angle. this will be learned separately if with_angle is true
        dec_boxes = dec_tight_boxes[:, :6]
    else:
        raise NotImplementedError

    dec_angles = dec_tight_boxes[:, 6]
    # # limit the angle bin range from 0 to 24
    # # TODO Is it necessary?
    # dec_angles = dec_tight_boxes[:, 6].long() - 1
    # dec_angles = torch.where(dec_angles > 0, dec_angles, torch.zeros_like(dec_angles))
    # dec_angles = torch.where(dec_angles < 24, dec_angles, torch.zeros_like(dec_angles))

    return enc_objs, enc_triples, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
           enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs, \
           encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes, changed_triples




def train():
    """ Train the network based on the provided argparse parameters
    """
    args.manualSeed = random.randint(1, 10000)  # optionally fix seed 7494
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # instantiate scene graph dataset for training
    dataset = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='train_scans',
        shuffle_objs=args.shuffle_objs,
        use_SDF=args.with_SDF,
        use_scene_rels=args.use_scene_rels,
        with_changes=args.with_changes,
        with_feats=args.with_feats,
        with_CLIP=args.with_CLIP,
        large=args.large,
        seed=False,
        bin_angle=args.bin_angle,
        room_type=args.room_type,
        recompute_feats=False,
        recompute_clip=False)

    # instantiate data loader from dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        num_workers=int(args.workers))

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    # instantiate the model
    diff_cfg = OmegaConf.load(args.diff_yaml)
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = dataset.box_normalized_stats
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = args.with_CLIP
    model = SGDiff(type=args.network_type, diff_opt=diff_cfg, vocab=dataset.vocab,
                replace_latent=args.replace_latent, with_changes=args.with_changes, residual=args.residual,
                gconv_pooling=args.pooling, with_angles=args.with_angles, clip=args.with_CLIP, separated=args.separated)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loadmodel:
        model.load_networks(exp=args.exp, epoch=args.loadepoch, restart_optim=False)

    # initialize tensorboard writer
    writer = SummaryWriter(args.exp + "/" + args.logf)

    print("---- Model and Dataset built ----")

    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

    # save parameters so that we can read them later on evaluation
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Saving all parameters under:")
    print(os.path.join(args.exp, 'args.json'))

    torch.autograd.set_detect_anomaly(True)
    counter = model.counter if model.counter else 0

    print("---- Starting training loop! ----")
    iter_start_time = time.time()
    start_epoch = model.epoch if model.epoch else 0
    for epoch in range(start_epoch, args.nepoch):
        print('Epoch: {}/{}'.format(epoch, args.nepoch))
        for i, data in enumerate(tqdm(dataloader)):

            # parse the data to the network
            try:
                enc_objs, enc_triples, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes, manipulated_triples = parse_data(data)
            except Exception as e:
                print('Exception', str(e))
                continue

            if args.bin_angle:
                # limit the angle bin range from 0 to 24
                dec_angles = torch.where(dec_angles > 0, dec_angles, torch.zeros_like(dec_angles))
                dec_angles = torch.where(dec_angles < 24, dec_angles, torch.zeros_like(dec_angles))

            model.diff.optimizerFULL.zero_grad()

            model = model.train()

            obj_selected, shape_loss, layout_loss, loss_dict = model.forward_mani(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                           dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                           dec_objs_to_scene, missing_nodes, manipulated_triples)

            if args.network_type == 'cs++':
                model.diff.ShapeDiff.update_loss()

            loss = shape_loss + layout_loss

            # optimize
            loss.backward()

            # Cap the occasional super mutant gradient spikes
            # Do now a gradient step and plot the losses
            if args.network_type == 'cs++':
                torch.nn.utils.clip_grad_norm_(model.diff.ShapeDiff.df_module.parameters(), 5.0)
            for group in model.diff.optimizerFULL.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                        print('NaN grad in step {}.'.format(counter))
                        p.grad[torch.isnan(p.grad)] = 0

            model.diff.optimizerFULL.step()

            counter += 1

            current_lr = model.diff.update_learning_rate()
            writer.add_scalar("learning_rate", current_lr, counter)

            if counter % 50 == 0:
                message = "loss at {}: box {:.4f}, shape {:.4f}. Lr:{:.6f}".format( counter, layout_loss, shape_loss, current_lr)
                if args.network_type == 'cs++':
                    loss_diff = model.diff.ShapeDiff.get_current_errors()
                    for k, v in loss_diff.items():
                        message += ' %s: %.6f ' % (k, v)
                print(message)

            writer.add_scalar('Loss_BBox', layout_loss, counter)
            writer.add_scalar('Loss_Translation', loss_dict['loss.trans'], counter)
            writer.add_scalar('Loss_Size', loss_dict['loss.size'], counter)
            writer.add_scalar('Loss_Angle', loss_dict['loss.angle'], counter)
            writer.add_scalar('Loss_IoU', loss_dict['loss.liou'], counter)
            writer.add_scalar('Loss_Shape', shape_loss, counter)

            # t = (time.time() - iter_start_time) / args.batchSize
            # loss_diff = model.diff.ShapeDiff.get_current_errors()
            # model.diff.visualizer.print_current_errors(writer, counter, loss_diff, t)
            if counter % 10000 == 0 and obj_selected is not None:
                model.diff.ShapeDiff.gen_shape_after_foward(num_obj=args.vis_num)
                model.diff.visualizer.display_current_results(writer, model.diff.ShapeDiff.get_current_visuals(
                    dataset.classes_r, obj_selected.detach().cpu().numpy(), num_obj=args.vis_num),
                                                                counter, phase='train')



        if epoch % 100 == 0:
            model.save(args.exp, args.outf, epoch, counter=counter)
            print('saved model_{}'.format(epoch))

    writer.close()


if __name__ == "__main__":
    train()
