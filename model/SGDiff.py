import json

import torch
import torch.nn as nn

import pickle
import os
import glob

import trimesh
from termcolor import colored
from model.CS_plus_plus import Sg2ScDiffModel as CS_pp


class SGDiff(nn.Module):

    def __init__(self, type='cs++', diff_opt = '../config/v2_full.yaml', vocab=None, replace_latent=False, with_changes=True, distribution_before=True,
                 residual=False, gconv_pooling='avg', with_angles=False, clip=True, separated=False):
        super().__init__()
        assert type in ['cs++', 'cs++_l'], '{} is not included'.format(type)

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles
        self.epoch = 0

        if self.type_ == 'cs++':
            self.diff_opt = diff_opt
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.diff = CS_pp(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, mlp_normalization="batch", separated=separated,
                              gconv_num_layers=5, use_angles=with_angles, distribution_before=distribution_before, replace_latent=replace_latent, residual=residual, use_clip=clip)
            self.diff.optimizer_ini()
        else:
            raise NotImplementedError
        self.counter = 0

    def forward_mani(self, enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_objs_grained,
                     dec_triples, dec_boxes, dec_angles, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes):

        if self.type_ == 'cs++':
            obj_selected, shape_loss, layout_loss, loss_dict = self.diff.forward(
                enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat, dec_objs, dec_objs_grained, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_objs_to_scene, missing_nodes,
                manipulated_nodes, dec_sdfs, dec_angles)
        else:
            raise NotImplementedError

        return obj_selected, shape_loss, layout_loss, loss_dict

    def load_networks(self, exp, epoch, strict=True, restart_optim=False):
        if self.type_ == 'cs++':
            from omegaconf import OmegaConf
            diff_cfg = OmegaConf.load(self.diff_opt)
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
            diff_state_dict = {}
            diff_state_dict['vqvae'] = ckpt.pop('vqvae')
            diff_state_dict['shape_df'] = ckpt.pop('shape_df')
            diff_state_dict['opt'] = ckpt.pop('opt')
            try:
                self.epoch = ckpt.pop('epoch')
                self.counter = ckpt.pop('counter')
            except:
                print('no epoch or counter record.')
            self.diff.load_state_dict(
                ckpt,
                strict=strict
            )
            print(colored('[*] GCN and layout successfully restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                        'model{}.pth'.format(epoch)),
                          'blue'))
            self.diff.ShapeDiff.vqvae.load_state_dict(diff_state_dict['vqvae'])
            self.diff.ShapeDiff.df.load_state_dict(diff_state_dict['shape_df'])

            if not restart_optim:
                import torch.optim as optim
                self.diff.optimizerFULL.load_state_dict(diff_state_dict['opt'])
                # self.vae_v2.scheduler = optim.lr_scheduler.StepLR(self.vae_v2.optimizerFULL, 10000, 0.9)
                self.diff.scheduler = optim.lr_scheduler.LambdaLR(self.diff.optimizerFULL, lr_lambda=self.diff.lr_lambda,
                                                        last_epoch=int(self.counter - 1))

            # for multi-gpu (deprecated)
            if diff_cfg.hyper.distributed:
                self.diff.ShapeDiff.make_distributed(diff_cfg)
                self.diff.ShapeDiff.df_module = self.diff.ShapeDiff.df.module
                self.diff.ShapeDiff.vqvae_module = self.diff.ShapeDiff.vqvae.module
            else:
                self.diff.ShapeDiff.df_module = self.diff.ShapeDiff.df
                self.diff.ShapeDiff.vqvae_module = self.diff.ShapeDiff.vqvae
            print(colored('[*] v2_shape successfully restored from: %s' % os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)), 'blue'))

    def decoder_with_changes_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, box_data=None, gen_shape=False):

        if self.type_ == 'cs++':
            boxes, sdfs, keep = self.diff.decoder_with_changes(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                               manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep
        else:
            raise NotImplementedError

    def decoder_with_changes_boxes(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        if self.type_ == 'cs++_l':
            return self.vae_box.decoder_with_changes(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes)
        else:
            raise NotImplementedError

    def decoder_boxes(self, z, objs, triples, attributes):
        if self.type_ == 'cs++_l':
            if self.with_angles:
                return self.vae_box.decoder(z, objs, triples, attributes)
            else:
                return self.vae_box.decoder(z, objs, triples, attributes), None

    def decoder_with_additions_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                               manipulated_nodes, gen_shape=False):
        if self.type_ == 'cs++_l':
            boxes, keep = self.decoder_with_additions_boxs(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                                manipulated_nodes)
            return boxes, None, keep
        elif self.type_ == 'cs++':
            boxes, sdfs, keep = self.diff.decoder_with_additions(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                         manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep
        else:
            raise NotImplementedError

    def decoder_with_additions_boxs(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        boxes, angles, keep = None, None, None
        if self.type_ == 'cs++_l':
            boxes, keep = self.vae_box.decoder_with_additions(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                            manipulated_nodes, (self.mean_est_box, self.cov_est_box))
        else:
            raise NotImplementedError
        return boxes, angles, keep

    def encode_box_and_shape(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, feats, boxes, angles=None, attributes=None):
        if not self.with_angles:
            angles = None
        if self.type_ == 'cs++_l' or self.type_ == 'cs++':
            return self.encode_box(objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles, attributes), (None, None)
        else:
            raise NotImplementedError

    def encode_box(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles=None, attributes=None):

        if self.type_ == 'cs++_l':
            z, log_var = self.vae_box.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat, encoded_enc_rel_feat, angles)
        elif self.type_ == 'cs++':
            z, log_var = self.diff.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat,
                                              encoded_enc_rel_feat, angles)
        else:
            raise NotImplementedError

        return z, log_var

    def sample_box_and_shape(self, point_classes_idx, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=False):
        if self.type_ == 'cs++_l':
            return self.diff.sample(point_classes_idx, self.mean_est, self.cov_est, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                   attributes, gen_shape=gen_shape)
        else:
            raise NotImplementedError

    def sample_box(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None):
        if self.type_ == 'cs++_l':
            return self.vae_box.sampleBoxes(self.mean_est_box, self.cov_est_box, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        elif self.type_ == 'cs++':
            return self.diff.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)[0]
        else:
            raise NotImplementedError

    def save(self, exp, outf, epoch, counter=None):
        if self.type_ == 'cs++_l':
            torch.save(self.vae_box.state_dict(), os.path.join(exp, outf, 'model_box_{}.pth'.format(epoch)))
        elif self.type_ == 'cs++':
            torch.save(self.diff.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        else:
            raise NotImplementedError
