import random
import torch.nn as nn
import torch.optim as optim
from model.graph import GraphTripleConvNet, _init_weights, make_mlp
from model.networks.diffusion_shape.echo2shape import EchoToShape
from model.networks.diffusion_shape.diff_utils.visualizer import Visualizer
from model.networks.diffusion_shape.diff_utils.distributed import get_rank
from model.networks.diffusion_layout.echo2layout import EchoToLayout
import numpy as np
from helpers.lr_scheduler import *
from omegaconf import OmegaConf


class Sg2ScDiffModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a separate embedding of shape and bounding box latents.
    """
    def __init__(self, vocab, diff_opt, diffusion_bs=8, embedding_dim=128, batch_size=32,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 separated=False,
                 replace_latent=False,
                 residual=False,
                 use_angles=False,
                 use_clip=True):
        super(Sg2ScDiffModel, self).__init__()

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        self.replace_all_latent = replace_latent
        self.batch_size = batch_size
        self.embedding_dim = gconv_dim
        self.vocab = vocab
        self.use_angles = use_angles
        self.clip = use_clip
        add_dim = 0
        if self.clip:
            add_dim = 512
        self.obj_classes_grained = list(set(vocab['object_idx_to_name_grained']))
        self.edge_list = list(set(vocab['pred_idx_to_name']))
        self.obj_classes_list = list(set(vocab['object_idx_to_name']))
        self.classes = dict(zip(sorted(self.obj_classes_list),range(len(self.obj_classes_list))))
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
        num_objs = len(self.obj_classes_list)
        num_preds = len(self.edge_list)

        # build graph encoder and manipulator
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, gconv_dim * 2)
        self.pred_embeddings_ec = nn.Embedding(num_preds, gconv_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, gconv_dim * 2) # TODO is this necessary?
        self.pred_embeddings_man_dc = nn.Embedding(num_preds, gconv_dim * 2)

        self.out_dim_ini_encoder = gconv_dim * 2 + add_dim
        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2 + add_dim,
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual,
            'output_dim': self.out_dim_ini_encoder
        }
        self.out_dim_manipulator = gconv_dim * 2 + add_dim
        gconv_kwargs_manipulation = {
            'input_dim_obj': self.out_dim_ini_encoder + gconv_dim + gconv_dim * 2 + add_dim, # latent_f + change_flag + obj_embedding + clip
            'input_dim_pred': gconv_dim * 2 + add_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': min(gconv_num_layers, 5),
            'mlp_normalization': mlp_normalization,
            'residual': residual,
            'output_dim': self.out_dim_manipulator
        }
        self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        self.diff_cfg = diff_opt
        self.diffusion_bs = diffusion_bs if self.diff_cfg.hyper.batch_size is None else self.diff_cfg.hyper.batch_size
        self.s_l_separated = separated
        if self.s_l_separated:
            gconv_kwargs_ec_rel = {
                'input_dim_obj': self.out_dim_manipulator + gconv_dim * 2 + add_dim,
                'input_dim_pred': gconv_dim * 2 + add_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mlp_normalization': mlp_normalization,
                'residual': residual,
                'output_dim': self.out_dim_manipulator
            }
            self.gconv_net_ec_rel_s = GraphTripleConvNet(**gconv_kwargs_ec_rel)
            self.gconv_net_ec_rel_l = GraphTripleConvNet(**gconv_kwargs_ec_rel)

        ## shape branch
        self.ShapeDiff = EchoToShape(self.diff_cfg)
        self.visualizer = Visualizer(self.diff_cfg) # visualizer
        if get_rank() == 0:
            self.visualizer.setup_io()
        rel_s_layers = [gconv_dim * 2 + add_dim, 960, 1280] # cross attn
        if self.ShapeDiff.df.conditioning_key == 'concat':
            rel_s_layers = [gconv_dim * 2 + add_dim, 1280, 4096]
        self.rel_s_mlp = make_mlp(rel_s_layers, batch_norm=mlp_normalization, norelu=True)
        self.sample_obj = self.diff_cfg.shape_branch.sampling
        shape_df_conf = OmegaConf.load(self.diff_cfg.shape_branch.df_cfg)
        if shape_df_conf.unet.params.messsage_passing:
            assert self.sample_obj == 'greedy', 'if messsage_passing in the shape branch, greedy sampling is the only option for now.'

        ## layout branch
        self.LayoutDiff = EchoToLayout(self.diff_cfg)

        # initialization
        self.rel_s_mlp.apply(_init_weights)
        self.lr_init = self.diff_cfg.hyper.lr_init
        self.lr_step = self.diff_cfg.hyper.lr_step
        self.lr_evo = self.diff_cfg.hyper.lr_evo

    # 0-35k->35k-70k->70k-120k->120k-
    # 1e-4 -> 5e-5 -> 1e-5 -> 5e-6
    def lr_lambda(self, counter):
        # 35000
        if counter < self.lr_step[0]:
            return 1.0
        # 70000
        elif counter < self.lr_step[1]:
            return self.lr_evo[0] / self.lr_init
        # 120000
        elif counter < self.lr_step[2]:
            return self.lr_evo[1] / self.lr_init
        else:
            return self.lr_evo[2] / self.lr_init

    def optimizer_ini(self):
        gcn_layout_df_params = [p for p in self.parameters() if p.requires_grad == True]
        shape_df_params = self.ShapeDiff.trainable_params
        trainable_params = gcn_layout_df_params + shape_df_params
        self.optimizerFULL = optim.AdamW(trainable_params, lr=1e-4)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizerFULL, lr_lambda=self.lr_lambda)
        self.optimizers = [self.optimizerFULL]

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        # print('[*] learning rate = %.7f' % lr)
        return lr

    def init_encoder(self, objs, triples, enc_text_feat, enc_rel_feat):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_embed = self.obj_embeddings_ec(objs)
        pred_embed = self.pred_embeddings_ec(p)
        if self.clip:
            obj_embed = torch.cat([enc_text_feat, obj_embed], dim=1)
            pred_embed = torch.cat([enc_rel_feat, pred_embed], dim=1)

        latent_obj_f, latent_pred_f = self.gconv_net_ec(obj_embed, pred_embed, edges)

        return obj_embed, pred_embed, latent_obj_f, latent_pred_f

    def layout_encoder(self, latent_obj_vec, obj_embs, pred_embs, triples):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs_ = torch.cat([latent_obj_vec, obj_embs], dim=1)

        obj_vecs_, pred_vecs_ = self.gconv_net_ec_rel_l(obj_vecs_, pred_embs, edges)

        return obj_vecs_, pred_vecs_

    def shape_encoder(self, latent_obj_vec, obj_embs, pred_embs, triples):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs_ = torch.cat([latent_obj_vec, obj_embs], dim=1)

        obj_vecs_, pred_vecs_ = self.gconv_net_ec_rel_s(obj_vecs_, pred_embs, edges)

        return obj_vecs_, pred_vecs_

    def manipulate(self, latent_f, objs, triples, dec_text_feat, dec_rel_feat):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_embed = self.obj_embeddings_ec(objs) # TODO is obj_embeddings_ec enough here?
        pred_embed = self.pred_embeddings_ec(p) # TODO is pred_embeddings_ec enough here?
        if self.clip:
            obj_embed = torch.cat([dec_text_feat, obj_embed], dim=1)
            pred_embed = torch.cat([dec_rel_feat, pred_embed], dim=1)

        obj_vecs_ = torch.cat([latent_f, obj_embed], dim=1)
        obj_vecs_, pred_vecs_ = self.gconv_net_manipulation(obj_vecs_, pred_embed, edges)

        return obj_vecs_, pred_vecs_, obj_embed, pred_embed


    def balance_objects(self, cat_grained_list, cat_list, n, shape_branch=False):
        assert len(cat_grained_list) == len(cat_list), "grained list and coarse list must have the same length"

        unique_grained_ids = torch.unique(cat_grained_list)
        selected_object_indices = []

        # find n fine-grained objects to ensure diffusion meet all fine-grained classes in the scene
        if len(unique_grained_ids) >= n:
            sampled_grained = random.sample(unique_grained_ids.tolist(), n)

        # fine-grained classes less than n, we take all fine-grained classes and randomly obtain the rest
        else:
            sampled_grained = unique_grained_ids.tolist()
            remaining_n = n - len(unique_grained_ids)
            sampled_grained += random.choices(cat_grained_list.tolist(), k=remaining_n)

        # find the corresponding ids in the coarse object classes
        for grained_cat_id in sampled_grained:
            selected_indices = [i for i, x in enumerate(cat_grained_list) if x == grained_cat_id]
            selected_index = selected_indices[random.choice(range(len(selected_indices)))]
            selected_object_indices.append(selected_index)

        return torch.tensor(selected_object_indices)

    def find_closest_sublist(self, nums, num):
        # Create a list of (value, index) tuples
        nums_indexed = [(val, idx) for idx, val in enumerate(nums)]
        # Sort the list by value
        nums_indexed.sort(reverse=True)

        # Initialize variables
        current_sum = 0
        selected_indices = []
        selected_values = []

        # Iterate through the sorted list
        for value, index in nums_indexed:
            if current_sum + value <= num:
                current_sum += value
                selected_indices.append(index)
                selected_values.append(value)
                if current_sum == num:
                    break  # Found the exact match

        # Return the indices and values of the selected elements
        # Sort the indices to return them in the original order
        return sorted(selected_indices), selected_values, current_sum

    def select_sdfs(self, dec_objs_to_scene, obj_cats, obj_cats_grained, triples, sdfs, s_feat_ucon, s_feat_con, sample_type='balance'):
        dec_objs_to_scene = dec_objs_to_scene.detach().cpu().numpy()
        batch_size = np.max(dec_objs_to_scene) + 1
        scene_ids = []
        sdf_selected = []
        uc_rel_s_selected = []
        c_rel_s_selected = []
        obj_cat_selected = []
        triples_selected= []
        if sample_type != 'greedy':
            num_obj = int(np.ceil(self.diffusion_bs / batch_size)) # how many objects should be picked in a scene
            for i in range(batch_size):
                scene_i_ids = np.where(dec_objs_to_scene == i)[0]
                # sdf, node classes, node fine_grained classes in the current scene
                sdf_candidates = sdfs[scene_i_ids]
                obj_cat = obj_cats[scene_i_ids]
                obj_cat_grained = obj_cats_grained[scene_i_ids]

                # relation embeddings conditioning the shape diffusion in the current scene
                uc_rel_s = s_feat_ucon[scene_i_ids]
                c_rel_s = s_feat_con[scene_i_ids]

                zeros_tensor = torch.zeros_like(sdf_candidates[0])
                mask = torch.ne(sdf_candidates, zeros_tensor) # find out the objects which are not the floor
                object_inds = torch.unique(torch.where(mask)[0]) # non-floor non-__scene__ object indices (for shape branch)
                if sample_type=='random':
                    # randomly choose num_obj elements for shape branch
                    perm_obj = torch.randperm(len(object_inds))
                    random_obj_inds = object_inds[perm_obj[:num_obj]]
                    sdf_selected.append(sdf_candidates[random_obj_inds])
                    uc_rel_s_selected.append(uc_rel_s[random_obj_inds])
                    c_rel_s_selected.append(c_rel_s[random_obj_inds])
                    obj_cat_selected.append(obj_cat[random_obj_inds])
                elif sample_type=='balance':
                    ## balance every fine-grained category.
                    # shape branch excludes the floor
                    selected_inds = self.balance_objects(obj_cat_grained[object_inds], obj_cat[object_inds], num_obj, shape_branch=True) # selected_inds are indices of obj_cat_grained[object_inds]
                    sdf_selected.append(sdf_candidates[object_inds][selected_inds])
                    uc_rel_s_selected.append(uc_rel_s[object_inds][selected_inds])
                    c_rel_s_selected.append(c_rel_s[object_inds][selected_inds])
                    obj_cat_selected.append(obj_cat[object_inds][selected_inds])
                else:
                    raise NotImplementedError
                scene_ids.append(np.repeat(i, num_obj))
        else:
            # with floor and __scene__ (if dataset has __scene__) inside
            num = 0
            triple_id_end = -1
            scene_list = np.unique(dec_objs_to_scene)
            for i in scene_list:
                ids = np.where(dec_objs_to_scene == i)[0]
                if self.diffusion_bs - num < len(ids):
                    break
                triple_id_end += len(ids)
                sdf_selected.append(sdfs[ids])
                uc_rel_s_selected.append(s_feat_ucon[ids])
                c_rel_s_selected.append(s_feat_con[ids])
                obj_cat_selected.append(obj_cats[ids])
                scene_ids.append(dec_objs_to_scene[ids])

                num += len(ids)
            triples_mask = (triples[:, 0] < num) & (triples[:, 2] < num)
            triples_selected = triples[triples_mask]

        sdf_selected = torch.cat(sdf_selected, dim=0).cuda()
        uc_rel_s_selected = torch.cat(uc_rel_s_selected, dim=0).cuda()
        c_rel_s_selected = torch.cat(c_rel_s_selected, dim=0).cuda()
        obj_cat_selected = torch.cat(obj_cat_selected, dim=0)
        scene_ids = np.concatenate(scene_ids, axis=0)
        diff_dict = {'sdf': sdf_selected[:self.diffusion_bs], 'uc_s': uc_rel_s_selected[:self.diffusion_bs],
                     'c_s': c_rel_s_selected[:self.diffusion_bs], "scene_ids": scene_ids[:self.diffusion_bs]}
        if len(triples_selected):
            diff_dict.update({'triples': triples_selected})
        return obj_cat_selected[:self.diffusion_bs], diff_dict

    def prepare_boxes(self, triples, obj_embed, relation_cond, scene_ids=None, obj_boxes=None, obj_angles=None):
        if obj_boxes is not None and obj_angles is not None:
            obj_boxes = torch.cat((obj_boxes, obj_angles.reshape(-1,1)), dim=-1)
        diff_dict = {'preds': triples, 'box': obj_boxes, 'uc_b': obj_embed,
                     'c_b': relation_cond, "obj_id_to_scene": scene_ids}
        return diff_dict

    def forward(self, enc_objs, enc_triples, enc_text_feat, enc_rel_feat, dec_objs, dec_objs_grained,
                dec_triples, dec_boxes, dec_text_feat, dec_rel_feat, dec_objs_to_scene, missing_nodes, manipulated_nodes, dec_sdfs, dec_angles):

        obj_embed, _, latent_obj_vecs, latent_pred_vecs = self.init_encoder(enc_objs, enc_triples, enc_text_feat, enc_rel_feat)

        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.out_dim_ini_encoder)
          zeros = torch.from_numpy(noise.reshape(1, self.out_dim_ini_encoder))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          latent_obj_vecs = torch.cat([latent_obj_vecs[:ad_id], zeros, latent_obj_vecs[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(latent_obj_vecs)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
        latent_obj_vecs_, pred_vecs_ , obj_embed_, pred_embed_ = self.manipulate(latent_obj_vecs_, dec_objs, dec_triples, dec_text_feat, dec_rel_feat) # contains all obj now

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                latent_obj_vecs = torch.cat([latent_obj_vecs[:touched_node], latent_obj_vecs_[touched_node:touched_node + 1], latent_obj_vecs[touched_node + 1:]], dim=0)
        else:
            latent_obj_vecs = latent_obj_vecs_

        # relation embeddings -> diffusion
        c_rel_feat_s = latent_obj_vecs
        uc_rel_feat_s = self.rel_s_mlp(obj_embed_) # embedding + CLIP
        uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)

        c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
        c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)

        obj_selected, shape_diff_dict = self.select_sdfs(dec_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_sdfs, uc_rel_feat_s, c_rel_feat_s, sample_type=self.sample_obj)
        self.ShapeDiff.set_input(shape_diff_dict)
        self.ShapeDiff.set_requires_grad([self.ShapeDiff.df], requires_grad=True)
        Shape_loss, Shape_loss_dict = self.ShapeDiff.forward()

        box_diff_dict = self.prepare_boxes(dec_triples, obj_embed_, obj_boxes=dec_boxes, obj_angles=dec_angles,
                                           relation_cond=latent_obj_vecs, scene_ids=dec_objs_to_scene)
        self.LayoutDiff.set_input(box_diff_dict)
        self.LayoutDiff.set_requires_grad([self.LayoutDiff.df], requires_grad=True)
        Layout_loss, Layout_loss_dict = self.LayoutDiff.forward()


        loss_dict = {**Shape_loss_dict, **Layout_loss_dict}

        return obj_selected, Shape_loss, Layout_loss, loss_dict

    def sample(self, dec_objs, dec_triplets, dec_text_feat, dec_rel_feat, gen_shape=False):
        with torch.no_grad():
            obj_embed, pred_embed, latent_obj_vecs, latent_pred_vecs = self.init_encoder(dec_objs, dec_triplets,
                                                                                         dec_text_feat,
                                                                                         dec_rel_feat)
            change_repr = []
            for i in range(len(latent_obj_vecs)):
                noisechange = np.zeros(self.embedding_dim)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
            change_repr = torch.stack(change_repr, dim=0)
            latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
            latent_obj_vecs_, pred_vecs_, obj_embed_, pred_embed_ = self.manipulate(latent_obj_vecs_, dec_objs,
                                                                                    dec_triplets, dec_text_feat,
                                                                                    dec_rel_feat)  # normal message passing

            box_diff_dict = self.prepare_boxes(dec_triplets, obj_embed_, relation_cond=latent_obj_vecs_)

            self.LayoutDiff.set_input(box_diff_dict)
            gen_box_dict = self.LayoutDiff.generate_layout_sg(
                box_dim=self.diff_cfg.layout_branch.denoiser_kwargs.in_channels)

            gen_sdf = None
            if gen_shape:
                # # relation embeddings -> diffusion
                c_rel_feat_s = latent_obj_vecs_
                uc_rel_feat_s = self.rel_s_mlp(obj_embed_)  # embedding + CLIP
                uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)
                c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
                c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)
                shape_diff_dict = {'obj_cat': dec_objs, 'triples': dec_triplets, 'c_s': c_rel_feat_s, 'uc_s': uc_rel_feat_s}
                gen_sdf = self.ShapeDiff.rel2shape(shape_diff_dict)

            return {'shapes': gen_sdf}, gen_box_dict

    def sample_with_changes(self, enc_objs, enc_triples, enc_text_feat, enc_rel_feat, dec_objs, dec_triplets, dec_text_feat, dec_rel_feat, manipulated_nodes, gen_shape=False):
        with torch.no_grad():
            obj_embed, pred_embed, latent_obj_vecs, latent_pred_vecs = self.init_encoder(enc_objs, enc_triples,
                                                                                         enc_text_feat,
                                                                                         enc_rel_feat)
            # mark changes in nodes
            change_repr = []
            for i in range(len(latent_obj_vecs)):
                if i not in manipulated_nodes:
                    noisechange = np.zeros(self.embedding_dim)
                else:
                    noisechange = np.random.normal(0, 1, self.embedding_dim)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
            change_repr = torch.stack(change_repr, dim=0)
            latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
            latent_obj_vecs_, pred_vecs_, obj_embed_, pred_embed_ = self.manipulate(latent_obj_vecs_, dec_objs,
                                                                                    dec_triplets, dec_text_feat,
                                                                                    dec_rel_feat)
            if not self.replace_all_latent:
                # take original nodes when untouched
                touched_nodes = torch.tensor(sorted(manipulated_nodes)).long()
                for touched_node in touched_nodes:
                    latent_obj_vecs = torch.cat(
                        [latent_obj_vecs[:touched_node], latent_obj_vecs_[touched_node:touched_node + 1],
                         latent_obj_vecs[touched_node + 1:]], dim=0)
            else:
                latent_obj_vecs = latent_obj_vecs_

            box_diff_dict = self.prepare_boxes(dec_triplets, obj_embed_, relation_cond=latent_obj_vecs)
            self.LayoutDiff.set_input(box_diff_dict)
            layout_dict = self.LayoutDiff.generate_layout_sg(
                box_dim=self.diff_cfg.layout_branch.denoiser_kwargs.in_channels)
            gen_sdf = None
            if gen_shape:
                # relation embeddings -> diffusion
                c_rel_feat_s = latent_obj_vecs
                uc_rel_feat_s = self.rel_s_mlp(obj_embed_)  # embedding + CLIP
                uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)
                c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
                c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)
                shape_diff_dict = {'obj_cat': dec_objs, 'triples': dec_triplets, 'c_s': c_rel_feat_s,
                                   'uc_s': uc_rel_feat_s}
                gen_sdf = self.ShapeDiff.rel2shape(shape_diff_dict)
        keep = []
        for i in range(len(layout_dict["translations"])):
            if i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        return keep, {'shapes': gen_sdf}, layout_dict

    def sample_with_additions(self, enc_objs, enc_triples, enc_text_feat, enc_rel_feat, dec_objs, dec_triplets, dec_text_feat, dec_rel_feat, missing_nodes, gen_shape=False):
        with torch.no_grad():
            obj_embed, pred_embed, latent_obj_vecs, latent_pred_vecs = self.init_encoder(enc_objs, enc_triples, enc_text_feat, enc_rel_feat)
            # append zero nodes
            nodes_added=[]
            for i in range(len(missing_nodes)):
                ad_id = missing_nodes[i] + i
                nodes_added.append(ad_id)
                noise = np.zeros(self.out_dim_ini_encoder)
                zeros = torch.from_numpy(noise.reshape(1, self.out_dim_ini_encoder))
                zeros.requires_grad = True
                zeros = zeros.float().cuda()
                latent_obj_vecs = torch.cat([latent_obj_vecs[:ad_id], zeros, latent_obj_vecs[ad_id:]], dim=0)

            change_repr = []
            for i in range(len(latent_obj_vecs)):
                if i not in missing_nodes:
                    noisechange = np.zeros(self.embedding_dim)
                else:
                    noisechange = np.random.normal(0, 1, self.embedding_dim)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
            change_repr = torch.stack(change_repr, dim=0)
            latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
            latent_obj_vecs_, pred_vecs_, obj_embed_, pred_embed_ = self.manipulate(latent_obj_vecs_, dec_objs, dec_triplets, dec_text_feat, dec_rel_feat)

            if not self.replace_all_latent:
                # take original nodes when untouched
                touched_nodes = torch.tensor(sorted(nodes_added)).long()
                for touched_node in touched_nodes:
                    latent_obj_vecs = torch.cat(
                        [latent_obj_vecs[:touched_node], latent_obj_vecs_[touched_node:touched_node + 1],
                         latent_obj_vecs[touched_node + 1:]], dim=0)
            else:
                latent_obj_vecs = latent_obj_vecs_

            box_diff_dict = self.prepare_boxes(dec_triplets, obj_embed_, relation_cond=latent_obj_vecs)
            self.LayoutDiff.set_input(box_diff_dict)
            layout_dict = self.LayoutDiff.generate_layout_sg(
                box_dim=self.diff_cfg.layout_branch.denoiser_kwargs.in_channels)
            gen_sdf = None
            if gen_shape:
                # relation embeddings -> diffusion
                c_rel_feat_s = latent_obj_vecs
                uc_rel_feat_s = self.rel_s_mlp(obj_embed_)  # embedding + CLIP
                uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)
                c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
                c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)
                shape_diff_dict = {'obj_cat': dec_objs, 'triples': dec_triplets, 'c_s': c_rel_feat_s,
                                   'uc_s': uc_rel_feat_s}
                gen_sdf = self.ShapeDiff.rel2shape(shape_diff_dict)

        keep = []
        for i in range(len(layout_dict["translations"])):
            if i not in nodes_added:
                keep.append(1)
            else:
                keep.append(0)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        return keep, {'shapes': gen_sdf}, layout_dict

    def state_dict(self, epoch, counter):
        state_dict_1_layout = super(Sg2ScDiffModel, self).state_dict()
        state_dict_basic = {'epoch': epoch, 'counter': counter, 'opt': self.optimizerFULL.state_dict()}
        state_dict_shape = {
            'vqvae': self.ShapeDiff.vqvae_module.state_dict(),
            'shape_df': self.ShapeDiff.df_module.state_dict(),
        }
        state_dict_1_layout.update(state_dict_basic)
        state_dict_1_layout.update(state_dict_shape)
        return state_dict_1_layout