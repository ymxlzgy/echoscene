import random
import torch.nn as nn
import torch.optim as optim
from model.graph import GraphTripleConvNet, _init_weights, make_mlp
from model.networks.diffusion_shape.sdfusion_txt2shape_model import SDFusionText2ShapeModel
from model.networks.diffusion_shape.diff_utils.visualizer import Visualizer
from model.networks.diffusion_shape.diff_utils.distributed import get_rank
from model.networks.diffusion_layout2.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
import numpy as np
from helpers.lr_scheduler import *


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
        self.ShapeDiff = SDFusionText2ShapeModel(self.diff_cfg)
        self.visualizer = Visualizer(self.diff_cfg) # visualizer
        if get_rank() == 0:
            self.visualizer.setup_io()
        rel_s_layers = [gconv_dim * 2 + add_dim, 960, 1280] # cross attn
        if self.ShapeDiff.df.conditioning_key == 'concat':
            rel_s_layers = [gconv_dim * 2 + add_dim, 1280, 4096]
        self.rel_s_mlp = make_mlp(rel_s_layers, batch_norm=mlp_normalization, norelu=True)

        ## layout branch
        self.LayoutDiff = DiffusionSceneLayout_DDPM(self.diff_cfg, edge_classes=num_preds)
        # l_crossattn_dim = self.diff_cfg.layout_branch.denoiser_kwargs.crossattn_dim
        # rel_l_layers = [gconv_dim * 2 + add_dim, 960, l_crossattn_dim] # cross attn
        # if self.LayoutDiff.df.model.conditioning_key == 'concat':
        #     l_concat_dim = self.diff_cfg.layout_branch.denoiser_kwargs.concat_dim
        #     rel_l_layers = [gconv_dim * 2 + add_dim, 1280, l_concat_dim]
        # self.rel_l_mlp = make_mlp(rel_l_layers, batch_norm=mlp_normalization, norelu=True)
        #
        #
        # # initialization
        self.rel_s_mlp.apply(_init_weights)
        # self.rel_l_mlp.apply(_init_weights)
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

        return obj_embed, pred_embed, obj_vecs_, pred_vecs_

    # def decoder(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes, manipulate=False):
    #     s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
    #     s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
    #     edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)
    #
    #     obj_vecs = self.obj_embeddings_dc(objs)
    #     pred_vecs = self.pred_embeddings_dc(p)
    #     if self.clip:
    #         obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
    #         pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)
    #     else:
    #         obj_vecs_, pred_vecs_ = obj_vecs, pred_vecs
    #
    #     # concatenate noise first
    #     if self.decoder_cat:
    #         obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)
    #         obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)
    #
    #     # concatenate noise after gconv
    #     else:
    #         obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)
    #         obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)
    #
    #     d3_pred = self.d3_net(obj_vecs_)
    #     if self.use_angles:
    #         angles_pred = F.log_softmax(self.angle_net(obj_vecs_), dim=1)
    #         return d3_pred, angles_pred
    #     else:
    #         return d3_pred

    # def decoder_with_additions(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, distribution=None,gen_shape=False):
    #     nodes_added = []
    #     if distribution is not None:
    #         mu, cov = distribution
    #
    #     for i in range(len(missing_nodes)):
    #         ad_id = missing_nodes[i] + i
    #         nodes_added.append(ad_id)
    #         noise = np.zeros(z.shape[1])  # np.random.normal(0, 1, 64)
    #         if distribution is not None:
    #             zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
    #         else:
    #             zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
    #         zeros.requires_grad = True
    #         zeros = zeros.float().cuda()
    #         z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)
    #
    #     gen_sdf = None
    #     if gen_shape:
    #         un_rel_feat, rel_feat = self.encoder_2(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
    #         sdf_candidates = dec_sdfs
    #         length = objs.size(0)
    #         zeros_tensor = torch.zeros_like(sdf_candidates[0])
    #         mask = torch.ne(sdf_candidates, zeros_tensor)
    #         ids = torch.unique(torch.where(mask)[0])
    #         obj_selected = objs[ids]
    #         diff_dict = {'sdf': dec_sdfs[ids], 'rel': rel_feat[ids], 'uc': un_rel_feat[ids]}
    #         gen_sdf = self.ShapeDiff.rel2shape(diff_dict, uc_scale=3.)
    #     dec_man_enc_pred = self.decoder(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
    #
    #     keep = []
    #     for i in range(len(z)):
    #         if i not in nodes_added and i not in manipulated_nodes:
    #             keep.append(1)
    #         else:
    #             keep.append(0)
    #
    #     keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
    #
    #     return dec_man_enc_pred, gen_sdf, keep
    #
    # def decoder_with_changes(self, z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, distribution=None,gen_shape=False):
    #     # append zero nodes
    #     if distribution is not None:
    #         (mu, cov) = distribution
    #     nodes_added = []
    #     for i in range(len(missing_nodes)):
    #       ad_id = missing_nodes[i] + i
    #       nodes_added.append(ad_id)
    #       noise = np.zeros(self.embedding_dim) # np.random.normal(0, 1, 64)
    #       if distribution is not None:
    #         zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
    #       else:
    #         zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
    #       zeros.requires_grad = True
    #       zeros = zeros.float().cuda()
    #       z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)
    #
    #     # mark changes in nodes
    #     change_repr = []
    #     for i in range(len(z)):
    #         if i not in nodes_added and i not in manipulated_nodes:
    #             noisechange = np.zeros(self.embedding_dim)
    #         else:
    #             noisechange = np.random.normal(0, 1, self.embedding_dim)
    #         change_repr.append(torch.from_numpy(noisechange).float().cuda())
    #     change_repr = torch.stack(change_repr, dim=0)
    #     z_prime = torch.cat([z, change_repr], dim=1)
    #     z_prime = self.manipulate(z_prime, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
    #
    #     if not self.replace_all_latent:
    #         # take original nodes when untouched
    #         touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
    #         for touched_node in touched_nodes:
    #             z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
    #     else:
    #         z = z_prime
    #
    #     gen_sdf = None
    #     if gen_shape:
    #         un_rel_feat, rel_feat = self.encoder_2(z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
    #         sdf_candidates = dec_sdfs
    #         length = dec_objs.size(0)
    #         zeros_tensor = torch.zeros_like(sdf_candidates[0])
    #         mask = torch.ne(sdf_candidates, zeros_tensor)
    #         ids = torch.unique(torch.where(mask)[0])
    #         obj_selected = dec_objs[ids]
    #         diff_dict = {'sdf': dec_sdfs[ids], 'rel': rel_feat[ids], 'uc': un_rel_feat[ids]}
    #         gen_sdf = self.ShapeDiff.rel2shape(diff_dict,uc_scale=3.)
    #     dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat,
    #                                     attributes)
    #     if self.use_angles:
    #         num_dec_objs = len(dec_man_enc_pred[0])
    #     else:
    #         num_dec_objs = len(dec_man_enc_pred)
    #
    #     keep = []
    #     for i in range(num_dec_objs):
    #       if i not in nodes_added and i not in manipulated_nodes:
    #         keep.append(1)
    #       else:
    #         keep.append(0)
    #
    #     keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
    #
    #     return dec_man_enc_pred, gen_sdf, keep

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

    def select_sdfs_boxes(self, dec_objs_to_scene, obj_cats, obj_cats_grained, sdfs, boxes, angles, s_feat_ucon, s_feat_con, b_feat_ucon, b_feat_con, random=False):
        assert sdfs.shape[0] == boxes.shape[0] # should be the number of all objects
        dec_objs_to_scene = dec_objs_to_scene.detach().cpu().numpy()
        batch_size = np.max(dec_objs_to_scene) + 1
        scene_ids = []
        sdf_selected = []
        box_selected = []
        angle_selected = []
        uc_rel_s_selected = []
        c_rel_s_selected = []
        uc_rel_b_selected = []
        c_rel_b_selected = []
        obj_cat_selected = []
        num_obj = int(np.ceil(self.diffusion_bs / batch_size)) # how many objects should be picked in a scene
        for i in range(batch_size):
            # sdf, node classes, node fine_grained classes in the current scene
            sdf_candidates = sdfs[np.where(dec_objs_to_scene == i)[0]]
            box_candidates = boxes[np.where(dec_objs_to_scene == i)[0]]
            angle_candidates = angles[np.where(dec_objs_to_scene == i)[0]]
            obj_cat = obj_cats[np.where(dec_objs_to_scene == i)[0]]
            obj_cat_grained = obj_cats_grained[np.where(dec_objs_to_scene == i)[0]]

            # relation embeddings conditioning the shape diffusion in the current scene
            uc_rel_s = s_feat_ucon[np.where(dec_objs_to_scene == i)[0]]
            c_rel_s = s_feat_con[np.where(dec_objs_to_scene == i)[0]]

            # relation embeddings conditioning the layout diffusion in the current scene
            uc_rel_b = b_feat_ucon[np.where(dec_objs_to_scene == i)[0]]
            c_rel_b = b_feat_con[np.where(dec_objs_to_scene == i)[0]]

            length = obj_cat.size(0)
            zeros_tensor = torch.zeros_like(sdf_candidates[0])
            mask = torch.ne(sdf_candidates, zeros_tensor) # find out the objects which are not the floor
            object_inds = torch.unique(torch.where(mask)[0]) # non-floor non-__scene__ object indices (for shape branch)
            all_inds = torch.arange(box_candidates[:-1,:].shape[0]) # all bbox indices except __scene__(for layout branch)
            if random:
                # randomly choose num_obj elements for shape branch
                perm_obj = torch.randperm(len(object_inds))
                random_obj_inds = object_inds[perm_obj[:num_obj]]
                sdf_selected.append(sdf_candidates[random_obj_inds])
                uc_rel_s_selected.append(uc_rel_s[random_obj_inds])
                c_rel_s_selected.append(c_rel_s[random_obj_inds])
                obj_cat_selected.append(obj_cat[random_obj_inds])

                # randomly choose num_obj elements for layout branch
                perm = torch.randperm(len(all_inds))
                random_obj_inds = all_inds[perm[:num_obj]]
                box_selected.append(box_candidates[random_obj_inds])
                angle_selected.append(angle_candidates[random_obj_inds])
                uc_rel_b_selected.append(uc_rel_b[random_obj_inds])
                c_rel_b_selected.append(c_rel_b[random_obj_inds])
            else:
                ## balance every fine-grained category.
                # shape branch excludes the floor
                selected_inds = self.balance_objects(obj_cat_grained[object_inds], obj_cat[object_inds], num_obj, shape_branch=True) # selected_inds are indices of obj_cat_grained[object_inds]
                sdf_selected.append(sdf_candidates[object_inds][selected_inds])
                uc_rel_s_selected.append(uc_rel_s[object_inds][selected_inds])
                c_rel_s_selected.append(c_rel_s[object_inds][selected_inds])
                obj_cat_selected.append(obj_cat[object_inds][selected_inds])

                # layout branch includes the floor
                selected_inds = self.balance_objects(obj_cat_grained[all_inds], obj_cat[all_inds], num_obj, shape_branch=False) # selected_inds are indices of obj_cat_grained[all_inds]
                box_selected.append(box_candidates[all_inds][selected_inds])
                angle_selected.append(angle_candidates[all_inds][selected_inds])
                uc_rel_b_selected.append(uc_rel_b[all_inds][selected_inds])
                c_rel_b_selected.append(c_rel_b[all_inds][selected_inds])

                # essential_names = self.obj_classes_grained
            scene_ids.append(np.repeat(i,num_obj))

        sdf_selected = torch.cat(sdf_selected, dim=0).cuda()
        box_selected = torch.cat(box_selected, dim=0).cuda()
        angle_selected = torch.cat(angle_selected, dim=0).unsqueeze(1).cuda()
        box_selected = torch.cat((box_selected, angle_selected), dim=1)
        uc_rel_s_selected = torch.cat(uc_rel_s_selected, dim=0).cuda()
        c_rel_s_selected = torch.cat(c_rel_s_selected, dim=0).cuda()
        uc_rel_b_selected = torch.cat(uc_rel_b_selected, dim=0).cuda()
        c_rel_b_selected = torch.cat(c_rel_b_selected, dim=0).cuda()
        obj_cat_selected = torch.cat(obj_cat_selected, dim=0)
        scene_ids = np.concatenate(scene_ids, axis=0)
        diff_dict = {'sdf': sdf_selected[:self.diffusion_bs], 'uc_s': uc_rel_s_selected[:self.diffusion_bs],
                     'c_s': c_rel_s_selected[:self.diffusion_bs], 'box': box_selected[:self.diffusion_bs],
                     'uc_b': uc_rel_b_selected[:self.diffusion_bs], 'c_b': c_rel_b_selected[:self.diffusion_bs],
                     "scene_ids": scene_ids[:self.diffusion_bs]}
        return obj_cat_selected[:self.diffusion_bs], diff_dict

    def select_sdfs(self, dec_objs_to_scene, obj_cats, obj_cats_grained, sdfs, s_feat_ucon, s_feat_con, random=False):
        dec_objs_to_scene = dec_objs_to_scene.detach().cpu().numpy()
        batch_size = np.max(dec_objs_to_scene) + 1
        scene_ids = []
        sdf_selected = []
        uc_rel_s_selected = []
        c_rel_s_selected = []
        obj_cat_selected = []
        num_obj = int(np.ceil(self.diffusion_bs / batch_size)) # how many objects should be picked in a scene
        for i in range(batch_size):
            # sdf, node classes, node fine_grained classes in the current scene
            sdf_candidates = sdfs[np.where(dec_objs_to_scene == i)[0]]
            obj_cat = obj_cats[np.where(dec_objs_to_scene == i)[0]]
            obj_cat_grained = obj_cats_grained[np.where(dec_objs_to_scene == i)[0]]

            # relation embeddings conditioning the shape diffusion in the current scene
            uc_rel_s = s_feat_ucon[np.where(dec_objs_to_scene == i)[0]]
            c_rel_s = s_feat_con[np.where(dec_objs_to_scene == i)[0]]

            zeros_tensor = torch.zeros_like(sdf_candidates[0])
            mask = torch.ne(sdf_candidates, zeros_tensor) # find out the objects which are not the floor
            object_inds = torch.unique(torch.where(mask)[0]) # non-floor non-__scene__ object indices (for shape branch)
            if random:
                # randomly choose num_obj elements for shape branch
                perm_obj = torch.randperm(len(object_inds))
                random_obj_inds = object_inds[perm_obj[:num_obj]]
                sdf_selected.append(sdf_candidates[random_obj_inds])
                uc_rel_s_selected.append(uc_rel_s[random_obj_inds])
                c_rel_s_selected.append(c_rel_s[random_obj_inds])
                obj_cat_selected.append(obj_cat[random_obj_inds])
            else:
                ## balance every fine-grained category.
                # shape branch excludes the floor
                selected_inds = self.balance_objects(obj_cat_grained[object_inds], obj_cat[object_inds], num_obj, shape_branch=True) # selected_inds are indices of obj_cat_grained[object_inds]
                sdf_selected.append(sdf_candidates[object_inds][selected_inds])
                uc_rel_s_selected.append(uc_rel_s[object_inds][selected_inds])
                c_rel_s_selected.append(c_rel_s[object_inds][selected_inds])
                obj_cat_selected.append(obj_cat[object_inds][selected_inds])
            scene_ids.append(np.repeat(i,num_obj))

        sdf_selected = torch.cat(sdf_selected, dim=0).cuda()
        uc_rel_s_selected = torch.cat(uc_rel_s_selected, dim=0).cuda()
        c_rel_s_selected = torch.cat(c_rel_s_selected, dim=0).cuda()
        obj_cat_selected = torch.cat(obj_cat_selected, dim=0)
        scene_ids = np.concatenate(scene_ids, axis=0)
        diff_dict = {'sdf': sdf_selected[:self.diffusion_bs], 'uc_s': uc_rel_s_selected[:self.diffusion_bs],
                     'c_s': c_rel_s_selected[:self.diffusion_bs], "scene_ids": scene_ids[:self.diffusion_bs]}
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
        obj_embed_, pred_embed_ , latent_obj_vecs_, pred_vecs_ = self.manipulate(latent_obj_vecs_, dec_objs, dec_triples, dec_text_feat, dec_rel_feat) # contains all obj now

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                latent_obj_vecs = torch.cat([latent_obj_vecs[:touched_node], latent_obj_vecs_[touched_node:touched_node + 1], latent_obj_vecs[touched_node + 1:]], dim=0)
        else:
            latent_obj_vecs = latent_obj_vecs_

        # # relation embeddings -> diffusion
        c_rel_feat_s = latent_obj_vecs
        # c_rel_feat_b = c_rel_feat_s = latent_obj_vecs
        # if self.s_l_separated:
        #     c_rel_feat_s, _ = self.shape_encoder(latent_obj_vecs, obj_embed_, pred_embed_, dec_triples)
        #     c_rel_feat_b, _ = self.layout_encoder(latent_obj_vecs, obj_embed_, pred_embed_, dec_triples)
        uc_rel_feat_s = self.rel_s_mlp(obj_embed_) # embedding + CLIP
        uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)
        # uc_rel_feat_b = self.rel_l_mlp(obj_embed_)
        # uc_rel_feat_b = torch.unsqueeze(uc_rel_feat_b, dim=1)
        #
        c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
        c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)
        # c_rel_feat_b = self.rel_l_mlp(c_rel_feat_b)
        # c_rel_feat_b = torch.unsqueeze(c_rel_feat_b, dim=1)

        obj_selected, shape_diff_dict = self.select_sdfs(dec_objs_to_scene, dec_objs, dec_objs_grained, dec_sdfs, uc_rel_feat_s, c_rel_feat_s, random=False)
        box_diff_dict = self.prepare_boxes(dec_triples, obj_embed_, obj_boxes=dec_boxes, obj_angles=dec_angles, relation_cond=latent_obj_vecs, scene_ids=dec_objs_to_scene)

        self.ShapeDiff.set_input(shape_diff_dict)
        self.ShapeDiff.set_requires_grad([self.ShapeDiff.df], requires_grad=True)
        Shape_loss, Shape_loss_dict = self.ShapeDiff.forward()

        self.LayoutDiff.set_input(box_diff_dict)
        self.LayoutDiff.set_requires_grad([self.LayoutDiff.df], requires_grad=True)
        Layout_loss, Layout_loss_dict = self.LayoutDiff.forward()


        loss_dict = {**Shape_loss_dict, **Layout_loss_dict}

        return obj_selected, Shape_loss, Layout_loss, loss_dict

    # def forward_no_mani(self, objs, triples, enc, attributes):
    #     mu, logvar = self.encoder(objs, triples, enc, attributes)
    #     # reparameterization
    #     std = torch.exp(0.5 * logvar)
    #     # standard sampling
    #     eps = torch.randn_like(std)
    #     z = eps.mul(std).add_(mu)
    #     keep = []
    #     dec_man_enc_pred = self.decoder(z, objs, triples, attributes)
    #     for i in range(len(dec_man_enc_pred)):
    #         keep.append(1)
    #     keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
    #     return mu, logvar, dec_man_enc_pred, keep

    def sample(self, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=False):
        with torch.no_grad():
            obj_embed, pred_embed, latent_obj_vecs, latent_pred_vecs = self.init_encoder(dec_objs, dec_triplets,
                                                                                         encoded_dec_text_feat,
                                                                                         encoded_dec_rel_feat)
            change_repr = []
            for i in range(len(latent_obj_vecs)):
                noisechange = np.zeros(self.embedding_dim)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
            change_repr = torch.stack(change_repr, dim=0)
            latent_obj_vecs_ = torch.cat([latent_obj_vecs, change_repr], dim=1)
            latent_obj_vecs_, pred_vecs_, obj_embed_, pred_embed_ = self.manipulate(latent_obj_vecs_, dec_objs,
                                                                                    dec_triplets, encoded_dec_text_feat,
                                                                                    encoded_dec_rel_feat)  # normal message passing

            ## relation embeddings -> diffusion
            # c_rel_feat_b = latent_obj_vecs_
            # if self.s_l_separated:
            #     c_rel_feat_b, _ = self.layout_encoder(latent_obj_vecs_, obj_embed_, pred_embed_, dec_triplets)
            # uc_rel_feat_b = self.rel_l_mlp(obj_embed_)
            # uc_rel_feat_b = torch.unsqueeze(uc_rel_feat_b, dim=1)
            #
            # c_rel_feat_b = self.rel_l_mlp(c_rel_feat_b)
            # c_rel_feat_b = torch.unsqueeze(c_rel_feat_b, dim=1)

            diff_dict = self.prepare_boxes(dec_triplets, obj_embed_, relation_cond=latent_obj_vecs_)

            self.LayoutDiff.set_input(diff_dict)
            gen_box_dict = self.LayoutDiff.generate_layout_sg(box_dim=self.diff_cfg.layout_branch.denoiser_kwargs.in_channels)

            if gen_shape:
                # # relation embeddings -> diffusion
                c_rel_feat_s = latent_obj_vecs
                uc_rel_feat_s = self.rel_s_mlp(obj_embed_)  # embedding + CLIP
                uc_rel_feat_s = torch.unsqueeze(uc_rel_feat_s, dim=1)
                c_rel_feat_s = self.rel_s_mlp(c_rel_feat_s)
                c_rel_feat_s = torch.unsqueeze(c_rel_feat_s, dim=1)
                sdf_candidates = dec_sdfs  # just use it to filter out floor and _scene_ (if have)
                length = dec_objs.size(0)
                zeros_tensor = torch.zeros_like(sdf_candidates[0])
                mask = torch.ne(sdf_candidates, zeros_tensor)
                ids = torch.unique(torch.where(mask)[0])
                diff_dict = {'obj_cat': dec_objs[ids], 'c_s': c_rel_feat_s[ids], 'uc_s': uc_rel_feat_s[ids]}
                gen_sdf = self.ShapeDiff.rel2shape(diff_dict, uc_scale=3.)

            return {'shapes': gen_sdf}, gen_box_dict

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
    def collect_train_statistics(self, train_loader, with_points=False):
        # model = model.eval()
        mean_cat = None
        if with_points:
            means, vars = {}, {}
            for idx in train_loader.dataset.point_classes_idx:
                means[idx] = []
                vars[idx] = []
            means[-1] = []
            vars[-1] = []

        for idx, data in enumerate(train_loader):
            if data == -1:
                continue
            try:
                objs, triples, tight_boxes, objs_to_scene, triples_to_scene = data['decoder']['objs'], \
                                                                              data['decoder']['tripltes'], \
                                                                              data['decoder']['boxes'], \
                                                                              data['decoder']['obj_to_scene'], \
                                                                              data['decoder']['triple_to_scene']

                enc_text_feat, enc_rel_feat = None, None
                if 'feats' in data['decoder']:
                    encoded_points = data['decoder']['feats']
                    encoded_points = encoded_points.cuda()
                if 'text_feats' in data['decoder'] and 'rel_feats' in data['decoder']:
                    enc_text_feat, enc_rel_feat = data['decoder']['text_feats'], data['decoder']['rel_feats']
                    enc_text_feat, enc_rel_feat = enc_text_feat.cuda(), enc_rel_feat.cuda()

            except Exception as e:
                print('Exception', str(e))
                continue

            objs, triples, tight_boxes = objs.cuda(), triples.cuda(), tight_boxes.cuda()
            boxes = tight_boxes[:, :6]
            angles = tight_boxes[:, 6].long() - 1
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
            attributes = None


            mean, logvar = self.encoder(objs, triples, boxes, attributes, enc_text_feat, enc_rel_feat, angles)
            mean, logvar = mean.cpu().clone(), logvar.cpu().clone()

            mean = mean.data.cpu().clone()
            if mean_cat is None:
                mean_cat = mean
            else:
                mean_cat = torch.cat([mean_cat, mean], dim=0)

        mean_est = torch.mean(mean_cat, dim=0, keepdim=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        cov_est_ = np.cov(mean_cat.numpy().T)
        n = mean_cat.size(0)
        d = mean_cat.size(1)
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i].numpy()
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est_
