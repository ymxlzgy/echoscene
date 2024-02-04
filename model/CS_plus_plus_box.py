import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import GraphTripleConvNet, _init_weights, make_mlp
import numpy as np


class Sg2ScVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a separate embedding of shape and bounding box latents.
    """
    def __init__(self, vocab, embedding_dim=128, batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 input_dim=6,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 use_AE=False,
                 replace_latent=False,
                 residual=False,
                 use_angles=False):
        super(Sg2ScVAEModel, self).__init__()
        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        box_embedding_dim = int(embedding_dim)
        if use_angles:
            angle_embedding_dim = int(embedding_dim / 4)
            box_embedding_dim = int(embedding_dim * 3 / 4)
            Nangle = 24
        obj_embedding_dim = embedding_dim

        self.replace_all_latent = replace_latent
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        self.use_AE = use_AE
        self.use_angles = use_angles

        num_objs = len(list(set(vocab['object_idx_to_name'])))
        num_preds = len(list(set(vocab['pred_idx_to_name'])))

        # build encoder and decoder nets
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec = nn.Embedding(num_preds, embedding_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim)
        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
            self.pred_embeddings_man_dc = nn.Embedding(num_preds, embedding_dim * 3)
        self.d3_embeddings = nn.Linear(input_dim, box_embedding_dim)
        if self.use_angles:
            self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)
        # weight sharing of mean and var
        self.mean_var = make_mlp([embedding_dim * 2 + 512, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        if self.use_angles:
            self.angle_mean_var = make_mlp([embedding_dim * 2 + 512, gconv_hidden_dim, embedding_dim * 2],
                                           batch_norm=mlp_normalization)
            self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
            self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net
        self.gconv_net_ec = None
        self.gconv_net_dc = None

        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2 + 512,
            'input_dim_pred': gconv_dim * 2 + 512,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim + 512,
            'input_dim_pred': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_manipulation = {
            'input_dim_obj': embedding_dim * 3 + 512,
            'input_dim_pred': embedding_dim * 3 + 512,
            'hidden_dim': gconv_hidden_dim,
            'output_dim': embedding_dim,
            'pooling': gconv_pooling,
            'num_layers': min(gconv_num_layers, 5),
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 2 + 512
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 2 + 512

        self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_dc = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        net_layers = [gconv_dim * 2 + 512, gconv_hidden_dim, input_dim]
        self.d3_net = make_mlp(net_layers, batch_norm=mlp_normalization, norelu=True)

        if self.use_angles:
            # angle prediction net
            angle_net_layers = [gconv_dim * 2 + 512, gconv_hidden_dim, Nangle]
            self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.d3_embeddings.apply(_init_weights)
        self.mean_var.apply(_init_weights)
        self.mean.apply(_init_weights)
        self.var.apply(_init_weights)
        self.d3_net.apply(_init_weights)

        if self.use_angles:
            self.angle_mean_var.apply(_init_weights)
            self.angle_mean.apply(_init_weights)
            self.angle_var.apply(_init_weights)

    def encoder(self, objs, triples, boxes_gt, attributes, enc_text_feat, enc_rel_feat, angles_gt=None):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_ec(objs)
        pred_vecs = self.pred_embeddings_ec(p)
        d3_vecs = self.d3_embeddings(boxes_gt)
        obj_vecs_ = torch.cat([enc_text_feat, obj_vecs], dim=1)
        pred_vecs_ = torch.cat([enc_rel_feat, pred_vecs], dim=1)
        if self.use_angles:
            angle_vecs = self.angle_embeddings(angles_gt)
            obj_vecs_ = torch.cat([obj_vecs_, d3_vecs, angle_vecs], dim=1)
        else:
            obj_vecs_ = torch.cat([obj_vecs_, d3_vecs], dim=1)

        if self.gconv_net_ec is not None:
            obj_vecs_, pred_vecs_ = self.gconv_net_ec(obj_vecs_, pred_vecs_, edges)

        obj_vecs_3d = self.mean_var(obj_vecs_)
        mu = self.mean(obj_vecs_3d)
        logvar = self.var(obj_vecs_3d)

        if self.use_angles:
            obj_vecs_angle = self.angle_mean_var(obj_vecs_)
            mu_angle = self.angle_mean(obj_vecs_angle)
            logvar_angle = self.angle_var(obj_vecs_angle)
            mu = torch.cat([mu, mu_angle], dim=1)
            logvar = torch.cat([logvar, logvar_angle], dim=1)

        return mu, logvar

    def manipulate(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_man_dc(p)
        obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
        pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)

        man_z = torch.cat([z, obj_vecs_], dim=1)
        man_z, _ = self.gconv_net_manipulation(man_z, pred_vecs_, edges)

        return man_z

    def decoder(self, z, objs, triples, dec_text_feat, dec_rel_feat, attributes, manipulate=False):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        pred_vecs = self.pred_embeddings_dc(p)
        obj_vecs_ = torch.cat([dec_text_feat, obj_vecs], dim=1)
        pred_vecs_ = torch.cat([dec_rel_feat, pred_vecs], dim=1)

        # concatenate noise first
        if self.decoder_cat:
            obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)
            obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)

        # concatenate noise after gconv
        else:
            obj_vecs_, pred_vecs_ = self.gconv_net_dc(obj_vecs_, pred_vecs_, edges)
            obj_vecs_ = torch.cat([obj_vecs_, z], dim=1)

        d3_pred = self.d3_net(obj_vecs_)
        if self.use_angles:
            angles_pred = F.log_softmax(self.angle_net(obj_vecs_), dim=1)
            return d3_pred, angles_pred
        else:
            return d3_pred
    '''
    def decoder_with_changes_placebo(self, z, dec_objs, dec_triples, attributes, missing_nodes, manipulated_nodes):
        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)
            noise = np.zeros(self.embedding_dim)  # np.random.normal(0, 1, 64)
            zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim))
            zeros.requires_grad = True
            zeros = zeros.float().cuda()
            z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())

        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, attributes)

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime
        dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, attributes)

        keep = []
        for i in range(len(dec_man_enc_pred)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_pred, keep
    '''
    def decoder_with_additions(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes, distribution=None):
        nodes_added = []
        if distribution is not None:
            mu, cov = distribution

        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)
            noise = np.zeros(z.shape[1])  # np.random.normal(0, 1, 64)
            if distribution is not None:
                zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
            else:
                zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
            zeros.requires_grad = True
            zeros = zeros.float().cuda()
            z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        keep = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return self.decoder(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes), keep
    '''
    def get_keetps(self, dec_objs, missing_nodes, manipulated_nodes):
        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)

        keep = []
        for i in range(len(dec_objs)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        return keep
    '''
    def decoder_with_changes(self, z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes, distribution=None):
        # append zero nodes
        if distribution is not None:
            (mu, cov) = distribution
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim) # np.random.normal(0, 1, 64)
          if distribution is not None:
            zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
          else:
            zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        if self.use_angles:
            num_dec_objs = len(dec_man_enc_pred[0])
        else:
            num_dec_objs = len(dec_man_enc_pred)

        keep = []
        for i in range(num_dec_objs):
          if i not in nodes_added and i not in manipulated_nodes:
            keep.append(1)
          else:
            keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_pred, keep

    def forward(self, enc_objs, enc_triples, enc, enc_text_feat, enc_rel_feat, attributes, enc_objs_to_scene, dec_objs,
                dec_triples, dec, dec_text_feat, dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes,
                enc_angles=None, dec_angles=None):

        mu, logvar = self.encoder(enc_objs, enc_triples, enc, attributes, enc_text_feat, enc_rel_feat, enc_angles)
        if self.use_AE:
            z = mu
        else:
            # reparameterization
            std = torch.exp(0.5*logvar)
            # standard sampling
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)

        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim) # np.random.normal(0, 1, 64)
          zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, dec_text_feat, dec_rel_feat, attributes)

        if not self.replace_all_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        if self.use_angles:
            dec_man_enc_pred, angles_pred = self.decoder(z, dec_objs, dec_triples, dec_text_feat, dec_rel_feat, attributes)
            orig_angles = []
            orig_gt_angles = []
        else:
            dec_man_enc_pred = self.decoder(z, dec_objs, dec_triples, attributes)

        orig_d3 = []
        orig_gt_d3 = []
        keep = []
        for i in range(len(dec_man_enc_pred)):
          if i not in nodes_added and i not in manipulated_nodes:
            orig_d3.append(dec_man_enc_pred[i:i+1])
            orig_gt_d3.append(dec[i:i+1])
            if self.use_angles:
                orig_angles.append(angles_pred[i:i+1])
                orig_gt_angles.append(dec_angles[i:i+1])
            keep.append(1)
          else:
            keep.append(0)

        orig_d3 = torch.cat(orig_d3, dim=0)
        orig_gt_d3 = torch.cat(orig_gt_d3, dim=0)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        if self.use_angles:
            orig_angles = torch.cat(orig_angles, dim=0)
            orig_gt_angles = torch.cat(orig_gt_angles, dim=0)
        else:
            orig_angles, orig_gt_angles, angles_pred = None, None, None

        return mu, logvar, orig_gt_d3, orig_gt_angles, orig_d3, orig_angles, dec_man_enc_pred, angles_pred, keep

    def forward_no_mani(self, objs, triples, enc, attributes):
        mu, logvar = self.encoder(objs, triples, enc, attributes)
        # reparameterization
        std = torch.exp(0.5 * logvar)
        # standard sampling
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        keep = []
        dec_man_enc_pred = self.decoder(z, objs, triples, attributes)
        for i in range(len(dec_man_enc_pred)):
            keep.append(1)
        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()
        return mu, logvar, dec_man_enc_pred, keep

    def sampleShape(self, point_classes_idx, point_ae, mean_est_shape, cov_est_shape, dec_objs, dec_triplets,
                    attributes=None):
        with torch.no_grad():
            z_shape = []
            for idxz in dec_objs:
                idxz = int(idxz.cpu())
                if idxz in point_classes_idx:
                    z_shape.append(torch.from_numpy(
                        np.random.multivariate_normal(mean_est_shape[idxz], cov_est_shape[idxz], 1)).float().cuda())
                else:
                    z_shape.append(torch.from_numpy(np.random.multivariate_normal(mean_est_shape[-1],
                                                                                  cov_est_shape[-1],
                                                                                  1)).float().cuda())
            z_shape = torch.cat(z_shape, 0)

            dc_shapes = self.decoder(z_shape, dec_objs, dec_triplets, attributes)
            points = point_ae.forward_inference_from_latent_space(dc_shapes, point_ae.get_grid())
        return points, dc_shapes

    def sampleBoxes(self, mean_est, cov_est, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None):
        with torch.no_grad():
            z = torch.from_numpy(
            np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()

            return self.decoder(z, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)

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

                if 'feats' in data['decoder']:
                    encoded_points = data['decoder']['feats']
                    encoded_points = encoded_points.cuda()
                enc_text_feat, enc_rel_feat=None,None
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
            if with_points:
                for i in range(len(objs)):
                    if objs[i] in train_loader.dataset.point_classes_idx:
                        means[int(objs[i].cpu())].append(mean[i].detach().cpu().numpy())
                        vars[int(objs[i].cpu())].append(logvar[i].detach().cpu().numpy())
                    else:
                        means[-1].append(mean[i].detach().cpu().numpy())
                        vars[-1].append(logvar[i].detach().cpu().numpy())
            else:
                if mean_cat is None:
                    mean_cat = mean
                else:
                    mean_cat = torch.cat([mean_cat, mean], dim=0)

        if with_points:
            for idx in train_loader.dataset.point_classes_idx + [-1]:
                if len(means[idx]) < 3:
                    means[idx] = np.zeros(128)
                    vars[idx] = np.eye(128)
                else:
                    mean_cat = np.stack(means[idx], 0)
                    mean_est = np.mean(mean_cat, axis=0, keepdims=True)  # size 1*embed_dim
                    mean_cat = mean_cat - mean_est
                    n = mean_cat.shape[0]
                    d = mean_cat.shape[1]
                    cov_est = np.zeros((d, d))
                    for i in range(n):
                        x = mean_cat[i]
                        cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
                    mean_est = mean_est[0]
                    means[idx] = mean_est
                    vars[idx] = cov_est
            return means, vars
        else:
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
