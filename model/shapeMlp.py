import torch
import torch.nn as nn
import numpy as np


class ShapeMLP(nn.Module):
    def __init__(self, num_objs,
                 embedding_dim=128):
        super(ShapeMLP, self).__init__()

        self.obj_emb = nn.Embedding(num_objs + 1, 16)
        self.obj_emb_de = nn.Embedding(num_objs + 1, 16)

        self.l1 = nn.Linear(128 + 16, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.l2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.l3 = nn.Linear(32, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.l3_mu = nn.Linear(embedding_dim, embedding_dim)
        self.l3_logvar = nn.Linear(embedding_dim, embedding_dim)

        self.l32 = nn.Linear(embedding_dim + 16, 32)
        self.bn22 = nn.BatchNorm1d(32)
        self.l22 = nn.Linear(32, 16)
        self.bn12 = nn.BatchNorm1d(16)
        self.l12 = nn.Linear(16, 128)


    def encoder(self, objs, shapes_gt):
        obj_vecs = self.obj_emb(objs)
        obj_vecs = torch.cat([obj_vecs, shapes_gt], dim=1)

        obj_vecs = self.bn1(torch.relu(self.l1(obj_vecs)))
        obj_vecs = self.bn2(torch.relu(self.l2(obj_vecs)))
        obj_vecs = self.bn3(torch.relu(self.l3(obj_vecs)))
        mu = self.l3_mu(obj_vecs)
        logvar = self.l3_logvar(obj_vecs)
        return mu, logvar

    def decoder_with_additions(self, z, objs, missing_nodes, manipulated_nodes, distribution=None):
        nodes_added = []
        if distribution is not None:
            mu, cov = distribution

        for i in range(len(missing_nodes)):
            ad_id = missing_nodes[i] + i
            nodes_added.append(ad_id)
            noise = np.zeros(z.shape[1])  # np.random.normal(0, 1, 64)
            if distribution is not None:
                if objs[i] in mu:
                    zeros = torch.from_numpy(np.random.multivariate_normal(mu[objs[i]], cov[objs[i]], 1)).float().cuda()
                else:
                    zeros = torch.from_numpy(np.random.multivariate_normal(mu[-1], cov[-1], 1)).float().cuda()
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

        return self.decoder(z, objs), keep

    def decoder(self, z, objs):

        obj_vecs = self.obj_emb_de(objs)
        obj_vecs = torch.cat([obj_vecs, z], dim=1)
        obj_vecs = self.bn22(torch.relu(self.l32(obj_vecs)))
        obj_vecs = self.bn12(torch.relu(self.l22(obj_vecs)))
        pred = self.l12(obj_vecs)
        return pred

    def forward(self, objs, shapes):

        mu, logvar = self.encoder(objs, shapes)

        # reparameterization
        std = torch.exp(0.5*logvar)
        # standard sampling
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        pred = self.decoder(z, objs)

        keep = []
        for i in range(len(objs)):
            keep.append(1)


        return mu, logvar, pred, keep

    def sampleShape(self, point_classes_idx, dec_objs, point_ae, mean_est_shape, cov_est_shape):
        with torch.no_grad():
            z_shape = []
            for idxz in dec_objs:
                idxz = int(idxz.cpu())
                if idxz in point_classes_idx:
                    z_shape.append(torch.from_numpy(
                        np.random.multivariate_normal(mean_est_shape[idxz], cov_est_shape[idxz],
                                                      1)).float().cuda())
                else:
                    z_shape.append(torch.from_numpy(np.random.multivariate_normal(mean_est_shape[-1],
                                                                                  cov_est_shape[-1],
                                                                                  1)).float().cuda())
            z_shape = torch.cat(z_shape, 0)

            dc_shapes = self.decoder(z_shape, dec_objs)
            points = point_ae.forward_inference_from_latent_space(dc_shapes, point_ae.get_grid())
        return points, dc_shapes

    def collect_train_statistics(self, train_loader):
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
                objs, triples, tight_boxes,  objs_to_scene, triples_to_scene = data['encoder']['objs'], \
                                                                               data['encoder']['tripltes'], \
                                                                               data['encoder']['boxes'], \
                                                                               data['encoder']['obj_to_scene'], \
                                                                               data['encoder']['triple_to_scene']

                encoded_points = data['encoder']['feats']
                encoded_points = encoded_points.cuda()

            except Exception as e:
                print('Exception', str(e))
                continue

            objs, triples, tight_boxes = objs.cuda(), triples.cuda(), tight_boxes.cuda()
            boxes = tight_boxes[:, :6]

            mask = [ob in train_loader.dataset.point_classes_idx for ob in objs]
            if sum(mask) <= 0:
                continue

            mean, logvar = self.encoder(objs, encoded_points)
            mean, logvar = mean, logvar
            mean = mean.data.cpu().clone()
            for i in range(len(objs)):
                if objs[i] in train_loader.dataset.point_classes_idx:
                    means[int(objs[i].cpu())].append(mean[i].detach().cpu().numpy())
                    vars[int(objs[i].cpu())].append(logvar[i].detach().cpu().numpy())
                else:
                    means[-1].append(mean[i].detach().cpu().numpy())
                    vars[-1].append(logvar[i].detach().cpu().numpy())

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
