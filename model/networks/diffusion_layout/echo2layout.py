import torch
from torch.nn import Module

from .diffusion_ddpm import DiffusionPoint
from .denoise_net import UNet1DModel

class EchoToLayout(Module):

    def __init__(self, config, n_classes=None):
        super().__init__()
        self.device = config.hyper.device
        self.rel_condition = config.layout_branch.relation_condition
        # define the denoising network
        if config.layout_branch.denoiser == "unet1d":
            denoise_net = UNet1DModel(**config.layout_branch.denoiser_kwargs)
        else:
            raise NotImplementedError()

        # define the diffusion type
        self.df = DiffusionPoint(
            denoise_net = denoise_net,
            config = config.layout_branch,
            **config.layout_branch.diffusion_kwargs
        )
        self.n_classes = n_classes # not used
        self.config = config
        
        # read object property dimension
        self.translation_dim = config.layout_branch.get("translation_dim", 3)
        self.size_dim = config.layout_branch.get("size_dim", 3)
        self.angle_dim = config.layout_branch.angle_dim
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim

        # param list
        trainable_models = [self.df]
        trainable_params = []
        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]
        self.trainable_params = trainable_params

        self.df.to(self.device)
        self.scene_ids=None

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, data_dict):
        vars_list = []
        try:
            self.x = data_dict['box']
            self.scene_ids = data_dict['obj_id_to_scene']
            B, D = self.x.shape
            vars_list.append('x')
        except:
            print('inference mode, no gt boxes and scene ids')

        self.preds = data_dict['preds']
        self.rel = data_dict['c_b']
        self.uc_rel = data_dict['uc_b']
        vars_list += ['preds', 'rel', 'uc_rel']
        self.tocuda(var_names=vars_list)

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(self.device, non_blocking=True))

    def forward(self):
        self.df.train()
        rel = self.rel
        obj_embed = self.uc_rel
        target_box = self.x
        triples = self.preds

        # Compute the loss
        self.loss, self.loss_dict = self.get_loss(obj_embed=obj_embed, obj_triples=triples, target_box=target_box, rel=rel)
        return self.loss, self.loss_dict

    def get_loss(self, obj_embed, obj_triples, target_box, rel):
        # Unpack the sample_params
        batch_size, D_params = target_box.shape
        if self.rel_condition:
            condition_cross = rel # use rel embed for cross attention
        else:
            raise NotImplementedError

        loss, loss_dict = self.df.get_loss_iter(obj_embed, obj_triples, target_box, scene_ids=self.scene_ids, condition_cross=condition_cross)

        return loss, loss_dict

    def sample(self, box_dim, batch_size, obj_embed=None, obj_triples=None, text=None, rel=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None):

        noise_shape = (batch_size, box_dim)
        condition = rel if self.rel_condition else None
        condition_cross = None
        # reverse sampling
        samples = self.df.gen_samples_sg(noise_shape, obj_embed.device, obj_embed, obj_triples, condition=condition, clip_denoised=clip_denoised)
        
        return samples

    @torch.no_grad()
    def generate_layout_sg(self, box_dim, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None):

        rel = self.rel
        obj_embed = self.uc_rel
        triples = self.preds

        samples = self.sample(box_dim, batch_size=len(obj_embed), obj_embed=obj_embed, obj_triples=triples, text=text, rel=rel, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)
        samples_dict = {
            "sizes": samples[:, 0:self.size_dim].contiguous(),
            "translations": samples[:, self.size_dim:self.size_dim + self.translation_dim].contiguous(),
            "angles": samples[:, self.size_dim + self.translation_dim:self.bbox_dim].contiguous(),
        }
        
        return samples_dict