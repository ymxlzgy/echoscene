from curses import noecho
from doctest import debug_script
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from .diffusion_ddpm import DiffusionPoint
from .denoise_net import UNet1DModel
import clip

class DiffusionSceneLayout_DDPM(Module):

    def __init__(self, config, edge_classes):
        super().__init__()
        self.device = config.hyper.device
        self.text_condition = config.layout_branch.get("text_condition", False)
        self.text_clip_embedding = config.layout_branch.get("text_clip_embedding", False)
        if self.text_condition:
            text_embed_dim = config.get("text_embed_dim", 512)
            if self.text_glove_embedding:
                self.fc_text_f = nn.Linear(50, text_embed_dim)
                print('use text as condition, and pretrained glove embedding')
            elif self.text_clip_embedding:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                for p in self.clip_model.parameters():
                    p.requires_grad = False
                print('use text as condition, and pretrained clip embedding')
            else:
                raise TypeError

        self.rel_condition = config.layout_branch.relation_condition
        # define the denoising network
        if config.layout_branch.denoiser == "unet1d":
            config.layout_branch.denoiser_kwargs.edge_classes = edge_classes
            denoise_net = UNet1DModel(**config.layout_branch.denoiser_kwargs)
        else:
            raise NotImplementedError()

        # define the diffusion type
        self.df = DiffusionPoint(
            denoise_net = denoise_net,
            config = config.layout_branch,
            **config.layout_branch.diffusion_kwargs
        )
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
        if self.text_condition:
            raise NotImplementedError # use text embed for cross attention
        elif self.rel_condition:
            condition_cross = rel # use rel embed for cross attention
        else:
            raise NotImplementedError

        loss, loss_dict = self.df.get_loss_iter(obj_embed, obj_triples, target_box, scene_ids=self.scene_ids, condition_cross=condition_cross)

        return loss, loss_dict

    def sample(self, box_dim, batch_size, obj_embed=None, obj_triples=None, text=None, rel=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None):

        noise_shape = (batch_size, box_dim)
        condition = rel if self.rel_condition else None

        if self.text_condition:
            if self.text_glove_embedding:
                condition_cross = self.fc_text_f(text) #sample_params["desc_emb"]
            elif self.text_clip_embedding:
                tokenized = clip.tokenize(text).to(device)
                condition_cross = self.clip_model.encode_text(tokenized)
            else:
                tokenized = self.tokenizer(text, return_tensors='pt',padding=True).to(device)
                #print('tokenized:', tokenized.shape)
                text_f = self.bertmodel(**tokenized).last_hidden_state
                print('after bert:', text_f.shape)
                condition_cross = self.fc_text_f( text_f )
        else:
            condition_cross = None
        # reverse sampling
        if ret_traj:
            samples = self.df.gen_sample_traj_sg(noise_shape, obj_embed.device, obj_embed, obj_triples, freq=freq, condition=condition, clip_denoised=clip_denoised)
        else:
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

    # @torch.no_grad()
    # def generate_layout_progressive(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False, num_step=100):
    #
    #     # output dictionary of sample trajectory & sample some key steps
    #     samples_traj = self.sample(room_mask, num_points, point_dim, batch_size, text=text, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds, freq=num_step)
    #     boxes_traj = {}
    #
    #     # delete the initial noisy
    #     samples_traj = samples_traj[1:]
    #
    #     for i in range(len(samples_traj)):
    #         samples = samples_traj[i]
    #         k_time = num_step * i
    #         boxes_traj[k_time] = self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)
    #     return boxes_traj
    

    @torch.no_grad()
    def delete_empty_from_network_samples(self, samples, device="cpu", keep_empty=False):
        
        samples_dict = {
            "sizes": samples[:, :, 0:self.size_dim].contiguous(),
            "translations": samples[:, :,  self.size_dim:self.size_dim+self.translation_dim].contiguous(),
            "angles": samples[:, :, self.size_dim+self.translation_dim:self.bbox_dim].contiguous(),
        }

        #initilization
        boxes = {
            "translations": torch.zeros(1, 0, self.translation_dim, device=device),
            "sizes": torch.zeros(1, 0, self.size_dim, device=device),
            "angles": torch.zeros(1, 0, self.angle_dim, device=device)
        }
        if self.objfeat_dim > 0:
            boxes["objfeats"] =  torch.zeros(1, 0, self.objfeat_dim, device=device)
    
        max_boxes = samples.shape[1]
        for i in range(max_boxes):
            # Check if we have the end symbol 
            if not keep_empty:
                continue
            else:
                for k in samples_dict.keys():
                    if k == "class_labels":
                        # we output raw probability maps for visualization
                        boxes[k] = torch.cat([ boxes[k], samples[:, i:i+1, self.bbox_dim:self.bbox_dim+self.class_dim-1].to(device) ], dim=1)
                    else:
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :].to(device) ], dim=1)

        if self.objfeat_dim > 0:
            return {
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu"),
        }
        else:
            return {
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu")
            }


    @torch.no_grad()
    def delete_empty_boxes(self, samples_dict, device="cpu", keep_empty=False):

        #initilization
        boxes = {
            "translations": torch.zeros(1, 0, self.translation_dim, device=device),
            "sizes": torch.zeros(1, 0, self.size_dim, device=device),
            "angles": torch.zeros(1, 0, self.angle_dim, device=device)
        }
        if self.objfeat_dim > 0:
            boxes["objfeats"] =  torch.zeros(1, 0, self.objfeat_dim, device=device)
    
        max_boxes = samples_dict["class_labels"].shape[1]
        for i in range(max_boxes):
            # Check if we have the end symbol 
            if not keep_empty and samples_dict['class_labels'][0, i, -1] > 0:
                continue
            else:
                for k in samples_dict.keys():
                    if k == "class_labels":
                        # we output raw probability maps for visualization
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :self.class_dim-1].to(device) ], dim=1)
                    else:
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :].to(device) ], dim=1)

        
        if self.objfeat_dim > 0:
                return {
                "class_labels": boxes["class_labels"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu"),
                "objfeats": boxes["objfeats"].to("cpu"),
            }
        else:
            return {
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu"),
            }


@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    return loss.item()
