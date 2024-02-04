from curses import noecho
from doctest import debug_script
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from .diffusion_ddpm import DiffusionPoint
from .denoise_net import Unet1D
import clip

class DiffusionSceneLayout_DDPM(Module):

    def __init__(self, config, n_classes=None):
        super().__init__()

        # TODO: Add the projection dimensions for the room features in the
        # config!!!
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

        self.rel_condition = config.layout_branch.denoiser_kwargs.relation_condition
        # define the denoising network
        if config.layout_branch.denoiser == "unet1d":
            denoise_net = Unet1D(**config.layout_branch.denoiser_kwargs)
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
        self.angle_dim = config.layout_branch.get("angle_dim", 1)
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

    def set_input(self, data_dict, max_sample):
        self.x = data_dict['box']
        self.scene_ids = data_dict['scene_ids']
        self.rel = data_dict['rel_b']
        B = self.x.shape[0]
        self.uc_rel = data_dict['uc_b']

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.rel =self.rel[:max_sample]
            self.rel = self.rel[:max_sample]
            self.uc_rel = self.uc_rel[:max_sample]

        vars_list = ['x']

        self.tocuda(var_names=vars_list)

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(self.device, non_blocking=True))

    def forward(self):
        self.df.train()
        rel = self.rel
        uc_rel = self.uc_rel
        target_box = self.x
        # Compute the loss
        self.loss, self.loss_dict = self.get_loss(uc_rel, rel, target_box)
        return self.loss, self.loss_dict

    def get_loss(self, uc_rel, rel, target_box):
        # Unpack the sample_params
        batch_size, len_params = target_box.shape
        condition = uc_rel
        if self.text_condition:
            raise NotImplementedError # use text embed for cross attention
        elif self.rel_condition:
            condition_cross = rel # use rel embed for cross attention
        else:
            condition_cross = None

        if target_box.shape[0] == 1:
            num_repeat = 32
            room_layout_target = target_box.repeat(num_repeat, 1, 1)
            if condition is not None:
                condition = condition.repeat(num_repeat, 1, 1)
            if condition_cross is not None:
                condition_cross = condition_cross.repeat(num_repeat, 1, 1)

        # denoise loss function
        loss, loss_dict = self.df.get_loss_iter(target_box, scene_ids=self.scene_ids, condition=condition, condition_cross=condition_cross)

        return loss, loss_dict

    def sample(self, room_mask, num_points, point_dim, batch_size=1, text=None, 
               partial_boxes=None, input_boxes=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None, 
                ):
        device = room_mask.device
        noise = torch.randn((batch_size, num_points, point_dim))#, device=room_mask.device)

        # get the latent feature of room_mask
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_mask)) #(B, F)
            
        else:
            room_layout_f = None

        # process instance & class condition f
        if self.instance_condition:
            if self.learnable_embedding:
                instance_indices = torch.arange(self.sample_num_points).long().to(device)[None, :].repeat(room_mask.size(0), 1)
                instan_condition_f = self.positional_embedding[instance_indices, :]
            else:
                instance_label = torch.eye(self.sample_num_points).float().to(device)[None, ...].repeat(room_mask.size(0), 1, 1)
                instan_condition_f = self.fc_instance_condition(instance_label) 
        else:
            instan_condition_f = None


        # concat instance and class condition   
        # concat room_layout_f and instan_class_f
        if room_layout_f is not None and instan_condition_f is not None:
            condition = torch.cat([room_layout_f[:, None, :].repeat(1, num_points, 1), instan_condition_f], dim=-1).contiguous()
        elif room_layout_f is not None:
            condition = room_layout_f[:, None, :].repeat(1, num_points, 1)
        elif instan_condition_f is not None:
            condition = instan_condition_f
        else:
            condition = None

        # concat room_partial condition
        if self.room_partial_condition:
            partial_valid   = torch.ones((batch_size, self.partial_num_points, 1)).float().to(device)
            ###partial_invalid = torch.ones((batch_size, num_points - self.partial_num_points, 1)).float().to(device)
            partial_invalid = torch.zeros((batch_size, num_points - self.partial_num_points, 1)).float().to(device)
            partial_mask    = torch.cat([ partial_valid, partial_invalid ], dim=1).contiguous()
            partial_input   = input_boxes * partial_mask
            partial_condition_f = self.fc_partial_condition(partial_input)
            condition = torch.cat([condition, partial_condition_f], dim=-1).contiguous()

        # concat  room_arrange condition
        if self.room_arrange_condition:
            arrange_input  = torch.cat([ input_boxes[:, :, self.translation_dim:self.translation_dim+self.size_dim], input_boxes[:, :, self.bbox_dim:] ], dim=-1).contiguous()
            arrange_condition_f = self.fc_arrange_condition(arrange_input)
            condition = torch.cat([condition, arrange_condition_f], dim=-1).contiguous()


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
            

        print('unconditional / conditional generation sampling')
        # reverse sampling
        if ret_traj:
            samples = self.df.gen_sample_traj(noise.shape, room_mask.device, freq=freq, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
        else:
            samples = self.df.gen_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
        
        return samples

    @torch.no_grad()
    def generate_layout(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False):
        
        samples = self.sample(room_mask, num_points, point_dim, batch_size, text=text, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)
        
        return self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)

    @torch.no_grad()
    def generate_layout_progressive(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False, num_step=100):
        
        # output dictionary of sample trajectory & sample some key steps
        samples_traj = self.sample(room_mask, num_points, point_dim, batch_size, text=text, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds, freq=num_step)
        boxes_traj = {}

        # delete the initial noisy
        samples_traj = samples_traj[1:]

        for i in range(len(samples_traj)):
            samples = samples_traj[i]
            k_time = num_step * i
            boxes_traj[k_time] = self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)
        return boxes_traj
    

    @torch.no_grad()
    def delete_empty_from_network_samples(self, samples, device="cpu", keep_empty=False):
        
        samples_dict = {
            "translations": samples[:, :, 0:self.translation_dim].contiguous(),
            "sizes": samples[:, :,  self.translation_dim:self.translation_dim+self.size_dim].contiguous(),
            "angles": samples[:, :, self.translation_dim+self.size_dim:self.bbox_dim].contiguous(),
        }
        if self.objfeat_dim > 0:
            samples_dict["objfeats"] = samples[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objfeat_dim]

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
