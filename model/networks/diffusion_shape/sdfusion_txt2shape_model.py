# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import time
from collections import OrderedDict
from functools import partial
from inspect import isfunction

import cv2
import numpy as np
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from .base_model import BaseModel
from model.networks.vqvae_networks.network import VQVAE
from model.networks.diffusion_shape.network import DiffusionUNet
from model.model_utils import load_vqvae

from helpers.util import bool_flag, _CustomDataParallel
from helpers.lr_scheduler import *
from fvcore.common.param_scheduler import MultiStepParamScheduler

# ldm util
from model.networks.diffusion_shape.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    exists,
    default,
)
from model.networks.diffusion_shape.samplers.ddim import DDIMSampler

# distributed 
from .diff_utils.distributed import reduce_loss_dict

# rendering
from .diff_utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionText2ShapeModel(BaseModel):
    def __init__(self, opt):
        super(SDFusionText2ShapeModel, self).__init__()
        BaseModel.initialize(self, opt)
        self.isTrain = opt.hyper.isTrain
        self.model_name = self.name()
        self.device = opt.hyper.device

        ######## START: Define Networks ########
        assert opt.shape_branch.df_cfg is not None
        assert opt.shape_branch.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.shape_branch.df_cfg)
        vq_conf = OmegaConf.load(opt.shape_branch.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult) - 1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)

        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)
        self.df.to(self.device)
        #self.df = _CustomDataParallel(self.df)
        self.init_diffusion_params(uc_scale=3., opt=opt)

        # sampler
        self.ddim_sampler = DDIMSampler(self)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.shape_branch.vq_ckpt, opt=opt)

        # # init cond model
        # bert_params = df_conf.bert.params
        # self.text_embed_dim = bert_params.n_embed
        # for param in self.cond_model.parameters():
        #     param.requires_grad = True
        ######## END: Define Networks ########

        # param list
        trainable_models = [self.df]
        trainable_params = []
        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]
            # print(len(trainable_params))

        self.trainable_params = trainable_params

        # if opt.shape_branch.ckpt is not None:
        #     self.load_ckpt(opt.shape_branch.ckpt, load_opt=self.isTrain)

        # transforms
        self.to_tensor = transforms.ToTensor()

        # setup renderer
        dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim,
                                           device=self.opt.hyper.device)

        self.df_module = self.df
        self.vqvae_module = self.vqvae

        # for debugging purpose
        self.ddim_steps = 100
        if self.opt.misc.debug == 1:
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def name(self):
        return 'SDFusion-Text2Shape-Model'

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.misc.local_rank],
            output_device=opt.misc.local_rank,
            broadcast_buffers=False,
        )
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.misc.local_rank],
            output_device=opt.misc.local_rank,
            broadcast_buffers=False,
        )
        # self.cond_model = nn.parallel.DistributedDataParallel(
        #     self.cond_model,
        #     device_ids=[opt.misc.local_rank],
        #     output_device=opt.misc.local_rank,
        #     broadcast_buffers=False,
        #     find_unused_parameters=True,
        # )

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, uc_scale=3., opt=None):
        
        df_conf = OmegaConf.load(opt.shape_branch.df_cfg)
        df_model_params = df_conf.model.params
        
        # ref: ddpm.py, line 44 in __init__()
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule(
            timesteps=df_model_params.timesteps,
            linear_start=df_model_params.linear_start,
            linear_end=df_model_params.linear_end,
        )
        
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # for cls-free guidance
        self.uc_scale = uc_scale


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
        ############################ END: init diffusion params ############################

    def set_input(self, input=None, max_sample=None):
        
        self.x = input['sdf']

        self.rel = input['rel_s']
        B = self.x.shape[0]
        self.uc_rel = input['uc_s']
        if self.df.conditioning_key == 'concat':
            self.rel = self.rel.view(B,-1,16,16,16)
            self.uc_rel = self.uc_rel.view(B,-1,16,16,16)
        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.rel = self.rel[:max_sample]
            self.uc_rel = self.uc_rel[:max_sample]

        vars_list = ['x']

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.df_module.eval()
        self.vqvae_module.eval()
        # self.cond_model.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'loss_total': loss.clone().detach().mean()})

        return x_noisy, target, loss, loss_dict


    def forward(self):

        self.switch_train()

        c_rel = self.rel # B, 77, 1280

        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        #    check: ldm.models.autoencoder.py, VQModelInterface's encode(self, x)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c_rel, t)

        self.loss_df = loss
        self.loss_dict = loss_dict
        return self.loss_df, self.loss_dict

    @torch.no_grad()
    def gen_shape_after_foward(self, num_obj, ddim_steps=None, uc_scale=None,  ddim_eta=0.,):
        self.switch_eval()
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        if uc_scale is None:
            uc_scale = self.uc_scale
        B = self.rel[:num_obj].shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                          batch_size=B,
                                                          shape=shape,
                                                          conditioning=self.rel[:num_obj],
                                                          verbose=False,
                                                          unconditional_guidance_scale=uc_scale,
                                                          unconditional_conditioning=self.uc_rel[:num_obj],
                                                          eta=ddim_eta)
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        self.switch_train()


    @torch.no_grad()
    def graph2shape(self, num_obj=6, ddim_steps=100, ddim_eta=0.0, uc_scale=None):

        self.switch_eval()

        # data = {
        #     'sdf': torch.zeros(num_obj),
        #     'text': [input_txt] * num_obj,
        # }

        # self.set_input(data)

        ddim_sampler = DDIMSampler(self)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        if uc_scale is None:
            uc_scale = self.uc_scale
        B = self.rel[:num_obj].shape[0]
        shape = self.z_shape
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                    batch_size=B,
                                                    shape=shape,
                                                    conditioning=self.rel[:num_obj],
                                                    verbose=False,
                                                    unconditional_guidance_scale=uc_scale,
                                                    unconditional_conditioning=None,
                                                    eta=ddim_eta)
        self.gen_df = self.vqvae_module.decode_no_quant(samples)

        return self.gen_df

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):
        print("Peeeepeeeeeeee.")
        self.switch_eval()

        if not infer_all:
            # max_sample = 16
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale
            
        # get noise, denoise, and decode with vqvae
        uc = self.uc_rel
        c_rel = self.rel
        B = c_rel.shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c_rel,
                                                     verbose=False,
                                                     unconditional_guidance_scale=uc_scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)

        self.switch_train()

    @torch.no_grad()
    def rel2shape(self, data, ddim_steps=100, ddim_eta=0.0, uc_scale=None):
        self.switch_eval()
        self.set_input(data)

        ddim_sampler = DDIMSampler(self)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.scale

        # get noise, denoise, and decode with vqvae
        uc = self.uc_rel
        c_text = self.rel
        B = c_text.shape[0]
        shape = self.z_shape

        demo_ = 0
        if demo_:
            seed_value = int(1683052538)
            print(seed_value)
            torch.manual_seed(seed_value)
            C, D, H, W = shape
            size = (B, C, D, H, W)
            noise = torch.randn(size, device='cuda')
        else:
            C, D, H, W = shape
            size = (1, C, D, H, W)
            torch.manual_seed(int(time.time()))
            single_noise = torch.randn(size, device='cuda')
            noise = single_noise.repeat(B, 1, 1, 1, 1)

        mini_B = 7
        number_mini_B = np.ceil(B / mini_B).astype(int)
        rest = B
        gen_df_list = []
        for i in range(number_mini_B):

            num = np.min((mini_B,rest)).astype(int)
            samples_mini, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                         batch_size=num,
                                                         shape=shape,
                                                         conditioning=c_text[i*mini_B:(i+1)*mini_B],
                                                         x_T=noise[i*mini_B:(i+1)*mini_B],
                                                         verbose=False,
                                                         unconditional_guidance_scale=uc_scale,
                                                         unconditional_conditioning=uc[i*mini_B:(i+1)*mini_B],
                                                         eta=ddim_eta)
            rest -= num
            # decode z
            gen_df_list.append(self.vqvae_module.decode_no_quant(samples_mini))

        self.gen_df = torch.cat(gen_df_list,dim=0)


        return self.gen_df

    @torch.no_grad()
    def txt2shape(self, input_txt, ngen=6, ddim_steps=100, ddim_eta=0.0, uc_scale=None):

        self.switch_eval()

        data = {
            'sdf': torch.zeros(ngen),
            'text': [input_txt] * ngen,
        }
        
        self.set_input(data)

        ddim_sampler = DDIMSampler(self)
        
        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.scale
            
        # get noise, denoise, and decode with vqvae
        uc = self.cond_model(self.uc_text)
        c_text = self.cond_model(self.text)  
        B = c_text.shape[0]
        shape = self.z_shape
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=B,
                                                     shape=shape,
                                                     conditioning=c_text,
                                                     verbose=False,
                                                     unconditional_guidance_scale=uc_scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)
        
        # decode z
        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        return self.gen_df

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        
        self.switch_eval()
        return ret

    def backward(self,retain_graph=False):
        

        self.loss = self.loss_df

        self.loss_dict = reduce_loss_dict(self.loss_dict)
        self.loss_total = self.loss_dict['loss_total']
        self.loss_simple = self.loss_dict['loss_simple']
        self.loss_vlb = self.loss_dict['loss_vlb']
        if 'loss_gamma' in self.loss_dict:
            self.loss_gamma = self.loss_dict['loss_gamma']

        self.loss.backward(retain_graph=retain_graph)

    def update_loss(self):

        self.loss = self.loss_df

        self.loss_total = self.loss_dict['loss_total']
        self.loss_simple = self.loss_dict['loss_simple']
        self.loss_vlb = self.loss_dict['loss_vlb']
        if 'loss_gamma' in self.loss_dict:
            self.loss_gamma = self.loss_dict['loss_gamma']


    def optimize_parameters(self, total_steps):

        self.set_requires_grad([self.df], requires_grad=True)
        # self.set_requires_grad([self.cond_model], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('simple', self.loss_simple.mean().data),
            ('vlb', self.loss_vlb.mean().data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.mean().data

        return ret

    def write_text_on_img(self, text, bs=16, img_shape=(3, 256, 256)):
        # write text as img
        b, c, h, w = len(text), 3, 256, 256
        img_text = np.ones((b, h, w, 3)).astype(np.float32) * 255
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_COMPLEX
        font_size = 0.5
        n_char_per_line = 25 # new line for text

        y0, dy = 20, 1
        for ix, txt in enumerate(text):
            # newline every "space" chars
            for i in range(0, len(txt), n_char_per_line):
                y = y0 + i * dy
                # new_txt.append(' '.join(words[i:i+space]))
                # txt_i = ' '.join(txt[i:i+space])
                txt_i = txt[i:i+n_char_per_line]
                cv2.putText(img_text[ix], txt_i, (10, y), font, font_size, (0., 0., 0.), 2)

        return img_text/255.

    def get_current_visuals(self, vocab, obj_ids, num_obj=2):
        texts = [vocab.get(id) for id in obj_ids[:num_obj]]
        with torch.no_grad():
            self.text_obj= texts # input text
            self.img_gt = render_sdf(self.renderer, self.x[:num_obj]).detach().cpu() # rendered gt sdf
            self.img_gen_df = render_sdf(self.renderer, self.gen_df).detach().cpu() # rendered generated sdf
        
        b, c, h, w = self.img_gt.shape
        img_shape = (3, h, w)
        # write text as img
        self.img_text = self.write_text_on_img(self.text_obj, bs=b, img_shape=img_shape)
        self.img_text = rearrange(torch.from_numpy(self.img_text), 'b h w c -> b c h w')

        vis_tensor_names = [
            # 'img',
            'img_gt',
            'img_gen_df',
            'img_text',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_step, save_opt=False):
        
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            # 'cond_model': self.cond_model_module.state_dict(),
            'df': self.df_module.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        # self.cond_model.load_state_dict(state_dict['cond_model'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


