import torch
import numpy as np
from functools import partial
from torch import nn
from omegaconf import OmegaConf
from model.networks.diffusion_networks.network import DiffusionUNet


from model.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    exists,
    default,
)

class sg_diff(nn.Module):
    def __init__(self, df_cfg= 'configs/sdfusion-txt2shape.yaml', vq_cfg= 'configs/vqvae_snet.yaml'):
        super(sg_diff, self).__init__()
        self.df_cfg =df_cfg
        self.vq_cfg = vq_cfg
        df_conf = OmegaConf.load(self.df_cfg)
        vq_conf = OmegaConf.load(self.vq_cfg)
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)
        self.df.to('cuda')

        self.init_diffusion_params(uc_scale=3., df_cfg=opt)

    def init_diffusion_params(self, uc_scale=3., df_cfg=None):

        df_conf = OmegaConf.load(df_cfg)
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