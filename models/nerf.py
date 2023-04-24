import torch
from torch import nn
import tinycudann as tcnn
import numpy as np
from torch.cuda.amp import custom_fwd, custom_bwd


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=2, W=64,
                 in_channels_xyz=3, in_channels_dir=3,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):

        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.encode_appearance = False if typ == 'coarse' else encode_appearance  # encode_appearance  #
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ == 'coarse' else encode_transient
        self.typ = typ
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        self.geo_feat_dim = 15

        # NGP
        self.rgb_act = "Sigmoid"

        self.scale = 0.5
        self.register_buffer('xyz_min', -torch.ones(1, 3)*self.scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*self.scale)


        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        # NGP

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1 + self.geo_feat_dim,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": self.D - 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=self.geo_feat_dim + 16 + self.in_channels_a,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": self.W,
                    "n_hidden_layers": self.D,
                }
            )

        if self.encode_transient:
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + 16,
                n_output_dims=64,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": self.D - 1,
                },
            )

            self.transient_sigma = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(64, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def density(self, x, return_feat=False):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        h = self.xyz_encoder(x)
        density_before_activation, xyz_encoder_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)  # !!!!!!!!!!dim=0
        sigmas = TruncExp.apply(density_before_activation)
        if return_feat: return sigmas, h, xyz_encoder_out
        return sigmas

    def forward(self, x, sigma_only=False, output_transient=True):

        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir, input_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir, self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir, input_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,self.in_channels_a], dim=-1)

        static_sigma, h, xyz_encoder_out = self.density(input_xyz, return_feat=True)

        if sigma_only:
            return static_sigma

        input_dir = input_dir / torch.norm(input_dir, dim=1, keepdim=True)
        input_dir = self.dir_encoder((input_dir + 1) / 2)
        input_static = torch.cat([input_dir, xyz_encoder_out, input_a], 1)
        static_rgb = self.rgb_net(input_static)
        static = torch.cat([static_rgb, static_sigma], 1)

        if not output_transient:
            return static

        input_transient = torch.cat([xyz_encoder_out, input_t], 1)
        transient_all = self.mlp_transient(input_transient)
        transient_sigma = self.transient_sigma(transient_all)
        transient_rgb = self.transient_rgb(transient_all)
        transient_beta = self.transient_beta(transient_all)

        transient = torch.cat([transient_rgb, transient_sigma, transient_beta], 1)

        return torch.cat([static, transient], 1)