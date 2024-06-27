import torch
import torch.nn as nn
import numpy as np
from modules.voxelization import Voxelization
from modules.voxelconv import ResVoxelConv, VoxelDeConv, VoxelConv
from .build import MODELS
from utils.logger import *


class VoxelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_channel = config.input_channel
        codebook_dim = config.codebook_dim

        base_dim = config.base_dim
        down_sample = config.down_sample
        assert np.log2(down_sample) % 1 == 0
        layer_num = int(np.log2(down_sample))
        dim_list = [base_dim * int(np.exp2(i)) for i in range(layer_num + 1)]
        with_se = config.with_se

        self.voxelization = Voxelization(config)
        layers = [VoxelConv(input_channel, base_dim, pooling=False, with_se=with_se)]
        for i in range(layer_num):
            layers.append(VoxelConv(dim_list[i], dim_list[i + 1], pooling=True, with_se=with_se))
            layers.append(ResVoxelConv(dim_list[i + 1], with_se=with_se))
        layers.append(VoxelConv(dim_list[-1], codebook_dim, pooling=False, with_se=with_se))
        self.blocks = nn.Sequential(*layers)

    def forward(self, pts):
        voxel = self.voxelization(pts)
        voxel_features = self.blocks(voxel)
        return voxel, voxel_features


class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_channel = config.codebook_dim
        output_channel = config.input_channel

        base_dim = config.base_dim
        down_sample = config.down_sample
        assert np.log2(down_sample) % 1 == 0
        layer_num = int(np.log2(down_sample))
        dim_list = [base_dim * int(np.exp2(layer_num - i)) for i in range(layer_num + 1)]
        with_se = config.with_se

        layers = [VoxelConv(input_channel, dim_list[0], pooling=False, with_se=with_se)]
        for i in range(layer_num):
            layers.append(VoxelDeConv(dim_list[i], dim_list[i + 1], scale=2, with_se=with_se))
            layers.append(ResVoxelConv(dim_list[i + 1], with_se=with_se))
        layers.append(GroupNorm(base_dim))
        layers.append(Swish())
        layers.append(nn.Conv3d(base_dim, output_channel, 3, 1, 1))

        self.blocks = nn.Sequential(*layers)

    def forward(self, voxel_features):
        voxel = self.blocks(voxel_features)
        return voxel


class Codebook(nn.Module):
    def __init__(self, config):
        super(Codebook, self).__init__()
        self.codebook_num = config.codebook_num
        self.codebook_dim = config.codebook_dim
        self.beta = config.get('beta', 1)

        self.embedding = nn.Embedding(self.codebook_num, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_num, 1.0 / self.codebook_num)

    def forward(self, z):
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, loss, min_encoding_indices


@MODELS.register_module()
class VQGAN(nn.Module):
    def __init__(self, config):
        super(VQGAN, self).__init__()
        self.config = config
        if config.with_color:
            config.input_channel = 4
        else:
            config.input_channel = 1
        codebook_dim = config.codebook_dim
        self.encoder = VoxelEncoder(config=config)
        self.decoder = VoxelDecoder(config=config)
        self.codebook = Codebook(config=config)
        self.quant_conv = nn.Conv3d(codebook_dim, codebook_dim, 1)
        self.post_quant_conv = nn.Conv3d(codebook_dim, codebook_dim, 1)
        self.apply(weights_init)

    def forward(self, pts):
        voxel, voxel_features = self.encoder(pts)
        quant_conv_encoded_voxel = self.quant_conv(voxel_features)
        codebook_mapping, q_loss, _ = self.codebook(quant_conv_encoded_voxel)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_voxel = self.decoder(post_quant_conv_mapping)
        return voxel, decoded_voxel, q_loss

    def encode(self, pts):
        voxel, voxel_features = self.encoder(pts)
        quant_conv_encoded_voxel = self.quant_conv(voxel_features)
        codebook_mapping, q_loss, codebook_indices = self.codebook(quant_conv_encoded_voxel)
        return voxel, codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_voxel = self.decoder(post_quant_conv_mapping)
        if self.config.with_color:
            decoded_voxel[:, 1:] = torch.clip(decoded_voxel[:, 1:], 0, 1)
        return decoded_voxel

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.blocks[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lam = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lam = torch.clamp(lam, 0, 1e4).detach()
        return 0.8 * lam

    def load_model_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

        self.load_state_dict(ckpt, strict=True)
        print_log(f'[VQGAN] Successful Loading the ckpt from {ckpt_path}', logger='VQGAN')

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=channels // 8, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


@MODELS.register_module()
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        if config.get('with_color', False):
            input_channel = 4
        else:
            input_channel = 1
        base_dim = config.base_dim
        layers = [
            VoxelConv(input_channel, base_dim, pooling=True, with_se=False),
            ResVoxelConv(base_dim, with_se=False),
            VoxelConv(base_dim, base_dim * 2, pooling=True, with_se=False),
            ResVoxelConv(base_dim * 2, with_se=False),
            VoxelConv(base_dim * 2, 1, pooling=False, with_se=False),
        ]
        self.blocks = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        x = self.blocks(x)
        return x
