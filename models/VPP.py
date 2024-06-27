import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import misc
from utils.logger import *
from .build import MODELS
from itertools import product
from timm.models.layers import trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1

from models.vqgan import VQGAN
from models.point_transformer import PointEncoder
from models.prompt import TextEncoder, ImageEncoder
from modules.voxelization import Voxelization, voxel_to_point
from models.transformer import TransformerEncoder, TransformerDecoder, FusionTransformer, SeqInvariantTransformer


def random_mask(x):
    B, N, _ = x.shape

    overall_mask = np.zeros([B, N])
    for i in range(B):
        num_mask = int(np.cos(np.random.random() * np.pi * 0.5) * N)
        mask = np.hstack([
            np.zeros(N - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        overall_mask[i, :] = mask
    overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
    return overall_mask.to(x.device)


class GridSmoother(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        print_log(f'[args] {config}', logger='GridSmoother')

        self.embed = nn.Linear(3, self.trans_dim, bias=False)
        self.projection = nn.Linear(self.trans_dim, 3, bias=False)
        self.voxelization = Voxelization(config=config)

        dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]
        self.blocks = SeqInvariantTransformer(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.rec_point_loss = ChamferDistanceL1().cuda()
        self.homogeneity_loss = HomogeneityLoss(k=5)
        self.kl_ratio = config.get('kl_ratio', 1.0)

    def forward(self, pts):
        B = pts.shape[0]
        voxel = self.voxelization(pts)
        rec_point_loss = torch.zeros(B).to(pts.device)
        homogeneity_loss = torch.zeros(B).to(pts.device)
        for i in range(B):
            grid_point = voxel_to_point(voxel[i])
            N = grid_point.shape[0]

            x = self.embed(grid_point.unsqueeze(dim=0))
            x = self.blocks(x)
            grid_point = self.projection(x)

            center_point = misc.fps(pts[i].unsqueeze(dim=0), N)
            rec_point_loss[i] = self.rec_point_loss(grid_point, center_point)
            homogeneity_loss[i] = self.homogeneity_loss(grid_point) * self.kl_ratio
        return rec_point_loss.mean(), homogeneity_loss.mean()

    def inference(self, grid_point):
        x = self.embed(grid_point)
        x = self.blocks(x)
        pred_point = self.projection(x)
        return pred_point


class VoxelGenerator(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.vqgan_config.codebook_dim
        self.depth = config.depth
        self.num_heads = config.num_heads

        self.cfg_ratio = config.get('cfg_ratio', 0.8)
        self.noise_ratio = config.get('noise_ratio', 0.5)
        self.rand_ratio = config.get('rand_ratio', 0.1)
        self.image_text_ratio = config.get('image_text_ratio', 0.7)

        self.codebook_resolution = config.vqgan_config.resolution // config.vqgan_config.down_sample
        self.codebook_num = config.vqgan_config.codebook_num

        self.prompt_dim = config.prompt_dim
        self.temperature = config.get('temperature', 1.0)
        print_log(f'[args] {config}', logger='VoxelGenerator')

        n_points = self.codebook_resolution
        x = torch.linspace(-1, 1, n_points)
        y = torch.linspace(-1, 1, n_points)
        z = torch.linspace(-1, 1, n_points)

        self.grid = nn.Parameter(torch.Tensor(list(product(x, y, z))), requires_grad=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.projection = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.trans_dim, self.codebook_num, bias=False)
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]
        self.blocks = FusionTransformer(
            embed_dim=self.trans_dim,
            prompt_dim=self.prompt_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, prompt_features, null_prompt_features, encoded_voxel):
        B, C, R, _, _ = encoded_voxel.shape
        encoded_voxel = encoded_voxel.reshape(B, C, R * R * R)
        encoded_voxel = encoded_voxel.transpose(1, 2)
        mask = random_mask(encoded_voxel)
        encoded_voxel[mask] = self.mask_token

        pos = self.pos_embed(self.grid)

        prompt_dim = prompt_features.shape[-1]

        gaussian_noise = torch.randn(B, prompt_dim, device=encoded_voxel.device)
        gaussian_noise = gaussian_noise / torch.norm(gaussian_noise, dim=-1, keepdim=True)
        gaussian_noise = self.noise_ratio * gaussian_noise * torch.norm(prompt_features, dim=-1, keepdim=True)
        prompt_features = prompt_features + gaussian_noise.detach()

        replace_indices = (torch.rand(B) < self.rand_ratio).reshape(B, 1).to(encoded_voxel.device)
        null_prompt_features = null_prompt_features.expand(B, prompt_dim)
        cfg_features = torch.where(replace_indices, null_prompt_features, prompt_features)

        x = self.blocks(encoded_voxel, pos, cfg_features)
        x = self.norm(x)

        logits = self.projection(x)

        return logits, mask

    def inference(self, prompt_features, null_prompt_feature, codebooks, steps, origin_tokens=None):

        B, _ = prompt_features.shape
        R = self.codebook_resolution
        device = next(self.parameters()).device

        init_tokens = origin_tokens.reshape(B, R * R * R, -1) if origin_tokens is not None else \
            self.mask_token.expand(B, R * R * R, -1)
        tokens = init_tokens.clone()
        pos_full = self.pos_embed(self.grid)
        prob = torch.zeros(B, R * R * R, self.codebook_num, device=device)
        topk_indices = None

        temp_grid = 1 - torch.mean(self.grid.clone()**2, dim=-1).reshape(R, R, R) + 1e-2

        for step in range(steps):

            mask_ratio = float(np.cos((step + 1) / steps * np.pi * 0.5))
            fix_len = int((1 - mask_ratio) * R * R * R)

            # transformer
            cfg_feature = null_prompt_feature + self.cfg_ratio * (prompt_features - null_prompt_feature)
            x = self.blocks(tokens, pos_full, cfg_feature)
            x = self.norm(x)
            logits = self.projection(x)

            temperature = temp_grid.reshape(1, R * R * R, 1) * self.temperature * (1 - step / steps)

            prob = F.softmax(logits / temperature, dim=-1)  # B, N, codebook_num
            index = self.probabilistic_select(prob)  # B, N, 1
            quant_embeding = codebooks(index.squeeze(dim=-1))  # B, N, codebook_dim

            score = torch.gather(prob, 2, index.long()).squeeze(dim=-1)  # B, N
            if topk_indices is not None:
                for i in range(B):
                    for j in range(topk_indices.shape[1]):
                        score[i][topk_indices[i][j]] = 1.0
            _, topk_indices = torch.topk(score, k=fix_len, dim=-1)  # B, fix_len

            tokens = init_tokens.clone()
            for i in range(B):
                for j in range(fix_len):
                    tokens[i][topk_indices[i][j]] = quant_embeding[i][topk_indices[i][j]]

        _, logits_top1_indices = torch.topk(prob, k=1, dim=-1)
        features = codebooks(logits_top1_indices.squeeze(dim=-1))
        features = features.transpose(1, 2)
        features = features.reshape(B, self.trans_dim, R, R, R)

        return features

    def probabilistic_select(self, logits):
        B, N, _ = logits.shape
        probs = logits.reshape(B * N, -1)
        topk_probs, topk_indices = torch.topk(probs, k=500, dim=1)
        index = torch.multinomial(topk_probs, num_samples=1)
        select = torch.gather(topk_indices, dim=1, index=index)
        return select.reshape(B, N, 1)


class PointUpsampler(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        print_log(f'[args] {config}', logger='TokenGenerator')

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features, center):

        bool_masked_pos = random_mask(center)
        features[bool_masked_pos] = self.mask_token
        pos = self.pos_embed(center)

        # transformer
        x = self.blocks(features, pos)
        x = self.norm(x)

        return x, bool_masked_pos

    def inference(self, center):
        B, G, _ = center.shape
        x_full = self.mask_token.expand(B, G, -1)
        pos_full = self.pos_embed(center)
        # transformer
        x = self.blocks(x_full, pos_full)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.point_config.trans_dim
        self.group_size = config.group_size
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.with_color = config.with_color
        self.channel = 6 if self.with_color else 3
        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, 0.1, self.decoder_depth)]
        self.blocks = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        # prediction head
        self.increase_dim = nn.Conv1d(self.trans_dim, self.channel * self.group_size, 1)

    def forward(self, center, token):
        decoder_pos = self.decoder_pos_embed(center)
        x_rec = self.blocks(token, decoder_pos)
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2)
        return rebuild_points


class HomogeneityLoss(nn.Module):
    def __init__(self, k, **kwargs):
        super().__init__()
        self.k = k

    def forward(self, center):
        k = min(self.k + 1, center.shape[1])
        dist_matrix = torch.cdist(center, center)
        _, indices = torch.topk(dist_matrix, k=k, largest=False)
        neighbor_indices = indices[:, 1:]
        mean_dists = torch.gather(dist_matrix, dim=1, index=neighbor_indices).mean(dim=1)

        p = F.softmax(mean_dists, dim=0)
        q = torch.ones_like(p) / len(p)
        loss = F.kl_div(p.log(), q, reduction='sum')

        return loss


@MODELS.register_module()
class VPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[VPP] ', logger='VPP')
        self.config = config
        self.mode = config.mode
        self.with_color = config.get('with_color', False)
        print_log(f'Training with color: {self.with_color}', logger='VPP')

        if self.mode == 'coarse':
            # load condition encoder
            self.text_encoder = TextEncoder(config=config)
            self.image_encoder = ImageEncoder(config=config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

            # load VQGAN
            config.vqgan_config.with_color = self.with_color
            self.vqgan = VQGAN(config=config.vqgan_config)
            self.vqgan.load_model_from_ckpt(ckpt_path=config.vqgan_config.ckpt_path)
            for p in self.vqgan.parameters():
                p.requires_grad = False

            # load voxel_generator
            config.voxel_config.prompt_dim = self.text_encoder.embed_dim
            config.voxel_config['vqgan_config'] = config.vqgan_config
            self.voxel_generator = VoxelGenerator(config=config.voxel_config)

            # loss
            self.rec_codebook_loss = nn.CrossEntropyLoss()

        elif self.mode == 'smoother':
            # load gird_smoother
            self.grid_smoother = GridSmoother(config=config.smooth_config)

        elif self.mode == 'fine':
            # load point_encoder
            config.encoder_config.num_group = config.num_group
            config.encoder_config.group_size = config.group_size
            self.encoder = PointEncoder(config=config.encoder_config)
            self.encoder.load_model_from_ckpt(ckpt_path=config.encoder_config.ckpt_path)
            for p in self.encoder.parameters():
                p.requires_grad = False

            # load point_upsampler
            self.group_size = config.point_config.group_size = config.group_size
            self.num_group = config.point_config.num_group = config.num_group
            config.point_config.with_color = self.with_color
            self.point_upsampler = PointUpsampler(config=config.point_config)

            # load decoder
            self.decoder = Decoder(config=config)

            # loss
            self.rec_token_loss = nn.SmoothL1Loss()
            self.rec_point_loss = ChamferDistanceL1().cuda()
        else:
            raise NotImplementedError

    def training_voxel_generator(self, pts, image, text, **kwargs):

        voxel, encoded_voxel, codebook_indices, _ = self.vqgan.encode(pts)

        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        null_prompt_features = self.text_encoder("")

        r = self.voxel_generator.image_text_ratio
        prompt_features = r * image_features + (1 - r) * text_features

        logits, mask = self.voxel_generator(prompt_features, null_prompt_features, encoded_voxel)
        B, N = mask.shape
        codebook_indices = codebook_indices.reshape(B, N)
        rec_codebook_loss = self.rec_codebook_loss(logits[mask], codebook_indices[mask])

        _, pred_encoding_indices = torch.topk(logits, k=1, dim=-1)
        pred_encoding_indices = pred_encoding_indices.squeeze(dim=-1)
        z_q = self.vqgan.codebook.embedding(pred_encoding_indices)
        z_q = z_q.transpose(1, 2)
        z_q = z_q.view(encoded_voxel.shape)
        decoded_voxel = self.vqgan.decode(z_q)

        return rec_codebook_loss, voxel, decoded_voxel

    def training_grid_smoother(self, pts, **kwargs):

        rec_loss, homogeneity_loss = self.grid_smoother(pts[:, :, :3])
        loss = rec_loss + homogeneity_loss
        print(rec_loss, homogeneity_loss)

        return loss

    def training_point_upsampler(self, pts):
        point_features, neighborhood, center = self.encoder(pts[:, :, :3])
        B, M, _ = center.shape

        point_features_rec, mask = self.point_upsampler(point_features, center)
        rec_token_loss = self.rec_token_loss(point_features[mask], point_features_rec[mask]).mean()

        rebuild_points = self.decoder(center, point_features_rec).reshape(B * M, -1, self.decoder.channel)
        gt_points = neighborhood.reshape(B * M, -1, self.decoder.channel)
        rec_point_loss = self.rec_point_loss(rebuild_points, gt_points)

        loss = rec_token_loss + rec_point_loss
        print(rec_token_loss, rec_point_loss)

        return loss


@MODELS.register_module()
class VPPInference(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[VPPInference] ', logger='VPPInference')
        self.config = config
        self.group_size = config.get('group_size', 32)
        self.npoints = config.get('npoints', 8192)
        self.num_group = int(self.npoints / self.group_size)
        self.steps = config.get('steps', 4)
        self.with_color = config.get('with_color', False)
        self.channel = 6 if self.with_color else 3

        # load text encoder
        if 'prompt_encoder' in config:
            self.text_encoder = TextEncoder(config=config)
            self.image_encoder = ImageEncoder(config=config)
            self.prompt_dim = self.text_encoder.embed_dim

        # load vqgan
        if 'voxel_config' in config:
            config.vqgan_config.with_color = self.with_color
            self.vqgan = VQGAN(config=config.vqgan_config)

        # load voxel_generator
        if 'voxel_config' in config:
            config.voxel_config.prompt_dim = self.prompt_dim
            config.voxel_config['vqgan_config'] = config.vqgan_config
            self.voxel_generator = VoxelGenerator(config=config.voxel_config)

        # load grid_smoother
        if 'smooth_config' in config:
            self.grid_smoother = GridSmoother(config=config.smooth_config)

        # load point upsampler and decoder
        if 'point_config' in config:
            config.point_config.with_color = self.with_color
            self.point_upsampler = PointUpsampler(config=config.point_config)
            self.decoder = Decoder(config=config)

        self.load_model_from_ckpt()
        for p in self.parameters():
            p.requires_grad = False

    def load_model_from_ckpt(self):
        ckpt = {}

        if 'voxel_config' in self.config:
            voxel_generator = torch.load(self.config.voxel_config.ckpt_path, map_location='cpu')
            voxel_ckpt = {k.replace("module.", ""): v for k, v in voxel_generator['base_model'].items()}
            ckpt.update(voxel_ckpt)
            print_log(f'[VPPInference] Loading voxel_generator...', logger='VPPInference')

        if 'smooth_config' in self.config:
            grid_smoother = torch.load(self.config.smooth_config.ckpt_path, map_location='cpu')
            smooth_ckpt = {k.replace("module.", ""): v for k, v in grid_smoother['base_model'].items()}
            ckpt.update(smooth_ckpt)
            print_log(f'[VPPInference] Loading grid_smoother...', logger='VPPInference')

        if 'point_config' in self.config:
            point_upsampler = torch.load(self.config.point_config.ckpt_path, map_location='cpu')
            point_ckpt = {k.replace("module.", ""): v for k, v in point_upsampler['base_model'].items()}
            ckpt.update(point_ckpt)
            print_log(f'[VPPInference] Loading point_upsampler...', logger='VPPInference')

        state_dict = self.state_dict()
        for key in state_dict:
            if key not in ckpt:
                raise ValueError(f"missing ckpt keys: {key}")
            state_dict[key] = ckpt[key]
        self.load_state_dict(state_dict, strict=True)

        print_log(f'[VPPInference] Successful Loading all the ckpt', logger='VPPInference')

    def upsample(self, center, **kwargs):

        B, M, _ = center.shape
        point_features = self.point_upsampler.inference(center)
        rebuild_points = self.decoder(center, point_features).reshape(B, M, -1, self.channel)
        rebuild_points[:, :, :, :3] = rebuild_points[:, :, :, :3] + center.unsqueeze(dim=2)
        rebuild_points = rebuild_points.reshape(B, -1, self.channel)

        return rebuild_points

    def forward(self, prompt_features, null_prompt_features, origin_tokens=None, **kwargs):

        B, _ = prompt_features.shape

        codebooks = self.vqgan.codebook.embedding
        quant_embeding = self.voxel_generator.inference(prompt_features, null_prompt_features
                                                        , codebooks, self.steps, origin_tokens)
        decoded_voxel = self.vqgan.decode(quant_embeding)

        generated_points = torch.zeros(B, self.npoints, self.channel, device=decoded_voxel.device)
        for i in range(B):
            grid_center = voxel_to_point(decoded_voxel[i]).contiguous().unsqueeze(0)
            coordinate = grid_center[:, :, :3]

            _, M, _ = coordinate.shape
            if M <= 1:
                generated_points[i] = torch.randn(self.npoints, self.channel, device=decoded_voxel.device)
                continue

            center = self.grid_smoother.inference(coordinate)
            point_features = self.point_upsampler.inference(center)
            rebuild_points = self.decoder(center, point_features).reshape(1, M, -1, 3)
            rebuild_points = rebuild_points + center.unsqueeze(2)
            rebuild_points = rebuild_points.reshape(-1, 3)
            num_points = rebuild_points.shape[0]
            if B == 1:
                return rebuild_points.unsqueeze(0)
            if num_points >= self.npoints:
                points = misc.fps(rebuild_points.unsqueeze(0), self.npoints).squeeze(0)
            else:
                points = F.interpolate(rebuild_points.reshape(1, self.channel, num_points),
                                       size=self.npoints, mode='linear').reshape(self.npoints, self.channel)
            generated_points[i] = points

        return generated_points

    def text_condition_generation(self, text):
        text_features = self.text_encoder(text)
        null_prompt_features = self.text_encoder("")
        return self.forward(text_features, null_prompt_features)

    def image_condition_generation(self, img):
        img_features = self.image_encoder(img)
        null_prompt_features = self.text_encoder("")
        return self.forward(img_features, null_prompt_features)
