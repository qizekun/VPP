import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.knn import knn_point
from models.transformer import TransformerEncoder


class Encoder(nn.Module):  # Embedding module
    def __init__(self, encoder_channel, input_channel=3, c1=128, c2=256, c3=512):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.input_channel = input_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.input_channel, c1, 1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(2 * c2, c3, 1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
            point_groups : B G N 3/6
            -----------------
            feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.input_channel)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group=64, group_size=32):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, pts):
        """
        input: B N 3/6
        ---------------------------
        output: B G M 3/6
        center : B G 3/6
        """
        batch_size, num_points, _ = pts.shape
        xyz = pts[:, :, :3]

        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = pts.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, -1).contiguous()
        # normalize
        neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] - center.unsqueeze(2)
        return neighborhood, center


class FourierPosEmbedding(torch.nn.Module):
    def __init__(self, input_dims=3, include_input=True, max_freq=3, num_freqs=8, log_sampling=True,
                 periodic_fns=None):
        super().__init__()
        if periodic_fns is None:
            periodic_fns = [torch.sin, torch.cos]
        embed_fns = []
        self.out_dim = 0

        if include_input:
            embed_fns.append(lambda x: x)
            self.out_dim += input_dims
        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=num_freqs)
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += input_dims
        self.embed_fns = embed_fns

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PointEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group

        input_channel = 3
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.trans_dim, input_channel=input_channel)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='PointEncoder')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='PointEncoder'
            )
        print_log(f'[PointEncoder] Successful Loading the ckpt from {ckpt_path}', logger='PointEncoder')

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        x = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        return x, neighborhood, center


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.trans_dim

        input_channel = 3  # For fair comparison with other methods, we use the same input channel as other methods.
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, input_channel=input_channel)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, 0.2, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.loss_ce = nn.CrossEntropyLoss()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('point_upsampler.'):
                    base_ckpt[k[len('point_upsampler.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log('missing_keys', logger='PointTransformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='PointTransformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='PointTransformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='PointTransformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {ckpt_path}', logger='PointTransformer')
        else:
            print_log('Training from scratch!!!', logger='PointTransformer')
            self.apply(self._init_weights)

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

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
