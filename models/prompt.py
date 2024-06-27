import os
import torch
import torch.nn as nn
from clip import clip


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        clip = load_clip_to_cpu(self.config.prompt_encoder)
        self.transformer = clip.transformer
        self.positional_embedding = clip.positional_embedding
        self.token_embedding = clip.token_embedding
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.dtype = clip.dtype
        self.embed_dim = self.transformer.width

    def forward(self, text):
        text = clip.tokenize(text, context_length=77).cuda()

        b, _ = text.shape
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # prompt_len_max = text.argmax(dim=-1).max()
        # text_feature = x[:, :prompt_len_max + 1] @ self.text_projection
        text_feature = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return text_feature


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'token_config' in config.keys():
            self.trans_dim = config.token_config.trans_dim
        elif 'vqgan_config' in config.keys():
            self.trans_dim = config.vqgan_config.codebook_dim
        else:
            raise NotImplementedError

        model = load_clip_to_cpu(self.config.prompt_encoder)
        image_model = model.visual

        for p in image_model.parameters():
            p.requires_grad = False

        self.ln_pre = image_model.ln_pre
        self.blocks = image_model.transformer.resblocks
        self.ln_post = image_model.ln_post
        self.visual_embed_depth = image_model.transformer.layers
        self.cls_token = image_model.class_embedding
        self.pos_embed = image_model.positional_embedding
        self.conv1 = image_model.conv1
        self.embed_dim = model.vision_width
        self.output_dim = image_model.output_dim
        self.proj = image_model.proj
        self.patch_size = model.vision_patch_size

    def embeddings(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed.to(x.dtype)
        x = self.ln_pre(x)
        return x

    def encoder(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def post_process(self, x):
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.post_process(x)
        return x
