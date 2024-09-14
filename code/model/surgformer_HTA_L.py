# 模型的base版本，重构了TimeSFormer模型，基本上与TimeSFormer模型结果一致；
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/syangcw/SurgVU/code")
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Spatial(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, B):
        B, N, C = x.shape
        # For Intra-Spatial: (BT, heads, K, C)
        # Atten: K*K, Values: K*C
        q = (
                self.q(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(
            x,
            "(b t) num_heads k c -> (b t) k (num_heads c)",
            b=B,
        )
        x = self.proj(x)
        return x


class Attention_Temporal(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q_4 = nn.Linear(dim, dim, bias=True)
        self.k_4 = nn.Linear(dim, dim, bias=True)
        self.v_4 = nn.Linear(dim, dim, bias=True)
        self.q_8 = nn.Linear(dim, dim, bias=True)
        self.k_8 = nn.Linear(dim, dim, bias=True)
        self.v_8 = nn.Linear(dim, dim, bias=True)
        self.q_16 = nn.Linear(dim, dim, bias=True)
        self.k_16 = nn.Linear(dim, dim, bias=True)
        self.v_16 = nn.Linear(dim, dim, bias=True)
        self.proj_4 = nn.Linear(dim, dim)
        self.proj_8 = nn.Linear(dim, dim)
        self.proj_16 = nn.Linear(dim, dim)

    def forward(self, x, B):
        BK, T, C = x.shape
        t1 = T // 4
        t2 = T // 2
        x_4 = x[: , T-t1: , ]
        x_8 = x[: , t2: , ]
        x_16 = x
        K = BK // B
        
        q_4 = self.q_4(x_4)
        k_4 = self.k_4(x_4)
        v_4 = self.v_4(x_4)

        q_4 = rearrange(
            q_4,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        k_4 = rearrange(
            k_4,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        v_4 = rearrange(
            v_4,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )

        q_8 = self.q_8(x_8)
        k_8 = self.k_8(x_8)
        v_8 = self.v_8(x_8)

        q_8 = rearrange(
            q_8,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        k_8 = rearrange(
            k_8,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        v_8 = rearrange(
            v_8,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )

        q_16 = self.q_16(x_16)
        k_16 = self.k_16(x_16)
        v_16 = self.v_16(x_16)

        q_16 = rearrange(
            q_16,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        k_16 = rearrange(
            k_16,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        v_16 = rearrange(
            v_16,
            "b t (num_heads c) -> b num_heads t c",
            num_heads=self.num_heads,
        )
        
        attn_4 = (q_4 @ k_4.transpose(-2, -1)) * self.scale
        attn_4 = attn_4.softmax(dim=-1)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_8 = (q_8 @ k_8.transpose(-2, -1)) * self.scale
        attn_8 = attn_8.softmax(dim=-1)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_16 = (q_16 @ k_16.transpose(-2, -1)) * self.scale
        attn_16 = attn_16.softmax(dim=-1)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        x_4 = self.proj_4(x_4)
        x_8[:, t1:, :] = 0.5 * x_8[:, t1:, :] + 0.5 * x_4
        x_8 = self.proj_8(x_8)
        x_16[:, t2: , :] = 0.5 * x_16[:, t2: , :] + 0.5 * x_8
        x_16 = self.proj_16(x_16)

        return x_16


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Spatial(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention_Temporal(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T, K):
        # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
        # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
        B, M, C = x.shape
        assert T * K + 1 == M

        # Temporal_Self_Attention
        xt = x[:, 1:, :]
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)

        res_temporal = self.drop_path(
            self.temporal_attn.forward(self.temporal_norm1(xt), B)
        )

        res_temporal = rearrange(
                res_temporal, "(b k) t c -> b (k t) c", b=B
            )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
        xt = self.temporal_fc(res_temporal) + x[:, 1:, :]

        # Spatial_Self_Attention
        init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
        cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
        cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
        res_spatial = self.drop_path(self.attn.forward(self.norm1(xs), B))

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
        cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
        res_spatial = res_spatial[:, 1:, ]  # BT, xK, C
        res_spatial = rearrange(
            res_spatial, "(b t) k c -> b (k t) c", b=B)
        res = res_spatial
        x = xt
        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
        )
        # 直接使用3D卷积来映射时序帧到视频序列tokens，在过程中进行Temporal Sample
        # 对于逐帧计算的Tool以及Phase，怎么处理模型结构的变化？降低视频序列长度并且放弃时序采样

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)
        x = rearrange(x, "(b t) c k -> b t k c", b=B)

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        all_frames=8,
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
        )
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // all_frames, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if "Block" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        # B, C, T, H, W
        x = self.patch_embed(x)
        # B, T, K, C
        B, T, K, C = x.size()
        W = int(math.sqrt(K))

        # 添加Spatial Position Embedding
        x = rearrange(x, "b t k c -> (b t) k c")
        x = x + self.pos_embed  # BT, HW, C  ---> 2*8, 196, 768
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # BT, 1, C
        x = torch.cat((cls_tokens, x), dim=1)  # BT, HW+1, C  ---> 2*8, 196+1, 768
        x = self.pos_drop(x)

        # 添加Temporal Position Embedding
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]  # 过滤掉cls_tokens
        x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
        x = x + self.time_embed  # BK, T, C  ---> 2*196, 8, 768
        x = self.time_drop(x)

        # 添加Cls token
        x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens
        x = torch.cat((cls_tokens, x), dim=1)  # 时空tokens对应的class token的添加；

        for blk in self.blocks:
            x = blk(x, B, T, K)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def surgformer_HTA_L(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()
        
        add_list = []
        for k in state_dict.keys():
            if "blocks" in k and "q_4" in k and "temporal_attn" in k:
                k_init = k.replace("q_4", "q")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "k_4" in k and "temporal_attn" in k:
                k_init = k.replace("k_4", "k")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "v_4" in k and "temporal_attn" in k:
                k_init = k.replace("v_4", "v")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "q_8" in k and "temporal_attn" in k:
                k_init = k.replace("q_8", "q")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "k_8" in k and "temporal_attn" in k:
                k_init = k.replace("k_8", "k")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "v_8" in k and "temporal_attn" in k:
                k_init = k.replace("v_8", "v")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "q_16" in k and "temporal_attn" in k:
                k_init = k.replace("q_16", "q")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "k_16" in k and "temporal_attn" in k:
                k_init = k.replace("k_16", "k")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "v_16" in k and "temporal_attn" in k:
                k_init = k.replace("v_16", "v")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "proj_4" in k and "temporal_attn" in k:
                k_init = k.replace("proj_4", "proj")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "proj_8" in k and "temporal_attn" in k:
                k_init = k.replace("proj_8", "proj")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "proj_16" in k and "temporal_attn" in k:
                k_init = k.replace("proj_16", "proj")
                k_init = k_init.replace("temporal_attn", "attn")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)
            if "blocks" in k and "temporal_norm1" in k:
                k_init = k.replace("temporal_norm1", "norm1")
                if k_init in checkpoint:
                    checkpoint[k] = checkpoint[k_init]
                    add_list.append(k)

        print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
        utils.load_state_dict(model, checkpoint)

    return model


if __name__ == "__main__":
    import utils
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    model = surgformer_HTA_L(
        pretrained=True,
        pretrain_path="/home/syangcw/SurgVU/code/pretrain_params/vit_large_patch16_224_surgery.pth"
    ).cuda()
    a = torch.rand((2,3,8,224,224)).cuda()
    y = model(a)
    print(y.shape)