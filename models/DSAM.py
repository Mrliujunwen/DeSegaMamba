import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 导入 Mamba 核心扫描函数
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not found. Falling back to selective_scan_fn_v1.")
    try:
        from selective_scan import selective_scan_fn as selective_scan_fn_v1

        selective_scan_fn = selective_scan_fn_v1
    except ImportError:
        print("Error: selective_scan is not installed. Please install it to use this model.")
        selective_scan_fn = None

# 导入我们论文中定义的模块
# 为了无缝集成，我们保留了原始文件名，但使用 'as' 关键字赋予其在本文档中更清晰的名称
from models.ATA import ATA
from models.SDS import SDS

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer. """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    r""" Patch Expansion Layer. """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, self.dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // self.dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    r""" Final Patch Expansion Layer. """

    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, self.dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // self.dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class DSAM_mixer(nn.Module):
    """
    Deformable Semantic Aggregation Mamba Mixer.
    This module replaces the original SS2D and implements the core state-space
    model logic with a single, efficient scan, as described in the DeSegaMamba paper.
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Projection for SSM parameters (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # Projection for Δ from rank
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Initialize Δ projection
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D-like A matrix
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ATA prompt projection to match dimension of C
        self.prompt_proj = nn.Linear(self.d_model, self.d_state, bias=False)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x, prompt):
        B, H, W, C = x.shape
        L = H * W

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, D_inner)

        # 2D convolution
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # Prepare for single scan
        x = x.view(B, self.d_inner, L).permute(0, 2, 1).contiguous()  # (B, L, D_inner)

        # Project to get SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Compute Δ
        dt = self.dt_proj(dt).permute(0, 2, 1)  # (B, D_inner, L)

        # Permute B and C to match selective_scan_fn's expected shape
        B_ssm = B_ssm.permute(0, 2, 1).contiguous()  # (B, d_state, L)
        C_ssm = C_ssm.permute(0, 2, 1).contiguous()  # (B, d_state, L)

        # Inject the non-causal prompt from ATA into C
        # prompt shape: (B, H, W, d_model) -> (B, L, d_model) -> (B, L, d_state)
        prompt_c = self.prompt_proj(prompt.view(B, L, -1)).permute(0, 2, 1)  # (B, d_state, L)
        C_ssm = C_ssm + prompt_c

        # Perform the selective scan
        A = -torch.exp(self.A_log.float())  # (D_inner, d_state)
        y = selective_scan_fn(
            x.permute(0, 2, 1).contiguous(), dt, A, B_ssm, C_ssm, self.D, z=None,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )

        # Output projection
        y = y.permute(0, 2, 1).contiguous()  # (B, L, D_inner)
        y = self.out_norm(y).view(B, H, W, -1)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class DSAM_block(nn.Module):
    """
    Deformable Semantic Aggregation Mamba (DSAM) block.
    This is the core building block of DeSegaMamba, replacing VSSBlock.
    It orchestrates SDS, ATA, and the DSAM_mixer.
    """

    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 16,
            dw_kernel_size: int = 7,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.sds = SDS(dim=hidden_dim, dw_kernel_size=dw_kernel_size)
        self.ata = ATA(dim=hidden_dim, ctx_dim=hidden_dim)
        self.mixer = DSAM_mixer(d_model=hidden_dim, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        B, H, W, C = input.shape
        L = H * W

        # Normalize input
        x_norm = self.ln_1(input)
        x_2d = x_norm.permute(0, 3, 1, 2).contiguous()

        # SDS for semantic-driven sequence reorganization
        sds_output, perm_indices, _ = self.sds(x_2d)
        sds_output_seq = sds_output.permute(0, 2, 3, 1).contiguous()

        # ATA for non-causal prompt generation
        prompt = self.ata(x_2d, x_2d)  # Using self as context
        prompt_seq = prompt.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        # Permute the prompt to align with the SDS-reordered sequence
        perm_indices_exp = perm_indices.unsqueeze(-1).expand(-1, -1, C)
        prompt_permuted = torch.gather(prompt_seq, 1, perm_indices_exp).view(B, H, W, C)

        # Core mixing operation
        mixer_output_permuted = self.mixer(sds_output_seq, prompt_permuted)

        # Un-permute the output sequence to restore original spatial order
        mixer_output_permuted_seq = mixer_output_permuted.view(B, L, C)
        unperm_indices = torch.argsort(perm_indices, dim=1).unsqueeze(-1).expand(-1, -1, C)
        mixer_output = torch.gather(mixer_output_permuted_seq, 1, unperm_indices).view(B, H, W, C)

        # Residual connection
        output = input + self.drop_path(mixer_output)
        return output


class DSAMLayer(nn.Module):
    """ A basic DeSegaMamba layer for one stage. """

    def __init__(
            self,
            dim,
            depth,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            dw_kernel_size=7,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            DSAM_block(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                d_state=d_state,
                dw_kernel_size=dw_kernel_size,
                **kwargs
            )
            for i in range(depth)])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DSAMLayer_up(nn.Module):
    """ A basic DeSegaMamba layer for one up-sampling stage. """

    def __init__(
            self,
            dim,
            depth,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            dw_kernel_size=7,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None

        self.blocks = nn.ModuleList([
            DSAM_block(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                d_state=d_state,
                dw_kernel_size=dw_kernel_size,
                **kwargs
            )
            for i in range(depth)])

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x


class DeSegaMamba(nn.Module):
    """
    DeSegaMamba: Rethinking Vision Mamba Design for Medical Image Segmentation
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96],
                 d_state=16, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False,
                 dw_kernel_sizes=[9, 7, 5, 3], dw_kernel_sizes_decoder=[3, 5, 7, 9],
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.patch_norm = patch_norm

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DSAMLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                dw_kernel_size=dw_kernel_sizes[i_layer],
            )
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DSAMLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                dw_kernel_size=dw_kernel_sizes_decoder[i_layer],
            )
            self.layers_up.append(layer)

        # Final layers for segmentation output
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_connections = []
        x = self.patch_embed(x)
        for layer in self.layers:
            skip_connections.append(x)
            x = layer(x)
        return x, skip_connections

    def forward_up_features(self, x, skip_connections):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_connections[-i])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_connections = self.forward_features(x)
        x = self.forward_up_features(x, skip_connections)
        x = self.forward_final(x)
        return x
