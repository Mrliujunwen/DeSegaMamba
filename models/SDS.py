import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --- Helper Functions and Modules ---

def ema_inplace(moving_avg, new, decay):
    """
    Helper function for updating Exponential Moving Average in-place.
    """
    if moving_avg.nelement() == 0:
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class ChannelAttention(nn.Module):
    """
    Channel Attention module (Squeeze-and-Excitation block) as described in the paper.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        return self.sigmoid(out)


# --- Main Module Implementation ---

class SDS(nn.Module):
    """
    Implementation of the Semantic-driven Deformable Scanning (SDS) module.
    It combines two key components:
    1. SRPO (Sub-pixel Reference Point Offset) for feature enhancement.
    2. SSG (Semantic-Similarity-Driven State Grouper) for sequence reorganization.
    """

    def __init__(self, dim, dw_kernel_size=7, num_centers=12, ema_decay=0.999):
        """
        Args:
            dim (int): Input feature dimension.
            dw_kernel_size (int): Kernel size for the depthwise convolution in SRPO.
                                  Paper suggests [9, 7, 5, 3] for different stages.
            num_centers (int): Number of semantic centers (M) for SSG. Paper suggests 12.
            ema_decay (float): Decay rate (λ) for EMA updates of centers. Paper suggests 0.999.
        """
        super().__init__()
        self.dim = dim

        # === SRPO (Sub-pixel Reference Point Offset) Layers ===
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=dw_kernel_size, padding=dw_kernel_size // 2, groups=dim)
        self.channel_attn = ChannelAttention(dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(dim)
        self.offset_conv = nn.Conv2d(dim, 2, kernel_size=1, bias=False)  # Output 2 channels for (dx, dy)

        # Learnable absolute positional bias map R. It's a single map shared across the batch.
        self.register_parameter('R', None)

        # === SSG (Semantic-Similarity-Driven State Grouper) Components ===
        self.num_centers = num_centers
        self.ema_decay = ema_decay
        self.register_buffer('centers', torch.randn(num_centers, dim))
        self.register_buffer('initted', torch.tensor(False))

    def forward(self, x):
        """
        Forward pass for the SDS module.
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
        Returns:
            tuple: A tuple containing:
                - output_map (torch.Tensor): Permuted and enhanced feature map (B, C, H, W).
                - perm_indices (torch.Tensor): The permutation map σ (B, HW).
                - centers (torch.Tensor): The learned semantic centers (M, C).
        """
        B, C, H, W = x.shape

        # --- SRPO Path: Feature Enhancement ---
        # 1. Generate spatial offsets Δo
        offset_features = self.dw_conv(x)
        offset_features = offset_features * self.channel_attn(offset_features)
        offset_features = self.gelu(offset_features)

        offset_features_seq = rearrange(offset_features, 'b c h w -> b (h w) c')
        offset_features_seq = self.ln(offset_features_seq)
        offset_features = rearrange(offset_features_seq, 'b (h w) c -> b c h w', h=H, w=W)

        offsets = self.offset_conv(offset_features)  # (B, 2, H, W)

        # 2. Create reference grid and apply offsets to get sub-pixel coordinates o'
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=x.dtype, device=x.device),
            torch.linspace(-1, 1, W, dtype=x.dtype, device=x.device),
            indexing='ij'
        )
        reference_grid = torch.stack((ref_x, ref_y), -1).unsqueeze(0).expand(B, -1, -1, -1)
        offset_grid = reference_grid + offsets.permute(0, 2, 3, 1)

        # 3. Sample features and positional bias using bilinear interpolation (Ψ)
        sampled_features = F.grid_sample(
            x, offset_grid, mode='bilinear', padding_mode='border', align_corners=False
        )

        if self.R is None:
            self.R = nn.Parameter(torch.zeros(1, 1, H, W, device=x.device))

        # --- FIX: Expand self.R to match the batch size of offset_grid ---
        # Original R has shape [1, 1, H, W]. We expand it to [B, 1, H, W].
        expanded_R = self.R.expand(B, -1, -1, -1)
        sampled_bias = F.grid_sample(
            expanded_R, offset_grid, mode='bilinear', padding_mode='border', align_corners=False
        )

        # 4. Formulate the final SRPO output x̄
        x_bar = sampled_features + sampled_bias

        # --- SSG Path: Sequence Reorganization ---
        x_seq = rearrange(x, 'b c h w -> b (h w) c').detach()

        if not self.initted and self.training:
            indices = torch.randperm(B * H * W)[:self.num_centers]
            initial_centers = x_seq.reshape(-1, C)[indices]
            self.centers.data.copy_(F.normalize(initial_centers, dim=-1))
            self.initted.data.copy_(torch.tensor(True))

        sim_scores = torch.einsum('bnc,mc->bnm', F.normalize(x_seq, dim=-1), F.normalize(self.centers, dim=-1))
        group_idx = torch.argmax(sim_scores, dim=-1)

        if self.training:
            with torch.no_grad():
                one_hot_assign = F.one_hot(group_idx, num_classes=self.num_centers).float()
                token_counts_per_center = one_hot_assign.sum(dim=1)
                summed_tokens = torch.einsum('bnc,bnm->bmc', x_seq, one_hot_assign)
                batch_centroids = summed_tokens / (token_counts_per_center.unsqueeze(-1) + 1e-8)
                new_centers = batch_centroids.mean(dim=0)
                if not torch.isnan(new_centers).any():
                    ema_inplace(self.centers, F.normalize(new_centers, dim=-1), self.ema_decay)

        perm_indices = torch.argsort(group_idx, dim=1)

        # --- Final Combination ---
        x_bar_seq = rearrange(x_bar, 'b c h w -> b (h w) c')
        perm_indices_expanded = perm_indices.unsqueeze(-1).expand(-1, -1, C)
        permuted_seq = torch.gather(x_bar_seq, 1, perm_indices_expanded)

        output_map = rearrange(permuted_seq, 'b (h w) c -> b c h w', h=H, w=W)

        return output_map, perm_indices, self.centers


if __name__ == '__main__':
    dim = 64
    num_centers_from_paper = 12
    batch_size = 4
    height, width = 32, 32

    sds_module = SDS(dim=dim, num_centers=num_centers_from_paper, dw_kernel_size=9).cuda()
    sds_module.train()

    input_tensor = torch.rand(batch_size, dim, height, width).to('cuda')

    output, p_indices, c_centers = sds_module(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output feature map shape: {output.shape}")
    print(f"Permutation indices shape: {p_indices.shape}")
    print(f"Centers shape: {c_centers.shape}")

    assert input_tensor.shape == output.shape, "Error: Mismatch between input and output feature map shapes!"
    assert p_indices.shape == (batch_size, height * width), "Error: Incorrect shape for permutation indices!"
    assert c_centers.shape == (num_centers_from_paper, dim), "Error: Incorrect shape for centers!"
    print("\nSuccess: All output dimensions are consistent with the design.")
