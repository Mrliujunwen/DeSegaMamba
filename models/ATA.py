import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ATA(nn.Module):
    """
    Attentive Token Aggregation (ATA) module
    As described in the DeSegaMamba paper.
    This implementation replaces the original ATA module while maintaining a compatible interface.
    """

    def __init__(self, dim, ctx_dim, S=7, K=32, num_heads=12):
        """
        Args:
            dim (int): Input feature dimension.
            ctx_dim (int): Context feature dimension.
            S (int): Spatial size for Key pooling, as per the paper.
            K (int): Spatial size for Value pooling, as per the paper.
            num_heads (int): Number of attention heads (G in the paper).
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."

        self.dim = dim
        self.ctx_dim = ctx_dim
        self.S = S
        self.K = K
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for Q (from feature map x) and K (from context ctx)
        self.wq = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.wk = nn.Conv2d(ctx_dim, dim, kernel_size=1, bias=False)

        # Adaptive pooling layers for Key and Value generation
        self.key_pool = nn.AdaptiveAvgPool2d((S, S))
        self.value_pool = nn.AdaptiveAvgPool2d((K, K))

        # Learnable matrix Wd for intra-row dependency modeling
        self.wd = nn.Linear(S * S, K * K, bias=False)

        # Trainable ratio for Top-K gating, initialized to 2/3 as per the paper
        self.topk_ratio = nn.Parameter(torch.tensor(2.0 / 3.0))

        # Final output projection
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, ctx):
        """
        Args:
            x (torch.Tensor): input feature map (B, C, H, W)
            ctx (torch.Tensor): context prior (B, C_ctx, H, W)
        Returns:
            torch.Tensor: output feature map (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1. Generate Query, Key, and Value
        # Q from x: (B, C, H, W) -> (B, G, H*W, C/G)
        q = self.wq(x)
        q = rearrange(q, 'b (g c) h w -> b g (h w) c', g=self.num_heads)

        # K from ctx: (B, C_ctx, H, W) -> (B, C, S, S) -> (B, G, S*S, C/G)
        k = self.wk(ctx)
        k = self.key_pool(k)
        k = rearrange(k, 'b (g c) h w -> b g (h w) c', g=self.num_heads, h=self.S, w=self.S)

        # V from x: (B, C, H, W) -> (B, C, K, K) -> (B, G, K*K, C/G)
        v = self.value_pool(x)
        v = rearrange(v, 'b (g c) h w -> b g (h w) c', g=self.num_heads, h=self.K, w=self.K)

        # 2. Compute Affinity (Φ)
        # (B, G, HW, C/G) @ (B, G, C/G, S*S) -> (B, G, HW, S*S)
        attn_phi = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Top-K Gating
        # Clamp ratio for stability during training
        ratio = self.topk_ratio.clamp(0.01, 1.0)
        k_val = max(1, int((self.S * self.S) * ratio))

        # Get top-k values and their indices
        topk_vals, topk_indices = torch.topk(attn_phi, k=k_val, dim=-1)

        # Create sparse affinity matrix by scattering the top-k values
        sparse_attn_phi = torch.zeros_like(attn_phi, device=x.device)
        sparse_attn_phi.scatter_(-1, topk_indices, topk_vals)

        # 4. Generate Context-aware Dynamic Operators (Ω)
        # Apply learnable matrix Wd and softmax
        # (B, G, HW, S*S) -> (B, G, HW, K*K)
        dynamic_ops_omega = self.wd(sparse_attn_phi)
        dynamic_ops_omega = torch.softmax(dynamic_ops_omega, dim=-1)

        # 5. Aggregate Value
        # (B, G, HW, K*K) @ (B, G, K*K, C/G) -> (B, G, HW, C/G)
        out = dynamic_ops_omega @ v

        # 6. Reshape output to image format
        # (B, G, HW, C/G) -> (B, C, H, W)
        out = rearrange(out, 'b g (h w) c -> b (g c) h w', h=H, w=W)

        # 7. Final output with projection and residual connection (P_i = proj(P) + x_i)
        out = self.proj_out(out) + x
        return out

if __name__ == '__main__':
    # Test case to verify dimensional consistency.
    # Parameters are chosen to be compatible with the module's requirements (dim divisible by num_heads).
    dim = 96
    ctx_dim = 96
    num_heads_from_paper = 12  # G in the paper is 12

    model = ATA(dim=dim, ctx_dim=ctx_dim, num_heads=num_heads_from_paper)

    # Create dummy inputs
    x_input = torch.randn(4, dim, 56, 56)
    ctx_input = torch.randn(4, ctx_dim, 56, 56)

    # Forward pass
    output = model(x_input, ctx_input)

    # Verify output dimensions
    print(f"Input shape (x): {x_input.shape}")
    print(f"Input shape (ctx): {ctx_input.shape}")
    print(f"Output shape: {output.shape}")
    assert x_input.shape == output.shape, "Error: Mismatch between input and output shapes!"
    print("\nSuccess: Input and output dimensions are consistent.")

