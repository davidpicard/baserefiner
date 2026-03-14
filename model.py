import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
from pom import PoMMixer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Paper Appendix A).
    
    Uses variance-based normalization instead of L2 norm.
    More stable and standard for modern transformers.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use variance (RMS) not norm: sqrt(mean(x^2))
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normalized).to(input_dtype)


class SwiGLU(nn.Module):
    """Gated Linear Unit with Swish activation.
    
    v_proj(x) * sigmoid(g_proj(x))
    More parameter-efficient and performant than GELU-based MLPs.
    """
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * mlp_dim)
        self.mlp_dim = mlp_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, gates = x[..., :self.mlp_dim], x[..., self.mlp_dim:]
        return x * torch.nn.functional.silu(gates)



class DiTBlock(nn.Module):
    """Transformer block with AdaLNZero conditioning for DiT.
    
    Implements a transformer layer with:
    - RMSNorm pre-normalization
    - Polynomial Mixer (PoM) with 2D RoPE instead of attention
    - Feed-forward network with SwiGLU
    - Residual connections with learned gates
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, emb_dim: int = None,
                 pom_degree: int = 3, pom_expand: int = 2):
        """
        Args:
            dim: Dimension of features
            num_heads: Number of attention heads (kept for compatibility, not used by PoM)
            mlp_ratio: Ratio of mlp hidden dimension to dim
            emb_dim: Dimension of conditioning embedding (if None, uses dim)
            pom_degree: Polynomial degree for PoM (2, 3, or 4)
            pom_expand: Expansion factor for PoM polynomial feature space
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads  # Stored for compatibility, not used by PoM
        emb_dim = emb_dim or dim
        
        # Adaptive layer norms (just normalization, no scale/shift)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Polynomial Mixer with 2D RoPE (parameterized by degree and expand)
        self.attn = PoMMixer(dim, degree=pom_degree, expand=pom_expand)
        
        # SwiGLU MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            SwiGLU(dim, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # Modulation: generates scale and shift for AdaLN
        # Also generates gates for residual connections
        # Total: 2 (attn scale+shift) + 2 (mlp scale+shift) + 2 (attn+mlp gates) = 6*dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask: torch.Tensor = None,
                patch_shape: tuple = None, num_cls_tokens: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            emb: Conditioning embedding of shape (batch, emb_dim)
            mask: Optional mask tensor of shape (batch, seq_len) with 1s for valid positions, 0s for masked
            patch_shape: Optional tuple (height, width) of patch grid for 2D RoPE
            num_cls_tokens: Number of class/context tokens (not spatially encoded)
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Generate modulation parameters from conditioning
        # Output: [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        mod = self.adaLN_modulation(emb)  # (batch, 6*dim)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            mod.chunk(6, dim=1)  # Each: (batch, dim)
        
        # Attention block with adaptive layer norm and gating
        x_norm = self.norm1(x)
        # Apply modulation (scale and shift)
        x_norm = x_norm * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        
        # Prepare attention mask if provided
        if mask is not None:
            attn_mask = (mask == 0).unsqueeze(1).unsqueeze(1).expand(-1, 1, mask.size(1), mask.size(1))
        else:
            attn_mask = None
        
        # Pass patch_shape and num_cls_tokens to attention for 2D RoPE
        attn_out = self.attn(x_norm,
                            mask=attn_mask,
                            patch_shape=patch_shape, num_cls_tokens=num_cls_tokens)
        # Apply gated residual
        x = x + gate_attn.unsqueeze(1) * attn_out
        
        # MLP block with adaptive layer norm and gating
        x_norm = self.norm2(x)
        # Apply modulation (scale and shift)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        mlp_out = self.mlp(x_norm)
        if mask is not None:
            mlp_out = mlp_out * mask.unsqueeze(-1)
        # Apply gated residual
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x



### Base+Refiner model

class Baseline(nn.Module):
    """Diffusion Transformer (DiT) model with AdaLNZero conditioning.
    
    Full architecture including:
    - Patch embedding for image inputs
    - Learned positional embeddings
    - Time/condition embedding projection
    - Stack of transformer blocks
    - Output projection to target space
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        emb_dim: int = 512,
        num_classes: int = None,
        learn_sigma: bool = False,
        prediction: str = "x",
        pom_degree: int = 3,
        pom_expand: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.prediction = prediction
        self.pom_degree = pom_degree
        self.pom_expand = pom_expand
        self.n_register = 16
        
        # Calculate number of patches
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding with bottleneck for manifold learning
        # Paper (Figure 4) shows bottleneck embedding improves x-prediction
        # Uses low-rank decomposition: raw_patch -> bottleneck -> hidden_dim
        bottleneck_dim = 2 * patch_dim // 3  # Paper shows 128-256 works well
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, bottleneck_dim),
            nn.Linear(bottleneck_dim, hidden_dim)
        )
        
        # Note: Positional embeddings are handled by RoPE (Rotary Position Embeddings)
        # applied within the attention layers, so we don't add learned positional embeddings
        # This is kept for potential future use or reference
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Class embedding network (if conditional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes+1, emb_dim)
        
        # Combined embedding projection
        self.embed_proj = nn.Linear(emb_dim, emb_dim)
        
        # In-context class conditioning tokens (Paper Section 4.4, Appendix A)
        # Prepends learnable class tokens to sequence at intermediate layers
        # Shows ~1.2 FID improvement according to Table 4
        num_in_context_tokens = 32
        self.in_context_class_tokens = nn.Parameter(
            0.02 * torch.randn(1, num_in_context_tokens, hidden_dim), 
            requires_grad=True
        )
        # Start block to insert tokens (depends on model size)
        # JiT-B: 4, JiT-L: 8, JiT-H: 10, JiT-G: 10 (Table 9)
        self.in_context_start_block = 4  # Default for Base model
        self.num_in_context_tokens = num_in_context_tokens
        
        # Stack of DiT blocks
        self.base_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, emb_dim, 
                    pom_degree=pom_degree, pom_expand=pom_expand)
            for _ in range(depth)
        ])
        
        # Final layer norm and output projection
        self.base_final_ln = RMSNorm(hidden_dim)
        
        # Output projection to image space
        out_channels = 2 * in_channels if learn_sigma else in_channels
        self.base_out_proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        
        # register
        self.registers = nn.Parameter(0.02*torch.randn(1, self.n_register, hidden_dim), requires_grad=True)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding (bottleneck sequential with two linear layers)
        for module in self.patch_embed:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Initialize output projection
        nn.init.constant_(self.base_out_proj.weight, 0)
        nn.init.constant_(self.base_out_proj.bias, 0)
    
    def _get_vt_from_x0(
            self, 
            x0: torch.Tensor,
            xt: torch.Tensor,
            t: torch.Tensor):
        # Ensure t is proper shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        return (x0 - xt)/(1-t).clamp(min=0.05)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches.
        
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        
        Returns:
            Patch tensor of shape (batch, num_patches, patch_dim)
        """
        batch, channels, height, width = x.shape
        
        # Reshape to patches
        x = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        return x
    
    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image.
        
        Args:
            x: Patch tensor of shape (batch, num_patches, patch_dim)
        
        Returns:
            Image tensor of shape (batch, channels, height, width)
        """
        batch = x.shape[0]
        channels = 2 * self.in_channels if self.learn_sigma else self.in_channels
        
        # Reshape from patches
        x = rearrange(
            x,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.input_size // self.patch_size,
            w=self.input_size // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=channels
        )
        return x
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _expand_mask_to_image(self, mask):
        return self._unpatchify(mask.unsqueeze(-1).expand(-1, -1, 3*self.patch_size**2))
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None,
        refiner_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (batch, channels, height, width)
            t: Diffusion time step of shape (batch,) or (batch, 1) in range [0, 1]
            y: Class label tensor of shape (batch,) (optional, for conditional generation)
        
        Returns:
            Denoising target of shape (batch, channels, height, width) or
            (batch, 2*channels, height, width) if learn_sigma=True
        """
        batch_size = x.shape[0]
        
        # Ensure t is correct shape
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (batch, 1)
        
        # Patchify input
        x_patch = self._patchify(x)  # (batch, num_patches, patch_dim)
        
        # Embed patches
        x_emb = self.patch_embed(x_patch)  # (batch, num_patches, hidden_dim)
        
        # Note: Positional information is handled by RoPE in the attention layers
        
        # Time embedding
        t_emb = self.timestep_embedding(t, self.emb_dim)
        t_emb = self.time_embed(t_emb)  # (batch, emb_dim)
        
        # Optionally add class embedding
        if y is None:
            y = torch.ones(1,).to(torch.long).to(x_emb.device) * self.num_classes
        if self.class_embed is not None:
            y_emb = self.class_embed(y)  # (batch, emb_dim)
            t_emb = t_emb + y_emb
        
        # Project combined embedding
        cond_emb = self.embed_proj(t_emb)  # (batch, emb_dim)

        # add registers
        x_emb = torch.cat([self.registers.expand(batch_size, -1, -1), x_emb], dim=1)
        
        # Calculate patch grid shape for 2D RoPE
        patch_grid_h = self.input_size // self.patch_size
        patch_grid_w = self.input_size // self.patch_size
        patch_shape = (patch_grid_h, patch_grid_w)

        #### BASE
        # Apply transformer blocks with in-context class token insertion
        for block_idx, block in enumerate(self.base_blocks):
            # Insert in-context class tokens at intermediate block (Paper Appendix A)
            if block_idx == self.in_context_start_block:
                class_tokens = self.in_context_class_tokens.expand(batch_size, -1, -1)
                x_emb = torch.cat([x_emb, class_tokens], dim=1)
            
            # Pass patch_shape for 2D RoPE and num_registers for identity padding
            x_emb = block(x_emb, cond_emb, patch_shape=patch_shape, num_cls_tokens=self.n_register)
        
        # Remove in-context tokens if they were added
        if self.in_context_start_block < self.depth:
            x_emb = x_emb[:, :-self.num_in_context_tokens, :]
        
        # Final layer norm
        x_emb_base = self.base_final_ln(x_emb[:, self.n_register:, :])
        
        # Remove class token and project to output
        x_out = self.base_out_proj(x_emb_base)  # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # Unpatchify
        out_base = self._unpatchify(x_out)  # (batch, out_channels, height, width)

        return out_base
