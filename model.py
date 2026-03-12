import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math


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


class RoPEAttention(nn.Module):
    """Multi-head attention with 2D Rotary Position Embeddings (RoPE).
    
    Supports 2D spatial RoPE for image patches with proper handling of
    class/in-context tokens that shouldn't have spatial positions.
    
    Key implementation details:
    - Normalize Q, K BEFORE applying RoPE (not after)
    - Use 2D spatial coordinates for patches
    - Apply identity rotation to class tokens (no spatial encoding)
    """
    
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        
        # QK normalization (applied BEFORE RoPE per reference implementation)
        if qk_norm:
            self.norm_q = RMSNorm(self.head_dim)
            self.norm_k = RMSNorm(self.head_dim)
        else:
            self.norm_q = None
            self.norm_k = None
        
        # Cache for 2D position embeddings
        self._rope_cache = {}
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None,
                patch_shape: tuple = None, num_cls_tokens: int = 0) -> tuple:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            attn_mask: Attention mask (boolean, True for positions to mask out)
            patch_shape: Optional tuple (height, width) of patch grid for 2D RoPE
            num_cls_tokens: Number of class/context tokens (not spatially encoded)
        
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 'b s n h d -> n b h s d')
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # CRITICAL: Apply QK normalization BEFORE RoPE (per reference implementation)
        # This ensures rotation angles are applied to normalized vectors
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Determine if using 2D spatial RoPE
        if patch_shape is not None and len(patch_shape) == 2:
            # 2D spatial RoPE for image patches
            q = self._apply_rope_2d(q, patch_shape, num_cls_tokens=num_cls_tokens)
            k = self._apply_rope_2d(k, patch_shape, num_cls_tokens=num_cls_tokens)
        else:
            # Fall back to 1D RoPE
            positions = torch.arange(seq_len, device=x.device)
            q = self._apply_rope(q, positions)
            k = self._apply_rope(k, positions)
        
        # Use scaled_dot_product_attention for efficient attention computation
        # Convert mask to boolean if needed (True for positions to attend away from)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask == 0
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            scale=self.scale
        )
        
        # Rearrange output back to sequence format
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        # Project to output dimension
        output = self.proj(attn_output)
        
        return output
    
    def _apply_rope_2d(self, x: torch.Tensor, patch_shape: tuple,
                      num_cls_tokens: int = 0) -> torch.Tensor:
        """Apply 2D rotary position embeddings for spatially-aware patches.
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim)
            patch_shape: Tuple (height, width) of the patch grid
            num_cls_tokens: Number of class tokens at start (get identity rotation)
        
        Returns:
            Tensor with 2D RoPE applied
        """
        height, width = patch_shape
        num_patches = height * width
        head_dim = x.shape[3]
        device = x.device
        dtype = x.dtype
        
        # Generate 2D position embeddings
        cos, sin = self._get_2d_rope_cache(height, width, head_dim, num_cls_tokens, device, dtype)
        
        # Ensure tensors are correct size
        seq_len = x.shape[2]
        if cos.shape[2] < seq_len:
            # Pad if needed (e.g., with in-context tokens)
            padding = seq_len - cos.shape[2]
            cos = torch.cat([cos, torch.ones(1, 1, padding, head_dim, device=device, dtype=dtype)], dim=2)
            sin = torch.cat([sin, torch.zeros(1, 1, padding, head_dim, device=device, dtype=dtype)], dim=2)
        else:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        
        # Apply rotation
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        return x_rot
    
    def _get_2d_rope_cache(self, height: int, width: int, head_dim: int,
                          num_cls_tokens: int, device: torch.device, dtype: torch.dtype):
        """Generate 2D RoPE embeddings with caching."""
        cache_key = (height, width, head_dim, num_cls_tokens)
        
        if cache_key in self._rope_cache:
            cos, sin = self._rope_cache[cache_key]
            return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)
        
        # Create 2D position grid
        y, x_grid = torch.meshgrid(torch.arange(height, device=device),
                                   torch.arange(width, device=device), indexing='ij')
        positions_flat = torch.stack([y.flatten(), x_grid.flatten()], dim=1)  # (num_patches, 2)
        
        # Compute RoPE for each spatial dimension separately
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        
        # RoPE for height dimension (even indices)
        freqs_h = torch.einsum('i,j->ij', positions_flat[:, 0], inv_freq)
        # RoPE for width dimension (odd indices)
        freqs_w = torch.einsum('i,j->ij', positions_flat[:, 1], inv_freq)
        
        # Interleave frequencies: [h_freqs_0, w_freqs_0, h_freqs_1, w_freqs_1, ...]
        freqs = torch.zeros(height * width, head_dim, device=device, dtype=torch.float32)
        freqs[:, 0::2] = freqs_h  # Even dims: height
        if head_dim > 1:
            freqs[:, 1::2] = freqs_w  # Odd dims: width
        
        # Compute sin/cos - apply directly without duplication
        # The RoPE formula uses both sin and cos via rotate_half mechanism
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)  # (1, 1, num_patches, head_dim)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        
        # Add identity padding for class tokens
        if num_cls_tokens > 0:
            cos_cls = torch.ones(1, 1, num_cls_tokens, head_dim, device=device, dtype=dtype)
            sin_cls = torch.zeros(1, 1, num_cls_tokens, head_dim, device=device, dtype=dtype)
            cos = torch.cat([cos_cls, cos], dim=2)  # (1, 1, num_cls + num_patches, head_dim)
            sin = torch.cat([sin_cls, sin], dim=2)
        
        # Cache
        self._rope_cache[cache_key] = (cos.to(dtype=torch.float32), sin.to(dtype=torch.float32))
        
        return cos, sin
    
    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply 1D rotary position embeddings (fallback for non-spatial sequences).
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim)
            positions: Position indices 0..seq_len-1
        
        Returns:
            Tensor with RoPE applied
        """
        seq_len, head_dim = x.shape[2], x.shape[3]
        device = x.device
        
        # Compute inverse frequencies: θ_j = 10000^(-2j/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        
        # Compute angles for each position: m * θ_j where m is position
        t = positions.float().unsqueeze(1)  # (seq_len, 1)
        freqs = torch.einsum('...i,j->ij', t, inv_freq)  # (seq_len, head_dim//2)
        
        # Duplicate frequencies for sin/cos pairs
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
        
        # Apply rotation: (x * cos) + (rotate_half(x) * sin)
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        return x_rot
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLNZero conditioning for DiT.
    
    Implements a transformer layer with:
    - AdaLNZero pre-normalization for both attention and MLP
    - Multi-head self-attention with RoPE
    - Feed-forward network with SwiGLU
    - Residual connections with learned gates
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, emb_dim: int = None):
        """
        Args:
            dim: Dimension of features
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dimension to dim
            emb_dim: Dimension of conditioning embedding (if None, uses dim)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        emb_dim = emb_dim or dim
        
        # Adaptive layer norms (just normalization, no scale/shift)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Attention with RoPE and QK normalization
        self.attn = RoPEAttention(dim, num_heads, qk_norm=True)
        
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
        attn_out = self.attn(x_norm, attn_mask=attn_mask,
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
        prediction: str = "x"
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
        self.n_register = 16
        
        # Calculate number of patches
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding with bottleneck for manifold learning
        # Paper (Figure 4) shows bottleneck embedding improves x-prediction
        # Uses low-rank decomposition: raw_patch -> bottleneck -> hidden_dim
        bottleneck_dim = 128  # Paper shows 128-256 works well
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
            DiTBlock(hidden_dim, num_heads, mlp_ratio, emb_dim)
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
