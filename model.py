import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm and empirically works well in transformers.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return x / (norm + self.eps) * self.weight


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


class AdaLNZero(nn.Module):
    """Adaptive RMSNorm with zero initialization.
    
    This layer normalizes its input using RMSNorm and then applies an affine transformation
    parameterized by the conditioning vector. Initialized such that the
    transformation is identity at initialization.
    """
    
    def __init__(self, dim: int, emb_dim: int):
        """
        Args:
            dim: Dimension of the input features
            emb_dim: Dimension of the conditioning embedding
        """
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(emb_dim, 2 * dim)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            emb: Conditioning embedding of shape (batch, emb_dim)
        
        Returns:
            Normalized and scaled tensor of shape (batch, seq_len, dim)
        """
        # Normalize the input with RMSNorm
        x_norm = self.norm(x)
        
        # Get scale and shift from embedding
        scale_shift = self.linear(emb)
        scale, shift = rearrange(scale_shift, 'b (n d) -> n b d', n=2)
        
        # Apply scale and shift
        # Add 1 to scale so default is identity
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        
        # QK normalization
        if qk_norm:
            self.norm_q = RMSNorm(self.head_dim)
            self.norm_k = RMSNorm(self.head_dim)
        else:
            self.norm_q = None
            self.norm_k = None
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            attn_mask: Attention mask
        
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 'b s n h d -> n b h s d')
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        positions = torch.arange(seq_len, device=x.device)
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)
        
        # Apply QK normalization
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            # print(f"scores: {scores.shape} mask: {attn_mask.shape}")
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)  # Handle -inf from mask
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        output = self.proj(attn_output)
        return output, attn_weights
    
    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings.
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, head_dim)
            positions: Position indices
        
        Returns:
            Tensor with RoPE applied
        """
        seq_len, head_dim = x.shape[2], x.shape[3]
        device = x.device
        
        # Compute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        
        # Compute angles: (seq_len, head_dim//2)
        t = positions.float().unsqueeze(1)
        freqs = torch.einsum('...i,j->ij', t, inv_freq)
        
        # Build rotation matrix: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
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
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
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
        
        # Adaptive layer norms
        self.ln1 = AdaLNZero(dim, emb_dim)
        self.ln2 = AdaLNZero(dim, emb_dim)
        
        # Attention with RoPE and QK normalization
        self.attn = RoPEAttention(dim, num_heads, qk_norm=True)
        
        # SwiGLU MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            SwiGLU(dim, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # gates
        self.gates = nn.Linear(emb_dim, 2*dim)
        nn.init.constant_(self.gates.weight, 0)
        nn.init.constant_(self.gates.bias, 0)

    
    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            emb: Conditioning embedding of shape (batch, emb_dim)
            mask: Optional mask tensor of shape (batch, seq_len) with 1s for valid positions, 0s for masked
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # gates
        gate_mha, gate_mlp = self.gates(emb).chunk(2, dim=1)

        # Self-attention with RoPE and QK normalization
        x_norm = self.ln1(x, emb)
        
        # Prepare attention mask: (batch, num_heads, seq_len, seq_len)
        if mask is not None:
            attn_mask = (mask == 0).unsqueeze(1).unsqueeze(1).expand(-1, 1, mask.size(1), mask.size(1))
        else:
            attn_mask = None
        
        attn_out, _ = self.attn(x_norm, attn_mask=attn_mask)
        x = x + gate_mha.unsqueeze(1) * attn_out
        
        # MLP with AdaLNZero
        x_norm = self.ln2(x, emb)
        mlp_out = self.mlp(x_norm)
        if mask is not None:
            mlp_out = mlp_out * mask.unsqueeze(-1)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x



### Base+Refiner model

class BaseRefiner(nn.Module):
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
        depth_base: int = 6,
        depth_refiner: int = 6,
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
        self.depth_base = depth_base
        self.depth_refiner = depth_refiner
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
        
        # Patch embedding: project patches to hidden_dim
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # Note: Positional embeddings are handled by RoPE (Rotary Position Embeddings)
        # applied within the attention layers, so we don't add learned positional embeddings
        # This is kept for potential future use or reference
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Class embedding network (if conditional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes+1, emb_dim)
        
        # Combined embedding projection
        self.embed_proj = nn.Linear(emb_dim, emb_dim)
        
        # Stack of DiT blocks
        self.base_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, emb_dim)
            for _ in range(depth_base)
        ])
        self.refiner_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, emb_dim)
            for _ in range(depth_refiner)
        ])
        
        # Final layer norm and output projection
        self.base_final_ln = AdaLNZero(hidden_dim, emb_dim)
        self.refiner_final_ln = AdaLNZero(hidden_dim, emb_dim)
        
        # Output projection to image space
        out_channels = 2 * in_channels if learn_sigma else in_channels
        self.base_out_proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        self.refiner_out_proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        
        # register
        self.registers = nn.Parameter(0.02*torch.randn(1, self.n_register, hidden_dim), requires_grad=True)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize output projection
        nn.init.constant_(self.base_out_proj.weight, 0)
        nn.init.constant_(self.base_out_proj.bias, 0)
        nn.init.constant_(self.refiner_out_proj.weight, 0)
        nn.init.constant_(self.refiner_out_proj.bias, 0)
    
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
        t_emb = self.time_embed(t)  # (batch, emb_dim)
        
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
        
        #### BASE
        # Apply transformer blocks
        for block in self.base_blocks:
            x_emb = block(x_emb, cond_emb)
        
        # Final layer norm
        x_emb_base = self.base_final_ln(x_emb[:, self.n_register:, :], cond_emb)
        
        # Remove class token and project to output
        x_out = self.base_out_proj(x_emb_base)  # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # Unpatchify
        out_base = self._unpatchify(x_out)  # (batch, out_channels, height, width)

        #### REFINER
        x_emb = x_emb.detach()
        block_mask = torch.cat([torch.ones(batch_size, self.n_register).to(x.device), refiner_mask], dim=1) if refiner_mask is not None else None
        # if block_mask is not None:
        #     print(f"refiner: x: {x_emb.shape} m: {block_mask.shape}")
        for block in self.refiner_blocks:
            x_emb = block(x_emb, cond_emb, mask=block_mask)

        # Final layer norm
        x_emb = self.refiner_final_ln(x_emb[:, self.n_register:, :], cond_emb)
        
        # Remove class token and project to output
        x_out = self.refiner_out_proj(x_emb)
        if refiner_mask is not None:
            x_out = x_out * refiner_mask.unsqueeze(-1) # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # Unpatchify
        out_refiner = self._unpatchify(x_out)  # (batch, out_channels, height, width)
        return out_base, out_refiner
