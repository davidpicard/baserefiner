"""
Polynomial Mixer (PoM) - An alternative to attention mechanisms.

Based on: https://github.com/davidpicard/pom/blob/compom/pom/pom.py

PoM uses polynomial expansions to capture higher-order interactions between
input features, offering a computationally efficient alternative to softmax attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from pom_triton import pom_triton


# =============================================================================
# Core Polynomial Functions
# =============================================================================

def pom_activation(x: torch.Tensor) -> torch.Tensor:
    """LeakyReLU activation for polynomial mixer."""
    return F.leaky_relu(x, 0.01)


def po2(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Second-order polynomial expansion."""
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h = torch.cat([h, h2], dim=-1)
    return (h * coeff).sum(-1)


def po3(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Third-order polynomial expansion."""
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h = torch.cat([h, h2, h3], dim=-1)
    return (h * coeff).sum(-1)


def po4(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Fourth-order polynomial expansion."""
    h = pom_activation(x).unsqueeze(-1)
    h2 = h * h
    h3 = h2 * h
    h4 = h2 * h2
    h = torch.cat([h, h2, h3, h4], dim=-1)
    return (h * coeff).sum(-1)


# =============================================================================
# Masking and Aggregation Functions
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply 2D mask mixing for self-attention."""
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True) / (1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))


def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply 3D mask mixing for cross-attention."""
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h


# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(x: torch.Tensor,
                            coeff: torch.Tensor,
                            k: int,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply polynomial aggregation with optional masking."""
    # Use optimized functions for common cases
    if k == 2:
        h = po2(x, coeff)
    elif k == 3:
        h = po3(x, coeff)
    elif k == 4:
        h = po4(x, coeff)
    else:
        # Generic case for k > 4
        h = pom_activation(x).unsqueeze(-1)
        h = torch.cat([h ** i for i in range(1, k + 1)], dim=-1)
        h = (h * coeff).sum(-1)

    # Apply masking if provided
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim() == 2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() == 3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise ValueError(f'Unsupported mask dimension: {mask.dim()}. Expected 2 or 3.')
    return h


def polynomial_selection_(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Apply polynomial selection with sigmoid gating."""
    b, n, c = x.shape
    bb, nn, d = h.shape
    s = F.sigmoid(x).unsqueeze(-1)
    h = h.view(bb, nn, c, -1)
    return (s * h).view(b, n, d)


# =============================================================================
# Main PoM Function
# =============================================================================

def pom(xq: torch.Tensor,
        xc: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Polynomial Mixer (PoM) operation.
    
    Args:
        xq: Query input tensor of shape (batch, query_len, dim)
        xc: Context input tensor of shape (batch, context_len, dim)
        coeff: Polynomial coefficients
        k: Polynomial order (degree of interactions)
        mask: Optional attention mask
    
    Returns:
        Output tensor after polynomial mixing
    """
    h = polynomial_aggregation_(xc, coeff, k, mask)
    o = polynomial_selection_(xq, h)
    return o


# =============================================================================
# PoM Module Class
# =============================================================================

class PoM(nn.Module):
    """Polynomial Mixer (PoM) Module.
    
    A neural network layer that captures higher-order interactions between
    input features through polynomial expansions.
    
    Args:
        dim: The dimensionality of the input features
        degree: The degree of the polynomial to capture
        expand: The expansion factor for the polynomial order
        n_groups: Number of groups for grouped convolutions
        bias: Whether to include bias terms in linear projections
    """

    def __init__(self, dim: int, degree: int = 3, expand: int = 2, n_groups: int = 1, bias: bool = False):
        super().__init__()
        
        self.dim = dim
        self.order = degree
        self.order_expand = expand
        self.n_groups = n_groups
        
        assert dim % n_groups == 0, "dim must be divisible by n_groups"
        assert dim * expand % dim == 0, "dim * expand must be divisible by dim"
        
        # Linear projections
        if self.n_groups > 1:
            self.po_proj = nn.Conv1d(dim, expand * dim, kernel_size=1, bias=bias, groups=n_groups)
        else:
            self.po_proj = nn.Linear(dim, expand * dim, bias=bias)
        
        # Polynomial coefficients
        self.po_coeff = nn.Parameter((2. * torch.randn(dim * expand, degree)).clamp(-2., 2.))
        
        # Selection and aggregation projections
        self.se_proj = nn.Linear(dim, dim, bias=bias)
        self.ag_proj = nn.Linear(expand * dim, dim, bias=bias)
        
        # self.pom = pom
        self.pom = pom_triton

    def forward(self, xq: torch.Tensor,
                xc: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the PoM module.
        
        Args:
            xq: Query input tensor of shape (batch, n_tokens, dim)
            xc: Context input tensor. If None, self-attention is performed
            mask: Optional attention mask tensor
        
        Returns:
            Output tensor after applying the PoM operation
        """
        if xc is None:
            xc = xq  # self-attention
        
        s = self.se_proj(xq)
        
        if self.n_groups > 1:
            h = self.po_proj(xc.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.po_proj(xc)
        
        sh = self.pom(s, h, self.po_coeff, self.order, mask)
        
        return self.ag_proj(sh)


class PoMMixer(nn.Module):
    """PoM with 2D Rotary Position Embeddings (RoPE) support.
    
    Combines the PoM operation with 2D spatial-aware position encodings
    for processing image patches.
    
    Args:
        dim: Feature dimension
        degree: Polynomial degree for PoM
        expand: Expansion factor for polynomial features
        n_groups: Number of groups for grouped convolutions
    """
    
    def __init__(self, dim: int, degree: int = 3, expand: int = 2, n_groups: int = 1):
        super().__init__()
        self.dim = dim
        self.degree = degree
        self.expand = expand
        self.n_groups = n_groups
        
        # PoM module
        self.pom = PoM(dim, degree=degree, expand=expand, n_groups=n_groups)
        
        # Cache for 2D position embeddings
        self._rope_cache = {}
    
    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None,
                patch_shape: tuple = None,
                num_cls_tokens: int = 0,
                token_positions: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with optional 2D RoPE.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask
            patch_shape: Optional tuple (height, width) of patch grid for 2D RoPE
            num_cls_tokens: Number of class/context tokens (not spatially encoded)
            token_positions: Optional (num_spatial_tokens, 2) tensor with 2D grid positions.
                           Use when tokens are routed/sparse to apply correct RoPE based on
                           original positions, not sequential order.
        
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Apply 2D RoPE if patch_shape is provided
        if patch_shape is not None and len(patch_shape) == 2:
            x = self._apply_rope_2d(x, patch_shape, num_cls_tokens, token_positions)
        
        # Apply PoM
        output = self.pom(x, xc=x, mask=mask)
        
        return output
    
    def _apply_rope_2d(self, x: torch.Tensor, patch_shape: tuple,
                      num_cls_tokens: int = 0, token_positions: torch.Tensor = None) -> torch.Tensor:
        """Apply 2D rotary position embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            patch_shape: Tuple (height, width) of patch grid
            num_cls_tokens: Number of non-spatial tokens (at start of sequence)
            token_positions: Optional (num_spatial_tokens, 2) tensor with 2D grid positions.
                           When provided, RoPE is computed for these specific positions
                           (used for routed tokens). When None, assumes dense grid.
        
        Returns:
            Tensor with RoPE applied
        """
        batch_size, seq_len, feat_dim = x.shape
        height, width = patch_shape
        
        if token_positions is not None:
            # Use explicit token positions (for routed tokens)
            cos, sin = self._get_sparse_rope_cache(
                token_positions, feat_dim, num_cls_tokens, x.device, x.dtype
            )
        else:
            # Use dense grid positions (normal case)
            num_patches = height * width
            cos, sin = self._get_2d_rope_cache(
                height, width, feat_dim, num_cls_tokens, x.device, x.dtype
            )
        
        # Ensure correct size
        if cos.shape[2] < seq_len:
            padding = seq_len - cos.shape[2]
            cos = torch.cat([cos, torch.ones(1, 1, padding, feat_dim, device=x.device, dtype=x.dtype)], dim=2)
            sin = torch.cat([sin, torch.zeros(1, 1, padding, feat_dim, device=x.device, dtype=x.dtype)], dim=2)
        else:
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        
        # Reshape for broadcasting: (batch, seq_len, dim)
        cos = cos.squeeze(0).squeeze(0)  # (seq_len, dim)
        sin = sin.squeeze(0).squeeze(0)  # (seq_len, dim)
        
        # Apply rotation: (x * cos) + (rotate_half(x) * sin)
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        
        return x_rot
    
    def _get_2d_rope_cache(self, height: int, width: int, feat_dim: int,
                          num_cls_tokens: int, device: torch.device, dtype: torch.dtype):
        """Generate 2D RoPE embeddings with caching."""
        cache_key = (height, width, feat_dim, num_cls_tokens)
        
        if cache_key in self._rope_cache:
            cos, sin = self._rope_cache[cache_key]
            return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)
        
        # Create 2D position grid
        y, x_grid = torch.meshgrid(torch.arange(height, device=device),
                                   torch.arange(width, device=device), indexing='ij')
        positions_flat = torch.stack([y.flatten(), x_grid.flatten()], dim=1)  # (num_patches, 2)
        
        # Compute RoPE for spatial dimensions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, feat_dim, 2, device=device).float() / feat_dim))
        
        # RoPE for height dimension (even indices)
        freqs_h = torch.einsum('i,j->ij', positions_flat[:, 0], inv_freq)
        # RoPE for width dimension (odd indices)
        freqs_w = torch.einsum('i,j->ij', positions_flat[:, 1], inv_freq)
        
        # Interleave frequencies
        freqs = torch.zeros(height * width, feat_dim, device=device, dtype=torch.float32)
        freqs[:, 0::2] = freqs_h
        if feat_dim > 1:
            freqs[:, 1::2] = freqs_w
        
        # Compute sin/cos
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        
        # Add identity padding for class tokens
        if num_cls_tokens > 0:
            cos_cls = torch.ones(1, 1, num_cls_tokens, feat_dim, device=device, dtype=dtype)
            sin_cls = torch.zeros(1, 1, num_cls_tokens, feat_dim, device=device, dtype=dtype)
            cos = torch.cat([cos_cls, cos], dim=2)
            sin = torch.cat([sin_cls, sin], dim=2)
        
        # Cache
        self._rope_cache[cache_key] = (cos.to(dtype=torch.float32), sin.to(dtype=torch.float32))
        
        return cos, sin
    
    def _get_sparse_rope_cache(self, token_positions: torch.Tensor, feat_dim: int,
                              num_cls_tokens: int, device: torch.device, dtype: torch.dtype):
        """Generate 2D RoPE embeddings for sparse token positions (TREAD routing).
        
        Used when tokens are routed - we only process a subset of tokens
        at specific 2D grid positions.
        
        Args:
            token_positions: (num_spatial_tokens, 2) tensor with 2D grid positions [h, w]
            feat_dim: Feature dimension
            num_cls_tokens: Number of class/register tokens (for identity padding)
            device: Device to create tensors on
            dtype: Data type for output
        
        Returns:
            Tuple of (cos, sin) tensors with RoPE embeddings
        """
        # Move positions to the right device
        positions_flat = token_positions.to(device=device).float()  # (num_tokens, 2)
        num_tokens = positions_flat.shape[0]
        
        # Compute RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, feat_dim, 2, device=device).float() / feat_dim))
        
        # RoPE for height dimension (even indices)
        freqs_h = torch.einsum('i,j->ij', positions_flat[:, 0], inv_freq)
        # RoPE for width dimension (odd indices)  
        freqs_w = torch.einsum('i,j->ij', positions_flat[:, 1], inv_freq)
        
        # Interleave frequencies
        freqs = torch.zeros(num_tokens, feat_dim, device=device, dtype=torch.float32)
        freqs[:, 0::2] = freqs_h
        if feat_dim > 1:
            freqs[:, 1::2] = freqs_w
        
        # Compute sin/cos
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
        
        # Add identity padding for class tokens
        if num_cls_tokens > 0:
            cos_cls = torch.ones(1, 1, num_cls_tokens, feat_dim, device=device, dtype=dtype)
            sin_cls = torch.zeros(1, 1, num_cls_tokens, feat_dim, device=device, dtype=dtype)
            cos = torch.cat([cos_cls, cos], dim=2)
            sin = torch.cat([sin_cls, sin], dim=2)
        
        return cos, sin
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
