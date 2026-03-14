"""
Triton-optimized implementations of PoM operations with autograd support.

This module provides Triton kernels for forward pass and uses PyTorch's autograd
for backward pass to ensure gradient correctness.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Triton Kernels for forward pass
# =============================================================================

@triton.jit
def po2_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    N,  # batch * seq_len (total tokens)
    D,  # feature dimension
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused kernel for 2nd order polynomial expansion."""
    n_block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)
    
    n_idx = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_idx = d_block_id * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    n_mask = n_idx < N
    d_mask = d_idx < D
    
    x = tl.load(
        x_ptr + n_idx[:, None] * D + d_idx[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0
    )
    
    h = tl.where(x >= 0, x, 0.01 * x)
    h2 = h * h
    
    c0 = tl.load(coeff_ptr + d_idx * 2, mask=d_mask, other=0.0)
    c1 = tl.load(coeff_ptr + d_idx * 2 + 1, mask=d_mask, other=0.0)
    
    out = h * c0[None, :] + h2 * c1[None, :]
    
    tl.store(
        out_ptr + n_idx[:, None] * D + d_idx[None, :],
        out,
        mask=n_mask[:, None] & d_mask[None, :]
    )


@triton.jit
def po3_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused kernel for 3rd order polynomial expansion."""
    n_block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)
    
    n_idx = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_idx = d_block_id * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    n_mask = n_idx < N
    d_mask = d_idx < D
    
    x = tl.load(
        x_ptr + n_idx[:, None] * D + d_idx[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0
    )
    
    h = tl.where(x >= 0, x, 0.01 * x)
    h2 = h * h
    h3 = h2 * h
    
    c0 = tl.load(coeff_ptr + d_idx * 3, mask=d_mask, other=0.0)
    c1 = tl.load(coeff_ptr + d_idx * 3 + 1, mask=d_mask, other=0.0)
    c2 = tl.load(coeff_ptr + d_idx * 3 + 2, mask=d_mask, other=0.0)
    
    out = h * c0[None, :] + h2 * c1[None, :] + h3 * c2[None, :]
    
    tl.store(
        out_ptr + n_idx[:, None] * D + d_idx[None, :],
        out,
        mask=n_mask[:, None] & d_mask[None, :]
    )


@triton.jit
def po4_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused kernel for 4th order polynomial expansion."""
    n_block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)
    
    n_idx = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_idx = d_block_id * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    n_mask = n_idx < N
    d_mask = d_idx < D
    
    x = tl.load(
        x_ptr + n_idx[:, None] * D + d_idx[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0
    )
    
    h = tl.where(x >= 0, x, 0.01 * x)
    h2 = h * h
    h3 = h2 * h
    h4 = h2 * h2
    
    c0 = tl.load(coeff_ptr + d_idx * 4, mask=d_mask, other=0.0)
    c1 = tl.load(coeff_ptr + d_idx * 4 + 1, mask=d_mask, other=0.0)
    c2 = tl.load(coeff_ptr + d_idx * 4 + 2, mask=d_mask, other=0.0)
    c3 = tl.load(coeff_ptr + d_idx * 4 + 3, mask=d_mask, other=0.0)
    
    out = h * c0[None, :] + h2 * c1[None, :] + h3 * c2[None, :] + h4 * c3[None, :]
    
    tl.store(
        out_ptr + n_idx[:, None] * D + d_idx[None, :],
        out,
        mask=n_mask[:, None] & d_mask[None, :]
    )


# =============================================================================
# Autograd Functions for proper gradient support
# =============================================================================

class PolyExpansionTriton2(torch.autograd.Function):
    """Autograd function for 2nd order polynomial expansion with Triton forward."""
    
    @staticmethod
    def forward(ctx, x, coeff):
        """Forward pass using Triton kernel."""
        batch, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        total_tokens = batch * seq_len
        out = torch.zeros_like(x_reshaped)
        
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_D = 256
        num_n_blocks = (total_tokens + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_d_blocks = (dim + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
        grid = (num_n_blocks, num_d_blocks)
        
        coeff_flat = coeff.flatten()
        po2_kernel[grid](x_reshaped, coeff_flat, out, total_tokens, dim,
                        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_D=BLOCK_SIZE_D)
        
        out = out.reshape(batch, seq_len, dim)
        
        # Save activations for backward
        h = torch.where(x >= 0, x, 0.01 * x)  # leaky ReLU
        ctx.save_for_backward(x, coeff, h)
        ctx.k = 2
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients efficiently."""
        x, coeff, h = ctx.saved_tensors
        k = ctx.k
        
        batch, seq_len, dim = x.shape
        
        # Compute derivatives with respect to x
        # dL/dx = grad_output * (c0 + 2*h*c1) * dh/dx, where dh/dx = 1 if x>=0 else 0.01
        h2 = h * h
        
        c0 = coeff[:, 0]
        c1 = coeff[:, 1]
        
        # Compute dout/dh = c0 + 2*h*c1
        dout_dh = c0[None, None, :] + 2 * h * c1[None, None, :]
        
        # Compute dh/dx (derivative of leaky ReLU)
        dh_dx = torch.where(x >= 0, torch.ones_like(x), 0.01 * torch.ones_like(x))
        
        # Chain rule: grad_x = grad_output * dout_dh * dh_dx
        grad_x = grad_output * dout_dh * dh_dx
        
        # Compute derivatives with respect to coeff
        # dL/dc0 = sum_n(grad_output[n] * h[n])
        # dL/dc1 = sum_n(grad_output[n] * h^2[n])
        grad_coeff = torch.zeros_like(coeff)
        grad_coeff[:, 0] = (grad_output * h).sum(dim=(0, 1))
        grad_coeff[:, 1] = (grad_output * h2).sum(dim=(0, 1))
        
        return grad_x, grad_coeff


class PolyExpansionTriton3(torch.autograd.Function):
    """Autograd function for 3rd order polynomial expansion with Triton forward."""
    
    @staticmethod
    def forward(ctx, x, coeff):
        """Forward pass using Triton kernel."""
        batch, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        total_tokens = batch * seq_len
        out = torch.zeros_like(x_reshaped)
        
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_D = 256
        num_n_blocks = (total_tokens + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_d_blocks = (dim + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
        grid = (num_n_blocks, num_d_blocks)
        
        coeff_flat = coeff.flatten()
        po3_kernel[grid](x_reshaped, coeff_flat, out, total_tokens, dim,
                        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_D=BLOCK_SIZE_D)
        
        out = out.reshape(batch, seq_len, dim)
        
        # Save activations for backward
        h = torch.where(x >= 0, x, 0.01 * x)
        ctx.save_for_backward(x, coeff, h)
        ctx.k = 3
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients efficiently."""
        x, coeff, h = ctx.saved_tensors
        k = ctx.k
        
        batch, seq_len, dim = x.shape
        
        h2 = h * h
        h3 = h2 * h
        
        c0 = coeff[:, 0]
        c1 = coeff[:, 1]
        c2 = coeff[:, 2]
        
        # dout/dh = c0 + 2*h*c1 + 3*h^2*c2
        dout_dh = c0[None, None, :] + 2 * h * c1[None, None, :] + 3 * h2 * c2[None, None, :]
        
        # dh/dx (derivative of leaky ReLU)
        dh_dx = torch.where(x >= 0, torch.ones_like(x), 0.01 * torch.ones_like(x))
        
        grad_x = grad_output * dout_dh * dh_dx
        
        # Derivatives with respect to coeff
        grad_coeff = torch.zeros_like(coeff)
        grad_coeff[:, 0] = (grad_output * h).sum(dim=(0, 1))
        grad_coeff[:, 1] = (grad_output * h2).sum(dim=(0, 1))
        grad_coeff[:, 2] = (grad_output * h3).sum(dim=(0, 1))
        
        return grad_x, grad_coeff


class PolyExpansionTriton4(torch.autograd.Function):
    """Autograd function for 4th order polynomial expansion with Triton forward."""
    
    @staticmethod
    def forward(ctx, x, coeff):
        """Forward pass using Triton kernel."""
        batch, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        total_tokens = batch * seq_len
        out = torch.zeros_like(x_reshaped)
        
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_D = 256
        num_n_blocks = (total_tokens + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_d_blocks = (dim + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
        grid = (num_n_blocks, num_d_blocks)
        
        coeff_flat = coeff.flatten()
        po4_kernel[grid](x_reshaped, coeff_flat, out, total_tokens, dim,
                        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_D=BLOCK_SIZE_D)
        
        out = out.reshape(batch, seq_len, dim)
        
        # Save activations for backward
        h = torch.where(x >= 0, x, 0.01 * x)
        ctx.save_for_backward(x, coeff, h)
        ctx.k = 4
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computing gradients efficiently."""
        x, coeff, h = ctx.saved_tensors
        k = ctx.k
        
        batch, seq_len, dim = x.shape
        
        h2 = h * h
        h3 = h2 * h
        h4 = h2 * h2
        
        c0 = coeff[:, 0]
        c1 = coeff[:, 1]
        c2 = coeff[:, 2]
        c3 = coeff[:, 3]
        
        # dout/dh = c0 + 2*h*c1 + 3*h^2*c2 + 4*h^3*c3
        dout_dh = (c0[None, None, :] + 2 * h * c1[None, None, :] + 
                  3 * h2 * c2[None, None, :] + 4 * h3 * c3[None, None, :])
        
        # dh/dx (derivative of leaky ReLU)
        dh_dx = torch.where(x >= 0, torch.ones_like(x), 0.01 * torch.ones_like(x))
        
        grad_x = grad_output * dout_dh * dh_dx
        
        # Derivatives with respect to coeff
        grad_coeff = torch.zeros_like(coeff)
        grad_coeff[:, 0] = (grad_output * h).sum(dim=(0, 1))
        grad_coeff[:, 1] = (grad_output * h2).sum(dim=(0, 1))
        grad_coeff[:, 2] = (grad_output * h3).sum(dim=(0, 1))
        grad_coeff[:, 3] = (grad_output * h4).sum(dim=(0, 1))
        
        return grad_x, grad_coeff


# =============================================================================
# PyTorch Wrapper Functions
# =============================================================================

def po2_triton(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized 2nd order polynomial expansion with gradient support.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Coefficients of shape (dim, 2)
    
    Returns:
        Output tensor of shape (batch, seq_len, dim)
    """
    return PolyExpansionTriton2.apply(x, coeff)


def po3_triton(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized 3rd order polynomial expansion with gradient support.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Coefficients of shape (dim, 3)
    
    Returns:
        Output tensor of shape (batch, seq_len, dim)
    """
    return PolyExpansionTriton3.apply(x, coeff)


def po4_triton(x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized 4th order polynomial expansion with gradient support.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Coefficients of shape (dim, 4)
    
    Returns:
        Output tensor of shape (batch, seq_len, dim)
    """
    return PolyExpansionTriton4.apply(x, coeff)


def polynomial_aggregation_triton(x: torch.Tensor,
                                  coeff: torch.Tensor,
                                  k: int,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Triton-optimized polynomial aggregation with gradient support.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        coeff: Polynomial coefficients
        k: Polynomial order
        mask: Optional mask
    
    Returns:
        Aggregated output of shape (batch, 1, dim)
    """
    if mask is not None:
        from pom import polynomial_aggregation_
        return polynomial_aggregation_(x, coeff, k, mask)
    
    batch, seq_len, dim = x.shape
    
    if k == 2:
        h = po2_triton(x, coeff[:, :2])
    elif k == 3:
        h = po3_triton(x, coeff[:, :3])
    elif k == 4:
        h = po4_triton(x, coeff[:, :4])
    else:
        from pom import polynomial_aggregation_
        return polynomial_aggregation_(x, coeff, k, mask)
    
    # Mean pooling
    h = h.mean(dim=1, keepdim=True)
    out_dim = coeff.shape[0]
    h = h.expand(batch, 1, out_dim)
    
    return h


def polynomial_selection_triton(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Polynomial selection with sigmoid gating."""
    b, n, c = x.shape
    bb, nn, d = h.shape
    s = F.sigmoid(x).unsqueeze(-1)
    h = h.view(bb, nn, c, -1)
    return (s * h).view(b, n, d)


def pom_triton(xq: torch.Tensor,
               xc: torch.Tensor,
               coeff: torch.Tensor,
               k: int,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Triton-optimized Polynomial Mixer (PoM) operation."""
    h = polynomial_aggregation_triton(xc, coeff, k, mask)
    o = polynomial_selection_triton(xq, h)
    return o
