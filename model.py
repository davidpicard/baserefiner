import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
from pom import PoMMixer


class TokenRouter(nn.Module):
    """Token Routing for Efficient Architecture-agnostic Diffusion Training (TREAD).
    
    Implementation of token routing mechanism from:
    https://arxiv.org/abs/2501.04765
    
    Routing allows selected tokens to skip computation in intermediate layers,
    reducing computational cost while improving training effectiveness. Unlike
    masking-based approaches, information is preserved and reintroduced at a
    later layer.
    
    IMPORTANT: When tokens are routed, special care must be taken with RoPE2D
    (2D rotary position embeddings). Each layer processes a different subset
    of tokens, so RoPE must be applied correctly to maintain proper position
    encoding even with missing tokens.
    """
    
    def __init__(self, start_layer: int, end_layer: int, drop_percent: int, 
                 total_layers: int, seed: int = None):
        """
        Args:
            start_layer: Layer index where routing begins (tokens taken from here)
            end_layer: Layer index where routed tokens rejoin (tokens reintroduced here)
            drop_percent: Percentage of tokens to route (0-100)
                         Equivalent to (1 - selection_rate) * 100
            total_layers: Total number of layers in model (for validation)
            seed: Random seed for reproducibility of token selection
        """
        super().__init__()
        
        assert 0 <= start_layer < end_layer <= total_layers, \
            f"Invalid routing range: start={start_layer}, end={end_layer}, total={total_layers}"
        assert 0 <= drop_percent <= 100, f"drop_percent must be in [0, 100], got {drop_percent}"
        
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.drop_percent = drop_percent
        self.total_layers = total_layers
        self.selection_rate = (100 - drop_percent) / 100.0
        self.seed = seed
        
    def get_routing_mask(self, batch_size: int, num_tokens: int, device: torch.device,
                        dtype: torch.dtype) -> tuple:
        """Generate routing mask for token selection.
        
        Args:
            batch_size: Batch size
            num_tokens: Number of tokens per sample (excluding class/register tokens)
            device: Device to create tensors on
            dtype: Data type for output
        
        Returns:
            Tuple of (routed_mask, direct_mask, route_indices)
            - routed_mask: Boolean mask of shape (batch, num_tokens), True where tokens are routed
            - direct_mask: Boolean mask of shape (batch, num_tokens), True where tokens pass through
            - route_indices: Selected indices for routed tokens
        """
        
        # Random selection: select a percentage of tokens to route
        num_routed = max(1, int(num_tokens * (1 - self.selection_rate)))
        
        # Generate route indices (same for all samples in batch for simplicity)
        all_indices = torch.arange(num_tokens, device=device)
        perm = torch.randperm(num_tokens, device=device)
        route_indices = perm[:num_routed]  # (num_routed,)
        
        # Create masks
        routed_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
        routed_mask[:, route_indices] = True
        
        direct_mask = ~routed_mask
        
        return routed_mask, direct_mask, route_indices
    
    def route_tokens_from_start(self, x: torch.Tensor, patch_shape: tuple,
                               num_cls_tokens: int) -> dict:
        """Extract tokens to be routed at start layer.
        
        IMPORTANT: Preserves cls/register tokens and spatial structure information
        for proper RoPE2D handling in later layers. Computes original 2D grid positions
        so RoPE2D can be applied correctly even when tokens are missing.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
                Assumed to have num_cls_tokens at the beginning
            patch_shape: Tuple (H, W) of patch grid dimensions
            num_cls_tokens: Number of class/register tokens at beginning of sequence
        
        Returns:
            Dictionary containing:
            - 'x_direct': Tokens that pass directly through routing layers
            - 'x_routed': Tokens that are routed
            - 'routed_mask': Boolean mask indicating routed positions
            - 'direct_mask': Boolean mask indicating direct positions
            - 'route_indices': Indices of routed tokens (relative to patches)
            - 'direct_indices': Indices of direct tokens (relative to patches)
            - 'direct_positions_2d': 2D grid positions (h, w) of direct tokens, shape (num_direct, 2)
            - 'patch_shape': Original patch shape for RoPE
            - 'num_cls_tokens': Number of class/register tokens
        """
        batch_size, seq_len, dim = x.shape
        
        # Separate class tokens from patches
        x_cls = x[:, :num_cls_tokens, :]  # (batch, num_cls_tokens, dim)
        x_patches = x[:, num_cls_tokens:, :]  # (batch, num_patches, dim)
        
        num_patches = x_patches.shape[1]
        height, width = patch_shape
        
        # Get routing mask
        routed_mask, direct_mask, route_indices = self.get_routing_mask(
            batch_size, num_patches, x.device, x.dtype
        )
        
        # Extract direct token indices (same for all batch samples)
        direct_indices = torch.where(direct_mask[0])[0]  # (num_direct,)
        
        # Compute 2D grid positions of direct tokens
        # Convert flat indices to 2D coordinates (h, w)
        direct_positions_2d = torch.stack([
            direct_indices // width,  # height coordinate
            direct_indices % width    # width coordinate
        ], dim=1).float()  # (num_direct, 2)
        
        # Extract routed and direct tokens
        x_routed = x_patches[:, route_indices, :]  # (batch, num_routed, dim)
        x_direct = x_patches[:, direct_indices, :]  # (batch, num_direct, dim)
        
        # Reconstruct sequences with class tokens at front
        x_direct_full = torch.cat([x_cls, x_direct], dim=1)  # (batch, num_cls+num_direct, dim)
        
        return {
            'x_direct': x_direct_full,
            'x_routed': x_routed,
            'routed_mask': routed_mask,
            'direct_mask': direct_mask,
            'route_indices': route_indices,
            'direct_indices': direct_indices,
            'direct_positions_2d': direct_positions_2d,
            'patch_shape': patch_shape,
            'num_cls_tokens': num_cls_tokens,
            'num_patches': num_patches,
        }
    
    def reintroduce_tokens_at_end(self, x_direct: torch.Tensor, x_routed: torch.Tensor,
                                  route_state: dict) -> torch.Tensor:
        """Reintroduce routed tokens at end layer.
        
        Carefully reconstructs the full sequence with routed tokens reintroduced
        in their original positions, maintaining proper alignment for downstream
        processing.
        
        Args:
            x_direct: Processed direct tokens (batch, num_cls+num_direct, dim)
            x_routed: Routed tokens (batch, num_routed, dim)
            route_state: Dictionary from route_tokens_from_start()
        
        Returns:
            Full tensor with tokens reintroduced (batch, seq_len, dim)
        """
        batch_size = x_direct.shape[0]
        num_cls_tokens = route_state['num_cls_tokens']
        
        # Separate cls tokens from processed patches
        x_cls = x_direct[:, :num_cls_tokens, :]
        x_direct_patches = x_direct[:, num_cls_tokens:, :]
        
        # Reconstruct full sequence
        num_patches = route_state['num_patches']
        x_recon = torch.zeros(batch_size, num_patches, x_direct.shape[-1],
                             device=x_direct.device, dtype=x_direct.dtype)
        
        # Place direct tokens
        direct_indices = torch.where(route_state['direct_mask'][0])[0]
        x_recon[:, direct_indices, :] = x_direct_patches
        
        # Place routed tokens
        route_indices = route_state['route_indices']
        x_recon[:, route_indices, :] = x_routed
        
        # Prepend class tokens
        x_full = torch.cat([x_cls, x_recon], dim=1)
        
        return x_full


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
                patch_shape: tuple = None, num_cls_tokens: int = 0, 
                token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            emb: Conditioning embedding of shape (batch, emb_dim)
            mask: Optional mask tensor of shape (batch, seq_len) with 1s for valid positions, 0s for masked
            patch_shape: Optional tuple (height, width) of patch grid for 2D RoPE
            num_cls_tokens: Number of class/context tokens (not spatially encoded)
            token_positions: Optional (num_spatial_tokens, 2) tensor with 2D grid positions of tokens.
                           Used during TREAD routing when tokens are sparse.
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Generate modulation parameters from conditioning
        # Output: [shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp]
        # print(f"x: {x.shape} pos: {token_positions}")
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
                            patch_shape=patch_shape, num_cls_tokens=num_cls_tokens,
                            token_positions=token_positions)
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



### Baseline model

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
        pom_expand: int = 2,
        use_tread: bool = False,
        tread_start_layer: int = 2,
        tread_end_layer: int = 8,
        tread_drop_percent: int = 50
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
        
        # TREAD Token Routing initialization
        self.use_tread = use_tread
        self.tread_start_layer = tread_start_layer
        self.tread_end_layer = tread_end_layer
        self.tread_drop_percent = tread_drop_percent
        
        if use_tread:
            self.token_router = TokenRouter(
                start_layer=tread_start_layer,
                end_layer=tread_end_layer,
                drop_percent=tread_drop_percent,
                total_layers=depth
            )
        else:
            self.token_router = None
        
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
        if y is None and self.num_classes is not None:
            y = torch.ones(1, dtype=torch.long, device=x_emb.device) * self.num_classes
        if hasattr(self, 'class_embed') and y is not None:
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

        #### BASE with TREAD Token Routing
        # Initialize routing state
        route_state = None
        x_routed = None
        
        # Apply transformer blocks with in-context class token insertion
        for block_idx, block in enumerate(self.base_blocks):
            # TREAD: Route tokens at start layer
            if self.use_tread and self.training and block_idx == self.tread_start_layer:
                route_state = self.token_router.route_tokens_from_start(
                    x_emb, patch_shape, num_cls_tokens=self.n_register
                )
                x_routed = route_state['x_routed']
                x_emb = route_state['x_direct']
                # For routed layers, we need to pass adjusted patch shape info
                # Store the direct-only patch indices for RoPE2D
                route_state['direct_patch_indices'] = torch.where(route_state['direct_mask'][0])[0]
            
            # Insert in-context class tokens at intermediate block (Paper Appendix A)
            if block_idx == self.in_context_start_block:
                class_tokens = self.in_context_class_tokens.expand(batch_size, -1, -1)
                x_emb = torch.cat([x_emb, class_tokens], dim=1)
            
            # TREAD: For routed layers, use adjusted patch_shape for RoPE2D
            # This is critical to avoid positional encoding issues
            if self.use_tread and self.training and self.tread_start_layer < block_idx < self.tread_end_layer and route_state is not None:
                # During routed layers, pass the route_state so RoPE2D can be applied correctly
                # Store route_state in block for RoPE2D handling
                x_emb = self._process_block_with_routing(
                    block, x_emb, cond_emb, patch_shape, 
                    route_state, num_cls_tokens=self.n_register
                )
            else:
                # Normal block processing without routing
                x_emb = block(x_emb, cond_emb, patch_shape=patch_shape, num_cls_tokens=self.n_register)
            
            # TREAD: Reintroduce routed tokens at end layer
            if self.use_tread and self.training and block_idx == self.tread_end_layer - 1 and route_state is not None:
                # Remove in-context tokens temporarily if they exist
                num_class_tokens = self.in_context_start_block < self.depth and self.num_in_context_tokens or 0
                if num_class_tokens > 0 and block_idx >= self.in_context_start_block:
                    x_emb_without_class = x_emb[:, :-num_class_tokens, :]
                    x_recon = self.token_router.reintroduce_tokens_at_end(x_emb_without_class, x_routed, route_state)
                    x_emb = torch.cat([x_recon, x_emb[:, -num_class_tokens:, :]], dim=1)
                else:
                    x_emb = self.token_router.reintroduce_tokens_at_end(x_emb, x_routed, route_state)
                route_state = None
                x_routed = None
        
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
    
    def _process_block_with_routing(self, block: nn.Module, x: torch.Tensor, 
                                   cond_emb: torch.Tensor, patch_shape: tuple,
                                   route_state: dict, num_cls_tokens: int) -> torch.Tensor:
        """Process a block with TREAD routing, handling RoPE2D correctly.
        
        This function passes the original 2D grid positions of tokens to the block
        so that RoPE2D can be applied based on the true spatial positions,
        not the sequential order of the remaining tokens.
        
        Args:
            block: The transformer block to process
            x: Input tensor with direct tokens only
            cond_emb: Conditioning embedding
            patch_shape: Original patch grid shape
            route_state: Routing state from token_router containing direct_positions_2d
            num_cls_tokens: Number of class/register tokens at sequence start
        
        Returns:
            Processed tensor
        """
        # Pass token positions so RoPE2D applies correct positional encoding
        # based on original 2D grid positions, not sequential order
        token_positions = route_state['direct_positions_2d']
        return block(x, cond_emb, patch_shape=patch_shape, num_cls_tokens=num_cls_tokens,
                    token_positions=token_positions)

