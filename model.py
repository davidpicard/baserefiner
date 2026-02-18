import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization.
    
    This layer normalizes its input and then applies an affine transformation
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
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
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
        # Normalize the input
        x_norm = self.ln(x)
        
        # Get scale and shift from embedding
        scale_shift = self.linear(emb)
        scale, shift = rearrange(scale_shift, 'b (n d) -> n b d', n=2)
        
        # Apply scale and shift
        # Add 1 to scale so default is identity
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        
        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            emb: Conditioning embedding of shape (batch, emb_dim)
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Self-attention with AdaLNZero
        x_norm = self.ln1(x, emb)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with AdaLNZero
        x_norm = self.ln2(x, emb)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


#### DiT model

class DiTModule(nn.Module):
        
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        emb_dim: int = 512,
        num_classes: int = None,
        learn_sigma: bool = True,
    ):
        """
        Args:
            input_size: Input image size (assumes square images)
            patch_size: Size of patches for tokenization
            in_channels: Number of input channels
            hidden_dim: Dimension of hidden features
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dimension to hidden_dim
            emb_dim: Dimension of time/condition embedding
            num_classes: Number of classes for conditional generation (None = unconditional)
            learn_sigma: Whether to predict sigma in addition to mean
        """
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
        
        # Calculate number of patches
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding: project patches to hidden_dim
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # Learnable class token (for conditioning if needed)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional embedding for patches + cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Class embedding network (if conditional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, emb_dim)
        
        # Combined embedding projection
        self.embed_proj = nn.Linear(emb_dim, emb_dim)
        
        # Stack of DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, emb_dim)
            for _ in range(depth)
        ])
        
        # Final layer norm and output projection
        self.final_ln = AdaLNZero(hidden_dim, emb_dim)
        
        # Output projection to image space
        out_channels = 2 * in_channels if learn_sigma else in_channels
        self.out_proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        
        # Initialize weights
        self._init_weights()
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize output projection
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
    
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
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None
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
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)  # (batch, num_patches+1, hidden_dim)
        
        # Add positional embeddings
        x_emb = x_emb + self.pos_embed  # (batch, num_patches+1, hidden_dim)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, emb_dim)
        
        # Optionally add class embedding
        if y is not None and self.class_embed is not None:
            y_emb = self.class_embed(y)  # (batch, emb_dim)
            t_emb = t_emb + y_emb
        
        # Project combined embedding
        cond_emb = self.embed_proj(t_emb)  # (batch, emb_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x_emb = block(x_emb, cond_emb)
        
        # Final layer norm
        x_emb = self.final_ln(x_emb, cond_emb)
        
        # Remove class token and project to output
        x_out = x_emb[:, 1:, :]  # (batch, num_patches, hidden_dim)
        x_out = self.out_proj(x_out)  # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # predict v from x_pred
        x_out = (x_out - x_patch)/(1-t).clamp(min=0.05)

        # Unpatchify
        out = self._unpatchify(x_out)  # (batch, out_channels, height, width)
        
        return out


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
        
        # Calculate number of patches
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding: project patches to hidden_dim
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # Learnable class token (for conditioning if needed)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional embedding for patches + cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Class embedding network (if conditional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, emb_dim)
        
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
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
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None
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
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)  # (batch, num_patches+1, hidden_dim)
        
        # Add positional embeddings
        x_emb = x_emb + self.pos_embed  # (batch, num_patches+1, hidden_dim)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, emb_dim)
        
        # Optionally add class embedding
        if y is not None and self.class_embed is not None:
            y_emb = self.class_embed(y)  # (batch, emb_dim)
            t_emb = t_emb + y_emb
        
        # Project combined embedding
        cond_emb = self.embed_proj(t_emb)  # (batch, emb_dim)
        
        #### BASE
        # Apply transformer blocks
        for block in self.base_blocks:
            x_emb = block(x_emb, cond_emb)
        
        # Final layer norm
        x_emb_base = self.base_final_ln(x_emb, cond_emb)
        
        # Remove class token and project to output
        x_out = x_emb_base[:, 1:, :]  # (batch, num_patches, hidden_dim)
        x_out = self.base_out_proj(x_out)  # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # predict v from x_pred
        # x_out = (x_out - x_patch)/(1-t).clamp(min=0.05)

        # Unpatchify
        out_base = self._unpatchify(x_out)  # (batch, out_channels, height, width)

        #### REFINER
        x_emb = x_emb.detach()
        for block in self.refiner_blocks:
            x_emb = block(x_emb, cond_emb)

        # Final layer norm
        x_emb = self.refiner_final_ln(x_emb, cond_emb)
        
        # Remove class token and project to output
        x_out = x_emb[:, 1:, :]  # (batch, num_patches, hidden_dim)
        x_out = self.refiner_out_proj(x_out)  # (batch, num_patches, out_channels*patch_size*patch_size)
        
        # predict v from x_pred
        # x_out = (x_out - x_patch)/(1-t).clamp(min=0.05)

        # Unpatchify
        out_refiner = self._unpatchify(x_out)  # (batch, out_channels, height, width)
        return out_base, out_refiner

