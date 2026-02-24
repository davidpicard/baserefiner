import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
from math import sqrt
from copy import deepcopy


class EMAModel(nn.Module):
    """Exponential Moving Average (EMA) of model parameters.
    
    Maintains an EMA copy of a model's parameters for improved stability
    and generalization, commonly used in diffusion models.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to track EMA for
            decay: EMA decay coefficient. New value = decay * old + (1 - decay) * current
        """
        super().__init__()
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model: nn.Module):
        """Update EMA model with current model parameters.
        
        Args:
            model: Current model to update from
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1.0 - self.decay) * model_param.data
    
    def forward(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)
    
    def get_ema_model(self) -> nn.Module:
        """Get the underlying EMA model."""
        return self.ema_model


class FlowMatchingLoss(nn.Module):
    """Computes the flow matching loss."""
    
    def __init__(self, loss_type: str = "mse"):
        """
        Args:
            loss_type: Type of loss - "mse" for mean squared error or "l1" for L1 loss
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            v_pred: Predicted velocity field
            v_target: Target velocity field (ground truth)
            weights: Optional per-sample weights
        
        Returns:
            Scalar loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Average over spatial dimensions
        if mask is not None:
            loss = loss * mask
            loss = loss.sum(dim=(1,2,3))/mask.sum(dim=(1,2,3))
        else:
            loss = loss.mean(dim=list(range(1, loss.ndim)))
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


### BASE REFINER
class BaseRefinerFlowMatchingModule(pl.LightningModule):
    """PyTorch Lightning module for flow matching training with DiT."""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        num_training_steps: int = 100000,
        loss_type: str = "mse",
        flow_matching_type: str = "conditional",
        use_timestep_weighting: bool = False,
        random_refiner_token: bool = False,
        refiner_weight: float = 1.0,
        ema_decay: float = 0.9999,
        use_ema: bool = True
    ):
        """
        Args:
            model: DiT model instance to train
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            loss_type: Type of loss ("mse" or "l1")
            flow_matching_type: Type of flow matching ("conditional" or "unconditional")
            use_timestep_weighting: Whether to weight loss by timestep
            random_refiner_token: Whether to randomly mask refiner tokens
            refiner_weight: Weight for refiner loss
            ema_decay: EMA decay coefficient (higher = slower decay, default 0.9999)
            use_ema: Whether to use EMA model tracking
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = FlowMatchingLoss(loss_type=loss_type)
        self.random_refiner_token = random_refiner_token
        self.refiner_weight = refiner_weight
        self.use_ema = use_ema
        
        # Initialize EMA model if enabled
        if self.use_ema:
            self.ema = EMAModel(model, decay=ema_decay)
        else:
            self.ema = None
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, t, y)
    
    def _get_flow_matching_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the flow matching target velocity.
        
        For conditional flow matching, the velocity is: v_t = x1 - x0
        (constant velocity field from x0 to x1)
        
        Args:
            x0: Noise sample (batch, channels, height, width)
            x1: Data sample (batch, channels, height, width)
            t: Time steps (batch,) in range [0, 1]
        
        Returns:
            Velocity target (batch, channels, height, width)
        """
        # Constant velocity: v = x1 - x0
        velocity = x1 - x0
        return velocity
    
    def _get_flow_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the interpolated sample at time t.
        
        x_t = (1 - t) * x0 + t * x1
        
        Args:
            x0: Noise sample
            x1: Data sample
            t: Time steps (batch,)
        
        Returns:
            Interpolated sample x_t
        """
        # Ensure t is proper shape for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        x_t = (1 - t) * x0 + t * x1
        return x_t
    
    def _compute_timestep_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute optional timestep-dependent weighting.
        
        Args:
            t: Time steps (batch,)
        
        Returns:
            Weights (batch,)
        """
        if self.hparams.use_timestep_weighting:
            # Uniform weighting strategy
            weights = torch.ones_like(t)
        else:
            weights = torch.ones_like(t)
        
        return weights
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step: compute flow matching loss.
        
        Args:
            batch: Dictionary containing:
                - "x": Data samples (batch, channels, height, width)
                - "y" (optional): Class labels (batch,)
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        x_data, y_labels = batch
        batch_size = x_data.shape[0]
        device = x_data.device
        batch_size = x_data.shape[0]
        device = x_data.device
        
        # Sample random time steps
        # t = torch.rand(batch_size, device=device)
        # log normal
        t = torch.sigmoid(torch.normal(mean=0., 
                                       std=0.8, 
                                       size=(x_data.shape[0],),
                                       device=x_data.device, 
                                       dtype=x_data.dtype))
        
        # Sample noise (x0 ~ N(0, I))
        x_noise = torch.randn_like(x_data)
        
        # Compute interpolated sample x_t
        x_t = self._get_flow_xt(x_noise, x_data, t)
        
        # Get flow matching target (velocity)
        v_target = self._get_flow_matching_target(x_noise, x_data, t)

        # random refiner tokens?
        if self.random_refiner_token:
            b = x_t.size(0)
            p = self.model.num_patches
            refiner_mask = torch.ones(b, p).to(x_t.device)
            for row in refiner_mask:
                perm = torch.randperm(p)
                row[perm[:torch.randint(low=0, high=p-8, size=(1,))]] = 0
        else:
            refiner_mask = None

        # random y dropout
        y_labels[torch.rand(y_labels.size(0)) < 0.1] = self.model.num_classes
        
        # Predict velocity
        pred_base, pred_refiner = self.model(x_t, t, y_labels, refiner_mask)
        if self.model.prediction == "x":
            v_pred_base = self.model._get_vt_from_x0(pred_base, x_t, t)
            v_pred_refiner = self.model._get_vt_from_x0(pred_base.detach()+pred_refiner, x_t, t)
        else:
            v_pred_base = pred_base
            v_pred_refiner = v_pred_base.detach() + pred_refiner
                
        # Compute loss
        weights = self._compute_timestep_weight(t)
        base_loss = self.loss_fn(v_pred_base, v_target, weights)
        base_loss_masked = self.loss_fn(v_pred_base, v_target, weights, self.model._expand_mask_to_image(refiner_mask))
        refiner_loss = self.loss_fn(v_pred_refiner, v_target, weights, self.model._expand_mask_to_image(refiner_mask))
        loss = base_loss + self.refiner_weight * refiner_loss
        
        # Logging
        self.log("train/base_loss", base_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/base_loss_masked", base_loss_masked, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/refiner_loss", refiner_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update EMA model
        if self.use_ema and self.ema is not None:
            self.ema.update(self.model)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Validation step: compute validation loss.
        Uses EMA model for validation if available.
        
        Args:
            batch: Dictionary containing validation data
            batch_idx: Batch index
        """
        x_data, y_labels = batch
        batch_size = x_data.shape[0]
        device = x_data.device
        
        # Sample random time steps
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        x_noise = torch.randn_like(x_data)
        
        # Compute interpolated sample
        x_t = self._get_flow_xt(x_noise, x_data, t)
        
        # Get flow matching target
        v_target = self._get_flow_matching_target(x_noise, x_data, t)
        
        # Use EMA model for validation if available
        model_for_validation = self.ema.ema_model if (self.use_ema and self.ema is not None) else self.model
        
        # Predict velocity
        pred_base, pred_refiner = model_for_validation(x_t, t, y_labels)
        if self.model.prediction == "x":
            v_pred_base = self.model._get_vt_from_x0(pred_base, x_t, t)
            v_pred_refiner = self.model._get_vt_from_x0(pred_base.detach()+pred_refiner, x_t, t)
        else:
            v_pred_base = pred_base
            v_pred_refiner = v_pred_base.detach() + pred_refiner
                
        # Compute loss
        weights = self._compute_timestep_weight(t)
        base_loss = self.loss_fn(v_pred_base, v_target, weights)
        refiner_loss = self.loss_fn(v_pred_refiner, v_target, weights)
        loss = base_loss + refiner_loss
        
        # Logging
        self.log("val/base_loss", base_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/refiner_loss", refiner_loss,on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Filter parameters with weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "embed" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step: int) -> float:
            warmup_steps = self.hparams.warmup_steps
            num_training_steps = self.hparams.num_training_steps
            
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
