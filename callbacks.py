import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from torchvision.utils import make_grid
import numpy as np
import logging

from sampler import HeunSampler

log = logging.getLogger(__name__)


class WandbImageLoggingCallback(Callback):
    """Callback to log generated images to W&B during training."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        num_samples: int = 4,
        log_interval: int = 100,
    ):
        """
        Args:
            model: The DiT model for sampling
            config: Training config containing sampling parameters
            num_samples: Number of samples to generate
            log_interval: Log every n steps
        """
        super().__init__()
        self.model = model
        self.config = config
        self.num_samples = num_samples
        self.log_interval = log_interval
        
        # Device will be set on first training batch
        self.device = None
        self.sampler = None
        self.seed_noise = None
        self.seed_classes = None
    
    def _setup_sampler_and_seeds(self, device):
        """Create sampler and seeded tensors on the correct device."""
        if self.device is not None:
            return  # Already initialized
        
        self.device = device
        
        # Setup sampler
        self.sampler = HeunSampler(
            model=self.model,
            num_steps=self.config.sampling.get("num_steps", 50),
            device=device,
            use_refiner=self.config.sampling.get("use_refiner", False)
        )
        
        # Create deterministic noise and classes for consistent progression
        self._setup_seed_tensors()
    
    def _setup_seed_tensors(self):
        """Create seeded noise and class labels for consistent sampling."""
        seed = self.config.get("seed", 3407)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create fixed noise tensor on the correct device
        self.seed_noise = torch.randn(
            self.num_samples,
            self.model.in_channels,
            self.model.input_size,
            self.model.input_size,
            device=self.device,
        )
        
        # Create fixed class labels if using conditional generation
        if self.model.num_classes is not None:
            self.seed_classes = torch.randint(
                0,
                self.model.num_classes,
                (self.num_samples,),
                device=self.device
            )
        else:
            self.seed_classes = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log images at specified intervals."""
        # Initialize sampler on first batch (ensures correct device)
        if self.sampler is None:
            self._setup_sampler_and_seeds(pl_module.device)
        
        if (trainer.global_step + 1) % self.log_interval == 0:
            self._log_images(trainer, pl_module)
    
    def _log_images(self, trainer, pl_module):
        """Generate and log images to W&B."""
        if not trainer.logger:
            return
        
        # Check if using W&B logger
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        
        if wandb_logger is None:
            return
        
        try:
            import wandb
            
            log_dict = {"step": trainer.global_step}
            
            # Check if we should log both base and refined
            use_refiner = self.config.sampling.get("use_refiner", False)
            
            # Always log base predictions
            with torch.no_grad():
                samples_base = self.sampler.sample(
                    batch_size=self.num_samples,
                    y=self.seed_classes if self.seed_classes is not None else None,
                    noise=self.seed_noise,
                    use_refiner=False
                )
            
            # Denormalize and log base
            samples_base = self._denormalize_and_grid(samples_base)
            log_dict["base_images"] = wandb.Image(
                samples_base,
                caption=f"Base - classes {self.seed_classes}"
            )
            
            # If refiner is enabled, also log refined predictions
            if use_refiner:
                with torch.no_grad():
                    samples_refined = self.sampler.sample(
                        batch_size=self.num_samples,
                        y=self.seed_classes if self.seed_classes is not None else None,
                        noise=self.seed_noise,
                        use_refiner=True
                    )
                
                # Denormalize and log refined
                samples_refined = self._denormalize_and_grid(samples_refined)
                log_dict["refined_images"] = wandb.Image(
                    samples_refined,
                    caption=f"Base + Refiner - classes {self.seed_classes}"
                )
            
            # Log to W&B
            wandb_logger.experiment.log(log_dict)
            
        except Exception as e:
            log.warning(f"Failed to log images to W&B: {e}")
    
    def _denormalize_and_grid(self, samples: torch.Tensor) -> torch.Tensor:
        """Denormalize images and create grid for logging.
        
        Args:
            samples: Tensor of shape (batch, channels, height, width)
        
        Returns:
            Grid image as tensor
        """
        # Denormalize images (inverse of ImageNet normalization)
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=samples.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=samples.device
        ).view(1, 3, 1, 1)
        samples = samples * std + mean
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        grid = make_grid(samples, nrow=4, normalize=False)
        
        # Move to CPU for logging
        return grid.cpu()
