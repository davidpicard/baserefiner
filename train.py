import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import logging

from callbacks import WandbImageLoggingCallback


log = logging.getLogger(__name__)


def instantiate_lightning_module(model, config: DictConfig):
    """Instantiate Lightning module from config using Hydra."""
    log.info("Instantiating Flow Matching Lightning module...")
    
    # Copy lightning_module config and inject the model
    lm_config = config.lightning_module.copy()
    
    # Handle num_training_steps calculation
    if lm_config.num_training_steps is None:
        lm_config.num_training_steps = config.training.max_epochs * 1000
    
    lightning_module = instantiate(lm_config, model=model)
    
    return lightning_module


def setup_callbacks(model, config: DictConfig) -> list:
    """Setup Lightning callbacks."""
    callbacks = []
    
    # Checkpoint callback
    if config.training.get("save_checkpoint", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="dit-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=config.training.get("save_top_k", 3),
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if config.training.get("early_stopping", False):
        early_stop = EarlyStopping(
            monitor="val/loss",
            patience=config.training.get("early_stopping_patience", 10),
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop)
    
    # W&B image logging callback
    if config.logging.get("use_wandb", False) and config.sampling.get("log_images", False):
        image_callback = WandbImageLoggingCallback(
            model=model,
            config=config,
            num_samples=config.sampling.get("num_samples", 4),
            log_interval=config.sampling.get("log_interval", 500),
        )
        callbacks.append(image_callback)
    
    return callbacks


def setup_logger(config: DictConfig) -> list:
    """Setup Lightning loggers."""
    loggers = []
    
    # TensorBoard logger
    if config.logging.get("use_tensorboard", True):
        tb_logger = TensorBoardLogger(
            save_dir=config.logging.log_dir,
            name=config.logging.get("experiment_name", "dit-flow-matching"),
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if config.logging.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=config.logging.get("wandb_project", "dit-flow-matching"),
            name=config.logging.get("experiment_name", None),
            entity=config.logging.get("wandb_entity", None),
            tags=config.logging.get("tags", []),
        )
        loggers.append(wandb_logger)
    
    return loggers


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    log.info("=" * 80)
    log.info("Starting Flow Matching Training")
    log.info("=" * 80)
    log.info(OmegaConf.to_yaml(config))
    
    # Set random seed for reproducibility
    if config.get("seed", None) is not None:
        pl.seed_everything(config.seed)
    
    # Instantiate components using Hydra
    log.info("Instantiating model...")
    model = instantiate(config.model)
    log.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    log.info("Instantiating data module...")
    # Override target_size in data config with model input_size
    data_config = config.data.copy()
    data_config.target_size = config.model.input_size
    data_module = instantiate(data_config)
    
    log.info("Instantiating Lightning module...")
    lightning_module = instantiate_lightning_module(model, config)
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(model, config)
    loggers = setup_logger(config)
    
    # Create trainer
    log.info("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.get("accelerator", "auto"),
        devices=config.training.get("devices", "auto"),
        strategy=config.training.get("strategy", "auto"),
        precision=config.training.get("precision", "32-true"),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.logging.get("log_every_n_steps", 100),
        val_check_interval=config.training.get("val_check_interval", 1.0),
        gradient_clip_val=config.training.get("gradient_clip_val", None),
        gradient_clip_algorithm=config.training.get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=config.training.get("accumulate_grad_batches", 1),
        enable_progress_bar=config.training.get("enable_progress_bar", True),
        deterministic=config.get("deterministic", False),
    )
    
    # Train
    log.info("Starting training...")
    trainer.fit(lightning_module, datamodule=data_module)
    
    log.info("=" * 80)
    log.info("Training completed!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
