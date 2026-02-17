import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import ImageNetDataset, SimpleImageDataset
from typing import Optional, Callable


class ImageNetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ImageNet.
    
    Handles loading, preprocessing, and batching of ImageNet data.
    Expects directory structure:
    - data_root/
      - train/
        - class_1/
        - class_2/
        - ...
      - val/
        - class_1/
        - class_2/
        - ...
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_size: int = 224,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root: Root directory containing 'train' and 'val' subdirectories
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            target_size: Target image size
            train_transform: Optional custom training transforms
            val_transform: Optional custom validation transforms
        """
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_size = target_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or None
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ImageNetDataset(
                root=self.data_root,
                split="train",
                transform=self.train_transform,
                target_size=self.target_size,
            )
            print(f"data: {len(self.train_dataset)} images in train")
            
            self.val_dataset = ImageNetDataset(
                root=self.data_root,
                split="val",
                transform=self.val_transform,
                target_size=self.target_size,
            )
            print(f"data: {len(self.val_dataset)} images in val")
        
        elif stage == "validate":
            self.val_dataset = ImageNetDataset(
                root=self.data_root,
                split="val",
                transform=self.val_transform,
                target_size=self.target_size,
            )
            print(f"data: {len(self.val_dataset)} images in val")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            self.setup(stage="fit")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage="validate")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class SimpleImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for simple image collections.
    
    Useful for custom datasets where you just need to load all images
    from a directory without class structure.
    """
    
    def __init__(
        self,
        image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        target_size: int = 224,
        train_split: float = 0.8,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_dir: Directory containing images
            batch_size: Batch size
            num_workers: Number of workers
            pin_memory: Whether to pin memory
            target_size: Target image size
            train_split: Fraction of data to use for training (between 0 and 1)
            transform: Optional custom transforms
        """
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_size = target_size
        self.train_split = train_split
        self.transform = transform
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val split from single directory."""
        if stage == "fit" or stage is None:
            full_dataset = SimpleImageDataset(
                image_dir=self.image_dir,
                target_size=self.target_size,
                transform=self.transform,
            )
            
            # Train/val split
            train_size = int(len(full_dataset) * self.train_split)
            val_size = len(full_dataset) - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            self.setup(stage="fit")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage="fit")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
