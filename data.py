import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Callable


class ImageNetDataset(Dataset):
    """Custom dataset for ImageNet in its original directory structure.
    
    Expects directory structure:
    - root/
      - class_1/
        - image1.jpg
        - image2.jpg
        - ...
      - class_2/
        - image1.jpg
        - ...
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_size: int = 224,
    ):
        """
        Args:
            root: Root directory containing 'train' and 'val' subdirectories
            split: Either 'train' or 'val'
            transform: Optional transform to apply to images
            target_size: Size to resize images to
        """
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        
        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Use ImageFolder for automatic class loading
        self.image_folder = ImageFolder(str(split_dir))
        
        # Setup default transform if none provided
        if transform is None:
            transform = self._get_default_transform()
        
        self.transform = transform
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transforms for ImageNet."""
        if self.split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_size + 32),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_folder)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            Tuple of (image_tensor, class_label)
        """
        image, label = self.image_folder[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images from a directory.
    
    Useful for custom image collections or simple test datasets.
    """
    
    def __init__(
        self,
        image_dir: str,
        target_size: int = 224,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    ):
        """
        Args:
            image_dir: Directory containing images
            target_size: Size to resize images to
            transform: Optional transform to apply
            extensions: Tuple of valid image extensions
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        
        # Collect all image paths
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(sorted(self.image_dir.glob(f'*{ext}')))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        # Setup transform
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            Image tensor
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
