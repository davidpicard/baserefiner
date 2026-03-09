"""
Evaluation script: Load checkpoint, generate images, and compute FID against ImageNet.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import scipy.linalg
import os
from PIL import Image

from model import BaseRefiner
from sampler import HeunSampler
from pytorch_lightning import LightningModule

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_inception_model(device: str = "cuda"):
    """Load InceptionV3 model for FID computation."""
    from torchvision.models import inception_v3
    
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    inception.to(device)
    
    # Remove the classifier layer to get features
    inception.fc = nn.Identity()
    
    return inception


def compute_inception_features(images: torch.Tensor, inception_model: nn.Module, device: str = "cuda") -> np.ndarray:
    """
    Compute Inception features for a batch of images.
    
    Args:
        images: Tensor of shape (batch, 3, height, width) in range [0, 1]
        inception_model: InceptionV3 model
        device: Device to run on
    
    Returns:
        Features of shape (batch, feature_dim)
    """
    images = images.to(device)
    
    # Normalize for InceptionV3 (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    
    # Resize to 299x299 if needed
    if images.shape[-1] != 299:
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        features = inception_model(images)
    
    return features.cpu().numpy()


def compute_statistics_from_features(features: np.ndarray) -> tuple:
    """
    Compute mean and covariance from features.
    
    Args:
        features: Features of shape (num_images, feature_dim)
    
    Returns:
        Tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features.T)
    return mu, sigma


def compute_fid_from_stats(mu_real: np.ndarray, sigma_real: np.ndarray, 
                           mu_fake: np.ndarray, sigma_fake: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID) from pre-computed statistics.
    
    Args:
        mu_real: Mean of real features
        sigma_real: Covariance of real features
        mu_fake: Mean of generated features
        sigma_fake: Covariance of generated features
    
    Returns:
        FID score
    """
    # Compute FID
    diff = mu_real - mu_fake
    
    # Compute square root of product of covariances
    sqrt_product = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_fake))
    
    # Handle numerical issues
    if np.iscomplexobj(sqrt_product):
        sqrt_product = np.real(sqrt_product)
    
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrt_product)
    return fid


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        real_features: Features from real images (num_real, feature_dim)
        fake_features: Features from generated images (num_fake, feature_dim)
    
    Returns:
        FID score
    """
    # Compute mean and covariance
    mu_real, sigma_real = compute_statistics_from_features(real_features)
    mu_fake, sigma_fake = compute_statistics_from_features(fake_features)
    
    return compute_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model = BaseRefiner()
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def generate_images(
    model: nn.Module,
    num_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
    use_refiner: bool = False,
    guidance_scale: float = 1.0,
    num_classes: int = 1000
) -> torch.Tensor:
    """
    Generate images using the model.
    
    Args:
        model: DiT or BaseRefiner model
        num_samples: Number of images to generate
        batch_size: Batch size for generation
        device: Device to run on
        use_refiner: Use refiner if model supports it
        guidance_scale: Classifier-free guidance scale
        num_classes: Number of classes for class labels
    
    Returns:
        Generated images of shape (num_samples, 3, h, w) in range [0, 1]
    """
    sampler = HeunSampler(
        model=model,
        num_steps=50,
        device=device,
        use_refiner=use_refiner
    )
    
    all_images = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating images"):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Random class labels for conditional generation
            if hasattr(model, 'num_classes') and model.num_classes is not None:
                y = torch.randint(0, num_classes, (current_batch_size,), device=device)
            else:
                y = None
            
            # Generate samples
            samples = sampler.sample(
                batch_size=current_batch_size,
                y=y,
                guidance_scale=guidance_scale
            )
            
            # Normalize to [0, 1]
            samples = (samples + 1) / 2  # Assuming model outputs in [-1, 1]
            samples = torch.clamp(samples, 0, 1)
            
            all_images.append(samples.cpu())
    
    return torch.cat(all_images, dim=0)


def load_reference_statistics(ref_path: str, device: str = "cuda", num_samples: int = 1000) -> tuple:
    """
    Load reference statistics (mean and covariance) from .npz file or compute from image directory online.
    
    Args:
        ref_path: Path to .npz file with statistics or image directory
        device: Device to run on
        num_samples: Number of images to process (if computing from directory)
    
    Returns:
        Tuple of (mean, covariance) or (None, None) if not found
    """
    ref_path = Path(ref_path)
    
    # If npz file exists, load it
    if ref_path.suffix == '.npz' and ref_path.exists():
        log.info(f"Loading reference statistics from {ref_path}")
        data = np.load(ref_path)
        if 'mean' in data and 'cov' in data:
            return data['mean'], data['cov']
        else:
            log.warning(f"Expected 'mean' and 'cov' in {ref_path}, found keys: {list(data.keys())}")
            return None, None
    
    # Otherwise, compute from image directory using online computation
    if ref_path.is_dir():
        return compute_statistics_online(str(ref_path), device, num_samples)
    
    raise ValueError(f"ref_path must be either .npz file or image directory, got {ref_path}")


def compute_features_from_directory(image_dir: str, device: str = "cuda", num_samples: int = 1000) -> np.ndarray:
    """
    Compute inception features from all images in a directory.
    
    Args:
        image_dir: Path to directory containing images
        device: Device to run on
        num_samples: Maximum number of images to load
    
    Returns:
        Features of shape (num_images, feature_dim)
    """
    inception_model = load_inception_model(device)
    
    # Get list of image files
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in image_dir.rglob('*')
        if f.suffix.lower() in image_extensions
    ])[:num_samples]
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    log.info(f"Found {len(image_files)} images in {image_dir}")
    
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    
    for img_path in tqdm(image_files, desc="Computing reference features"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            features = compute_inception_features(img_tensor, inception_model, device)
            all_features.append(features)
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
            continue
    
    return np.concatenate(all_features, axis=0) if all_features else None


def compute_statistics_online(image_dir: str, device: str = "cuda", num_samples: int = 1000) -> tuple:
    """
    Compute inception statistics from images in a directory using online computation.
    This method avoids storing all features in memory and can handle millions of images.
    
    Uses the formula: Cov = E[XX^T] - E[X]E[X]^T
    
    Args:
        image_dir: Path to directory containing images
        device: Device to run on
        num_samples: Maximum number of images to process
    
    Returns:
        Tuple of (mean, covariance) or (None, None) if failed
    """
    inception_model = load_inception_model(device)
    
    # Get list of image files
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in image_dir.rglob('*')
        if f.suffix.lower() in image_extensions
    ])[:num_samples]
    
    if not image_files:
        log.error(f"No images found in {image_dir}")
        return None, None
    
    log.info(f"Found {len(image_files)} images. Computing statistics online (memory-efficient)...")
    
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Online statistics computation
    n = 0
    sum_x = None
    sum_xx = None  # Will accumulate X^T @ X for covariance
    
    for img_path in tqdm(image_files, desc="Computing statistics"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            features = compute_inception_features(img_tensor, inception_model, device)  # shape (B, feat_dim)
            features = features.astype(np.float64)  # Use float64 for numerical stability
            
            # Update online statistics
            batch_size = features.shape[0]
            n += batch_size
            
            if sum_x is None:
                sum_x = np.sum(features, axis=0)
                sum_xx = features.T @ features  # (D, B) @ (B, D) = (D, D)
            else:
                sum_x += np.sum(features, axis=0)
                sum_xx += features.T @ features
                
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
            continue
    
    if n == 0:
        log.error("No valid images processed")
        return None, None
    
    log.info(f"Processed {n} images")
    
    # Compute mean and covariance from accumulated statistics
    mean = sum_x / n
    # Covariance: E[XX^T] - E[X]E[X]^T
    cov = (sum_xx / n) - np.outer(mean, mean)
    
    return mean, cov


def compute_and_save_reference_statistics(image_dir: str, output_path: str, device: str = "cuda", num_samples: int = 1000) -> bool:
    """
    Compute statistics over all images in a directory and save to .npz file.
    Uses online computation to handle millions of images without storing all features.
    
    Args:
        image_dir: Path to directory containing images
        output_path: Path to save .npz file with statistics
        device: Device to run on
        num_samples: Maximum number of images to process
    
    Returns:
        True if successful, False otherwise
    """
    log.info(f"Computing statistics over all images in {image_dir}")
    
    # Use online computation for memory efficiency
    mu, sigma = compute_statistics_online(image_dir, device, num_samples)
    
    if mu is None or sigma is None:
        log.error("Failed to compute statistics")
        return False
    
    # Save to npz
    np.savez_compressed(output_path, mean=mu, cov=sigma)
    log.info(f"Saved statistics to {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiT model or compute reference features")
    parser.add_argument("--mode", type=str, choices=['calc', 'ref'], default='calc',
                       help="Mode: 'calc' to compute FID, 'ref' to compute and store reference features")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file (required for 'calc' mode)")
    parser.add_argument("--ref-path", type=str, required=True,
                       help="For 'calc' mode: path to .npz file or image directory with reference features. "
                            "For 'ref' mode: path to image directory to compute features from")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of images to generate/load")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for generation")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--use-refiner", action="store_true",
                       help="Use refiner model if available")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Save results/features to file")
    
    args = parser.parse_args()
    device = args.device
    
    # Mode: compute and store reference statistics
    if args.mode == 'ref':
        ref_path = Path(args.ref_path)
        
        if not ref_path.is_dir():
            raise ValueError(f"For 'ref' mode, ref_path must be a directory, got {ref_path}")
        
        output_path = args.output_file or 'reference_statistics.npz'
        
        success = compute_and_save_reference_statistics(
            str(ref_path), output_path, device, args.num_samples
        )
        
        if success:
            print(f"\nReference statistics computed and saved to {output_path}")
            print(f"Mode: 'ref' complete - statistics saved efficiently")
        return
    
    # Mode: calculate FID
    if args.mode == 'calc':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for 'calc' mode")
        
        log.info(f"Loading checkpoint from {args.checkpoint}")
        model = load_checkpoint(args.checkpoint, device=device)
        
        log.info(f"Generating {args.num_samples} images...")
        generated_images = generate_images(
            model=model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
            use_refiner=args.use_refiner,
            guidance_scale=args.guidance_scale,
            num_classes=getattr(model, 'num_classes', None) or 1000
        )
        
        # Compute inception features for generated images
        log.info("Computing inception features for generated images...")
        inception_model = load_inception_model(device)
        fake_features = []
        
        for i in tqdm(range(0, len(generated_images), args.batch_size), desc="Computing generated features"):
            batch = generated_images[i:i+args.batch_size]
            features = compute_inception_features(batch, inception_model, device)
            fake_features.append(features)
        
        fake_features = np.concatenate(fake_features, axis=0)
        
        # Load or compute reference statistics
        log.info("Loading reference statistics...")
        mu_real, sigma_real = load_reference_statistics(args.ref_path, device, args.num_samples)
        
        if mu_real is None or sigma_real is None:
            log.warning("Skipping FID computation without reference statistics")
            return
        
        # Compute statistics for generated images
        mu_fake, sigma_fake = compute_statistics_from_features(fake_features)
        
        # Compute FID
        log.info("Computing FID...")
        fid_score = compute_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)
        
        log.info(f"FID Score: {fid_score:.4f}")
        
        # Save results
        results = {
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "guidance_scale": args.guidance_scale,
            "use_refiner": args.use_refiner,
            "fid_score": fid_score
        }
        
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            log.info(f"Results saved to {args.output_file}")
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Num Samples: {args.num_samples}")
        print(f"Guidance Scale: {args.guidance_scale}")
        print(f"Use Refiner: {args.use_refiner}")
        print(f"FID Score: {fid_score:.4f}")
        print("="*50)


if __name__ == "__main__":
    main()
