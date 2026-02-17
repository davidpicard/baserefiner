# DiT Flow Matching Framework

A PyTorch Lightning implementation of Diffusion Transformers (DiT) with Flow Matching training and a base + refiner model architecture for iterative refinement.

## Features

- **DiT Architecture**: Vision Transformer-based diffusion model with adaptive layer normalization
- **Flow Matching Training**: Linear interpolation trajectory from noise to data
- **Dual-Model Support**: Base model + optional refiner for progressive refinement
- **Distributed Training**: Multi-node, multi-GPU support via PyTorch Lightning + SLURM
- **Sampling Methods**: Euler, Heun (RK2), and adaptive step-size ODE solvers
- **Experiment Tracking**: Weights & Biases integration with visual progression monitoring
- **Production Ready**: Checkpoint recovery, SLURM auto-detection, comprehensive configuration

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pytorch-lightning hydra-core omegaconf wandb einops
```

## Project Structure

```
baserefiner/
├── model.py               # DiT architecture with AdaLNZero conditioning
├── training_module.py     # PyTorch Lightning module with flow matching loss
├── sampler.py             # ODE solvers (Euler, Heun, AdaptiveStep)
├── callbacks.py           # W&B visualization callback
├── data.py                # Dataset classes (ImageNet, SimpleImage)
├── data_module.py         # PyTorch Lightning DataModule
├── train.py               # Training orchestration with SLURM support
├── configs/
│   └── config.yaml        # Master configuration file
├── slurm_submit.sh        # SLURM job submission helper
├── .gitignore             # Repository hygiene
└── README.md              # This file
```

## Quick Start

### 1. Single GPU Training

```bash
python train.py \
    data.data_root=/path/to/imagenet \
    data.batch_size=32 \
    training.max_epochs=100
```

### 2. Multi-GPU Training (Local)

```bash
python train.py \
    training.num_nodes=1 \
    training.strategy=ddp \
    training.devices=4 \
    data.batch_size=32
```

### 3. SLURM Multi-Node Training

```bash
sbatch slurm_submit.sh \
    --nodes=2 \
    --gpus-per-node=8 \
    --data-root=/path/to/imagenet \
    --time=12:00:00
```

### 4. With Weights & Biases Logging

```bash
python train.py \
    data.data_root=/path/to/imagenet \
    logging.use_wandb=true \
    logging.project_name=dit-flow-matching \
    logging.entity=your-wandb-entity
```

### 5. Base + Refiner Model

```bash
python train.py \
    model.use_refiner=true \
    sampling.use_refiner=true \
    sampling.log_images=true
```

### 6. Resume from Checkpoint

```bash
# Automatically resumes from last.ckpt if it exists
python train.py data.data_root=/path/to/imagenet training.resume_from_checkpoint=last

# Or via SLURM:
sbatch slurm_submit.sh --resume --nodes=2 --gpus-per-node=8
```

## Configuration

The master configuration in `configs/config.yaml` is organized into sections:

### Model Configuration

```yaml
model:
  _target_: model.DiTModule
  patch_size: 2
  dim: 384
  num_blocks: 12
  num_heads: 6
  mlp_ratio: 4.0
  dropout: 0.0
  use_refiner: false  # Set to true for base + refiner architecture
```

### Data Configuration

```yaml
data:
  _target_: data_module.ImageNetDataModule
  data_root: /path/to/imagenet
  batch_size: 32
  num_workers: 8
  train_test_split: 0.8
```

### Training Configuration

```yaml
training:
  max_epochs: 100
  learning_rate: 1e-4
  warmup_steps: 5000
  # Multi-node settings
  num_nodes: 1
  strategy: ddp
  devices: 1
  sync_batchnorm: true
  # Checkpointing
  save_last: true
  resume_from_checkpoint: null
```

### Sampling Configuration

```yaml
sampling:
  num_steps: 50
  num_samples: 4
  sampler_type: heun  # euler, heun, or adaptive
  log_interval: 500
  use_refiner: false  # Use refiner velocity when available
  log_images: true
```

### Logging Configuration

```yaml
logging:
  use_wandb: false
  project_name: dit-flow-matching
  entity: null  # Your W&B entity
  experiment_name: null
  use_tensorboard: false
  tensorboard_dir: ./logs/tensorboard
```

## Advanced Usage

### Custom Configuration File

Create `configs/my_experiment.yaml`:

```yaml
defaults:
  - config

model:
  dim: 768
  num_blocks: 24
  num_heads: 12

training:
  max_epochs: 200
  learning_rate: 5e-5
```

Then run:
```bash
python train.py --config-name=my_experiment
```

### Override Specific Parameters

```bash
python train.py \
    model.dim=512 \
    model.num_blocks=16 \
    training.learning_rate=2e-4 \
    data.batch_size=64
```

### Distributed Training with Custom Setup

```bash
# Multi-node with custom synchronization
python train.py \
    training.num_nodes=4 \
    training.strategy=ddp \
    training.sync_batchnorm=true \
    training.find_unused_parameters=false
```

## SLURM Integration

The framework automatically detects SLURM environment variables and configures distributed training.

### Automatic Detection

The `train.py` script reads:
- `SLURM_JOB_NUM_NODES` → configures multi-node training
- `SLURM_GPUS_ON_NODE` → sets GPU count per node
- `SLURM_JOB_ID` → names output directory

### Job Submission Helper

The `slurm_submit.sh` script simplifies job submission. It generates and submits a SLURM job script with proper configuration.

**Basic Usage:**

```bash
# Single-node, 4 GPUs, 1 hour
sbatch slurm_submit.sh --gpus-per-node=4

# Multi-node, 8 GPUs per node, 12 hours
sbatch slurm_submit.sh --nodes=2 --gpus-per-node=8 --time=12:00:00

# With custom data and W&B logging
sbatch slurm_submit.sh \
    --nodes=4 \
    --gpus-per-node=8 \
    --data-root=/datasets/imagenet \
    --use-wandb \
    --experiment-name=large-scale-training

# Resume previous run
sbatch slurm_submit.sh --resume --nodes=2
```

**Available Options:**

```
--nodes N                 Number of nodes (default: 1)
--gpus-per-node N         Number of GPUs per node (default: 1)
--time HH:MM:SS           Time limit (default: 01:00:00)
--partition NAME          SLURM partition (default: gpu)
--job-name NAME           Job name
--data-root PATH          Path to ImageNet root
--batch-size N            Batch size (default: 32)
--config NAME             Config file name
--experiment-name NAME    Experiment name for logging
--use-wandb               Enable W&B logging
--resume                  Resume from last checkpoint
--help                    Show help message
```

### Job Monitoring

After submission, monitor your job:

```bash
# Check job status
squeue -j <JOB_ID>

# View logs in real-time
tail -f logs/<JOB_ID>.log

# Cancel job
scancel <JOB_ID>
```

## Model Architecture

### DiT (Diffusion Transformer)

The `DiTModule` consists of:
- **Patch Embedding**: Converts images to patch tokens (learnable projection)
- **Position Embeddings**: Absolute position encoding
- **Time Embeddings**: Sinusoidal embeddings for diffusion timesteps
- **Class Embeddings**: Optional class conditioning tokens
- **Transformer Blocks**: Standard transformer with AdaLNZero pre-norm

### AdaLNZero Layer Normalization

Adaptive layer normalization with zero-initialized parameters, enabling efficient conditioning on diffusion time and class labels.

```python
class AdaLNZero(nn.Module):
    # Learns affine transformation [γ, β] from time/class embeddings
    # Initialized to preserve input (γ=1, β=0)
    # Enables effective conditioning without affecting pre-training
```

### Base + Refiner Architecture

When `use_refiner=true`, the model has two separate transformer block stacks:
- **Base Model**: Coarse structural prediction
- **Refiner Model**: Detail refinement on top of base

Both receive the same embeddings but process through separate blocks, producing independent velocity predictions that are summed during sampling.

## Training Details

### Flow Matching Loss

The framework minimizes the velocity prediction error:

```
L = ||v_predicted - v_target||²
```

Where:
- `v_target = x₁ - x₀` (velocity from noise to data)
- `x_t = (1-t)x₀ + tx₁` (linear ODE trajectory)
- `v_predicted = model(x_t, t, class)` (predicted velocity)

### Learning Rate Schedule

- **Warmup**: Linear increase over `warmup_steps`
- **Decay**: Cosine annealing to zero over remaining epochs

Configurable via:
```yaml
training:
  learning_rate: 1e-4
  warmup_steps: 5000
  max_epochs: 100
```

### Distributed Training

- **Strategy**: Distributed Data Parallel (DDP) for multi-GPU
- **Synchronization**: Optional batch normalization synchronization across GPUs
- **Gradient**: Accumulated and synchronized at each step

## Sampling Methods

### Euler Method (First-Order)

Simplest integration method:
```python
x_{t+dt} = x_t + v(x_t, t) * dt
```

Fast but less accurate.

### Heun Method (Second-Order RK2)

Runge-Kutta 2nd order integration:
```python
k1 = v(x_t, t)
k2 = v(x_t + k1*dt, t+dt)
x_{t+dt} = x_t + 0.5*(k1 + k2)*dt
```

Good accuracy-speed tradeoff.

### Adaptive Step-Size Method

Adjusts step size based on local error:
```python
if ||error|| > tolerance:
    decrease step size
else:
    increase step size
```

Best quality but slower.

**Selection:**

```bash
python train.py sampling.sampler_type=heun
```

## Monitoring with Weights & Biases

When enabled, the framework logs:

### Metrics
- Training loss, validation loss (live updates)
- Learning rate progression
- Gradient norms

### Images
- Base model samples (every `log_interval` steps)
- Refined samples (when `use_refiner=true`)
- Grid visualization of progression

### Configuration
```yaml
logging:
  use_wandb: true
  project_name: dit-flow-matching
  entity: your-wandb-entity
```

Then run with W&B enabled:
```bash
python train.py logging.use_wandb=true
```

## Checkpointing and Recovery

### Automatic Checkpointing

The framework saves checkpoints every epoch (or on validation improvement if configured):

```
outputs/SLURM_<job_id>/checkpoints/
├── last.ckpt           # Most recent checkpoint
├── best.ckpt           # Best validation checkpoint (if tracking metric)
└── epoch_N.ckpt        # Specific epoch
```

### Automatic Recovery

On SLURM job requeue or manual restart:
```bash
# Automatically resumes from last.ckpt
python train.py data.data_root=/path/to/imagenet
```

The framework checks for `last.ckpt` in the checkpoint directory and resumes from there.

### Manual Checkpoint Loading

```python
# Resume from specific checkpoint
python train.py training.resume_from_checkpoint=/path/to/checkpoint.ckpt
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model dimension:
```bash
python train.py data.batch_size=16 model.dim=256
```

### SLURM Job Killed

Increase time limit:
```bash
sbatch slurm_submit.sh --time=24:00:00
```

### W&B Connection Issues

Check internet connection or disable W&B:
```bash
python train.py logging.use_wandb=false
```

### Device Mismatch Errors

Usually resolved automatically, but if issues persist, ensure consistent device setup:
```bash
# Explicitly set device
python train.py lightning_module.device=cuda:0
```

### Multi-GPU Not Detected

Verify CUDA setup:
```bash
python -c "import torch; print(torch.cuda.device_count())"
```

## Performance Tips

1. **Increase Workers**: Set `data.num_workers=16` for faster data loading
2. **Mixed Precision**: Enable for 2x speedup on newer GPUs (automatically done with `training.precision=16`)
3. **Distributed**: Use multi-node training for data parallel scaling
4. **Batch Size**: Largest sustainable batch size for better gradient estimates
5. **Gradient Accumulation**: Simulate larger batches on limited memory

## Example Commands

### Minimal Single GPU
```bash
python train.py data.data_root=/data/imagenet
```

### Full Multi-Node SLURM
```bash
sbatch slurm_submit.sh \
    --nodes=16 \
    --gpus-per-node=8 \
    --data-root=/datasets/imagenet \
    --batch-size=512 \
    --use-wandb \
    --time=24:00:00 \
    --experiment-name=large-scale-dit
```

### Base + Refiner with Monitoring
```bash
python train.py \
    model.use_refiner=true \
    sampling.use_refiner=true \
    logging.use_wandb=true \
    sampling.log_images=true \
    data.data_root=/data/imagenet
```

### Resume Training
```bash
# Automatic
python train.py data.data_root=/data/imagenet

# Explicit via SLURM
sbatch slurm_submit.sh --resume --nodes=2 --gpus-per-node=8
```


## License

MIT License - see LICENSE file for details.

