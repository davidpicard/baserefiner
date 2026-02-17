#!/bin/bash
# SLURM job submission script for distributed training
# Usage: sbatch slurm_submit.sh [OPTIONS]
#
# Example:
#   sbatch slurm_submit.sh --nodes=2 --gpus-per-node=8 --data-root=/path/to/imagenet
#   sbatch slurm_submit.sh --nodes=1 --gpus-per-node=4 --resume  # Resume from checkpoint

set -e

# Default parameters
NODES=1
GPUS_PER_NODE=1
TIME="01:00:00"
PARTITION="gpu"
JOB_NAME="dit-flow-matching"
DATA_ROOT="/path/to/imagenet"
BATCH_SIZE=32
CONFIG="config"
USE_WANDB=false
RESUME=false
EXPERIMENT_NAME="dit-flow-matching"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help)
            echo "Usage: sbatch slurm_submit.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nodes N                Number of nodes (default: 1)"
            echo "  --gpus-per-node N        Number of GPUs per node (default: 1)"
            echo "  --time HH:MM:SS          Time limit (default: 01:00:00)"
            echo "  --partition NAME         SLURM partition (default: gpu)"
            echo "  --job-name NAME          Job name (default: dit-flow-matching)"
            echo "  --data-root PATH         Path to ImageNet root (default: /path/to/imagenet)"
            echo "  --batch-size N           Batch size (default: 32)"
            echo "  --config NAME            Config file (default: config, .yaml extension added)"
            echo "  --experiment-name NAME   Experiment name for logging"
            echo "  --use-wandb              Enable Weights & Biases logging"
            echo "  --resume                 Resume from last checkpoint"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate total tasks
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# Create job script
JOB_SCRIPT=$(mktemp)
cat > "$JOB_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --nodes=NODES
#SBATCH --gpus-per-node=GPUS_PER_NODE
#SBATCH --ntasks-per-node=GPUS_PER_NODE
#SBATCH --cpus-per-task=8
#SBATCH --time=TIME
#SBATCH --partition=PARTITION
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --requeue

# Create logs directory
mkdir -p logs

# Load modules
module load cuda/11.8
module load cudnn/8.6

# Activate environment (adjust as needed)
source venv/bin/activate

# Print environment info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo ""

# Set environment variables for distributed training
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=true

# Build training command
CMD="python train.py \
    --config-name=CONFIG \
    data.data_root=DATA_ROOT \
    data.batch_size=BATCH_SIZE \
    logging.experiment_name=EXPERIMENT_NAME"

# Add resume flag if needed
if [ "RESUME" = "true" ]; then
    CMD="$CMD training.resume_from_checkpoint=last"
fi

# Add W&B logging if enabled
if [ "USE_WANDB" = "true" ]; then
    CMD="$CMD logging.use_wandb=true"
fi

echo "Running: $CMD"
echo ""

# Run training
eval $CMD
EOF

# Replace placeholders in the job script
sed -i "s|JOBNAME|$JOB_NAME|g" "$JOB_SCRIPT"
sed -i "s|NODES|$NODES|g" "$JOB_SCRIPT"
sed -i "s|GPUS_PER_NODE|$GPUS_PER_NODE|g" "$JOB_SCRIPT"
sed -i "s|TIME|$TIME|g" "$JOB_SCRIPT"
sed -i "s|PARTITION|$PARTITION|g" "$JOB_SCRIPT"
sed -i "s|CONFIG|$CONFIG|g" "$JOB_SCRIPT"
sed -i "s|DATA_ROOT|$DATA_ROOT|g" "$JOB_SCRIPT"
sed -i "s|BATCH_SIZE|$BATCH_SIZE|g" "$JOB_SCRIPT"
sed -i "s|EXPERIMENT_NAME|$EXPERIMENT_NAME|g" "$JOB_SCRIPT"
sed -i "s|RESUME|$RESUME|g" "$JOB_SCRIPT"
sed -i "s|USE_WANDB|$USE_WANDB|g" "$JOB_SCRIPT"

# Make script executable and submit
chmod +x "$JOB_SCRIPT"

echo "Submitting SLURM job..."
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Time limit: $TIME"
echo "Partition: $PARTITION"
echo "Config: $CONFIG"
echo "Data root: $DATA_ROOT"
echo "Experiment: $EXPERIMENT_NAME"
if [ "$RESUME" = "true" ]; then
    echo "Mode: RESUME"
fi
echo ""

# Submit the job
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Job submitted with ID: $JOB_ID"
echo "Job script saved to: $JOB_SCRIPT"
echo ""
echo "To monitor job:"
echo "  squeue -j $JOB_ID"
echo "  tail -f logs/$JOB_ID.log"
echo ""
echo "To cancel job:"
echo "  scancel $JOB_ID"
