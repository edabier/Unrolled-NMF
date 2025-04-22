#!/bin/bash
#SBATCH --job-name=my_job             # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=P100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job
DATA_DIR="~/data"
LR="1e-3"
EPOCHS=100
BATCH_SIZE=32

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate amt_env

# Execute the Python script with specific arguments
srun python my_script.py --data $DATA_DIR --lr $LR --epochs $EPOCHS --batch-size $BATCH_SIZE

# Print job completion time
echo "Job finished at: $(date)"
