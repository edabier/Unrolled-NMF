#!/bin/bash
# SBATCH --job-name=ralmu_training             # Name of your job
# SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
# SBATCH --error=%x_%j.err             # Error file
# SBATCH --partition=P100              # Partition to submit to (A100, V100, etc.)
## SBATCH --gres=gpu:1                  # Request 1 GPU
## SBATCH --cpus-per-task=8             # Request 8 CPU cores
## SBATCH --mem=32G                     # Request 32 GB of memory
# SBATCH --time=4:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for the job
LR="1e-3"
EPOCHS=10
BATCH_SIZE=1
SUBSET=0.01
SPLIT=0.8

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate amt-env
# wandb login

# Execute the Python script with specific arguments
srun python /home/ids/edabier/AMT/Unrolled-NMF/trainer.py --lr $LR --epochs $EPOCHS --batch $BATCH_SIZE --subset $SUBSET --split $SPLIT

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"