#!/bin/bash
#SBATCH --job-name=ralmu_training     # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH -p P100                       # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=1             # Request 8 CPU cores
#SBATCH --time=4:00:00                # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for the job
N_WORKERS=$SLURM_CPUS_PER_TASK
ITER=10
LR="1e-3"
EPOCHS=2
BATCH_SIZE=10
LENGTH=10
SUBSET=0.1
SPLIT=0.8
# FILTER=True
# DOWNSAMPLE=True

# Activate the environment
eval "$(conda shell.bash hook)"
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate amt-env
# wandb login

# Execute the Python script with specific arguments
srun python -m memory_profiler /home/ids/edabier/AMT/Unrolled-NMF/test_trainer.py --num_workers $N_WORKERS --iter $ITER --lr $LR --epochs $EPOCHS --batch $BATCH_SIZE --length $LENGTH --subset $SUBSET --split $SPLIT # --filter $FILTER
# srun python -m memory_profiler /home/ids/edabier/AMT/Unrolled-NMF/maps_to_cqt.py --subset $SUBSET --downsample $DOWNSAMPLE

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"