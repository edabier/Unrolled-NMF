#!/bin/bash
#SBATCH --job-name=training     # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH -p P100                    # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=1             # Request 1 CPU cores
#SBATCH --time=24:00:00                # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for the job
N_WORKERS=$SLURM_CPUS_PER_TASK
EPOCHS=20
BATCH_SIZE=4
LENGTH=10
SPLIT=0.8
CLIP=1

# Activate the environment
eval "$(conda shell.bash hook)"
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate amt-env
# wandb login

# Execute the Python script with specific arguments

# SUBSET=0.05
# LR="1e-3"
# ITER=10
# srun python /home/ids/edabier/AMT/Unrolled-NMF/warmup_trainer.py --num_workers $N_WORKERS --iter $ITER --lr $LR --epochs $EPOCHS --batch $BATCH_SIZE --length $LENGTH --subset $SUBSET --split $SPLIT --clip $CLIP

SUBSET=0.1
LR="1e-2"
ITER=10
srun python /home/ids/edabier/AMT/Unrolled-NMF/trainer.py --num_workers $N_WORKERS --iter $ITER --lr $LR --epochs $EPOCHS --batch $BATCH_SIZE --length $LENGTH --subset $SUBSET --split $SPLIT --clip $CLIP

# srun python /home/ids/edabier/AMT/Unrolled-NMF/maps_to_cqt.py --subset $SUBSET
# srun python /home/ids/edabier/AMT/Unrolled-NMF/guitarset_to_cqt.py --subset $SUBSET
# srun python /home/ids/edabier/AMT/Unrolled-NMF/run_test.py --num_workers $N_WORKERS --iter $ITER --length $LENGTH --subset $SUBSET --split $SPLIT

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"