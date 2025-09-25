#!/bin/bash
#SBATCH --job-name=testing_models     # Name of your job
#SBATCH --output=%x_%j.out      # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err       # Error file
#SBATCH -p P100                 # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --cpus-per-task=1       # Request 1 CPU cores
#SBATCH --time=24:00:00         # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for the job
N_WORKERS=$SLURM_CPUS_PER_TASK
ITER=10
SUBSET=0.1
LENGTH=10
SPLIT=0.8

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate amt-env

# Execute the Python script with specific arguments
srun python /home/ids/edabier/AMT/Unrolled-NMF/run_test.py --num_workers $N_WORKERS --iter $ITER --length $LENGTH --subset $SUBSET --split $SPLIT

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"