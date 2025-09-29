#!/bin/bash
#SBATCH --job-name=guitarset_to_cqt     # Name of your job
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
SUBSET=1 

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate amt-env

# Execute the Python script with specific arguments
srun python /home/ids/edabier/AMT/Unrolled-NMF/guitarset_to_cqt.py --subset $SUBSET

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"