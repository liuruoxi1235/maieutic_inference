#!/bin/bash
#SBATCH --job-name=maieutic_prompting
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:1                # Use only 1 GPU to avoid distributed issues
#SBATCH --cpus-per-task=4           # Allocate some CPUs for the task
#SBATCH --mem=11G                   # Allocate memory
#SBATCH --time=24:00:00             # Set a time limit
#SBATCH --output=/home/rliu79/reasoning/logs/outputs/maieutic_prompting_%j.out
#SBATCH --error=/home/rliu79/reasoning/logs/errors/maieutic_prompting_%j.err


# Set environment variables to avoid distributed issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false    # Disable tokenizer parallelism warnings
export CUDA_VISIBLE_DEVICES=0          # Use only the first GPU
export VLLM_USE_MODELSCOPE=True

set -x -e

# Load modules and activate conda environment
module load anaconda
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate reasoning

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs:"
nvidia-smi --list-gpus


# Run the Python script directly (no torchrun needed)
python /home/rliu79/reasoning/Agent_v3_5.py
#python /home/rliu79/reasoning/try.py

echo "END TIME: $(date)"