#!/bin/bash
#SBATCH --account=rrg-ubcxzh
#SBATCH --job-name=pspra_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules for Compute Canada
module load python/3.11
# module load scipy-stack
module load cuda  # if using GPU models (TCN, RNN)

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Set default values if not provided
EXPERIMENT=${EXPERIMENT:-"p1"}
MODEL_NAME=${MODEL_NAME:-"TCN"}
RESULT_PATH=${RESULT_PATH:-"../results"}
MODEL_SAVE_PATH=${MODEL_SAVE_PATH:-"../models"}
MAX_GAP=${MAX_GAP:-21}
OVERWRITE=${OVERWRITE:-""}

# Change to script directory
cd /home/changbi/scratch/PSPRA/script
source /home/changbi/scratch/PSPRA/.venv/bin/activate

# Run the training script
if [ -n "$OVERWRITE" ]; then
    python train.py --experiment $EXPERIMENT --model_name $MODEL_NAME --result_path $RESULT_PATH --model_save_path $MODEL_SAVE_PATH --data_path ../data_real --max_gap $MAX_GAP --overwrite
else
    python train.py --experiment $EXPERIMENT --model_name $MODEL_NAME --result_path $RESULT_PATH --model_save_path $MODEL_SAVE_PATH --data_path ../data_real --max_gap $MAX_GAP
fi

echo "Job completed for experiment: $EXPERIMENT, model: $MODEL_NAME, max_gap: $MAX_GAP"
