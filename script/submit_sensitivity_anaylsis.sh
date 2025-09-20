#!/bin/bash

# Script to submit all experiments with optional repetition
# Usage: ./submit_all_experiments.sh [num_repeats]

# Parse arguments
NUM_REPEATS=${1:-1}  # Default to 1 if not specified

# Define experiments and models
# EXPERIMENTS=("p1" "p2" "p3" "p3m")
EXPERIMENTS=("p3m")
# MODELS=("TCN" "RNN" "GBC" "RF" "LR" "catboost" "ADA" "LDA" "ET" "LGB" "XGB" "stacking_ensemble")
MODELS=("stacking_ensemble")
MAX_GAPS=(7)

# Configuration
RESULT_PATH_BASE="../results"
MODEL_SAVE_PATH_BASE="../models"
OVERWRITE_FLAG=""  # Set to "--overwrite" if you want to overwrite existing results

# Create necessary directories
mkdir -p logs

echo "Starting submission of all experiments..."
echo "Number of repeats: $NUM_REPEATS"
echo "Total jobs to submit: $((${#EXPERIMENTS[@]} * ${#MODELS[@]} * ${#MAX_GAPS[@]} * $NUM_REPEATS))"

# Submit jobs for each combination
job_count=0
for repeat in $(seq 1 $NUM_REPEATS); do
    for max_gap in "${MAX_GAPS[@]}"; do
        # Create paths with max gap suffix
        RESULT_PATH="${RESULT_PATH_BASE}_maxGAP${max_gap}"
        MODEL_SAVE_PATH="${MODEL_SAVE_PATH_BASE}_maxGAP${max_gap}"
        
        # Create directories for this max gap
        mkdir -p $RESULT_PATH
        mkdir -p $MODEL_SAVE_PATH
        
        for experiment in "${EXPERIMENTS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "Submitting job for experiment: $experiment, model: $model, max_gap: $max_gap (repeat: $repeat)"
                
                # Submit the job with account specification
                if [ -n "$OVERWRITE_FLAG" ]; then
                    sbatch --account=rrg-ubcxzh --export=EXPERIMENT=$experiment,MODEL_NAME=$model,RESULT_PATH=$RESULT_PATH,MODEL_SAVE_PATH=$MODEL_SAVE_PATH,MAX_GAP=$max_gap,OVERWRITE="--overwrite" train_job.sh
                else
                    sbatch --account=rrg-ubcxzh --export=EXPERIMENT=$experiment,MODEL_NAME=$model,RESULT_PATH=$RESULT_PATH,MODEL_SAVE_PATH=$MODEL_SAVE_PATH,MAX_GAP=$max_gap train_job.sh
                fi
                
                job_count=$((job_count + 1))
                
                # Add delay to avoid overwhelming the scheduler
                sleep 1
            done
        done
    done
done

echo "Submitted $job_count jobs successfully!"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in the logs/ directory"

# Show current queue status
echo ""
echo "Current job queue:"
squeue -u $USER
