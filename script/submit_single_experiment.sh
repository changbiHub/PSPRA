#!/bin/bash

# Script to submit a single experiment
# Usage: ./submit_single_experiment.sh <experiment> <model> [--overwrite]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment> <model> [--overwrite]"
    echo "Available experiments: p1, p2, p3, p3m"
    echo "Available models: TCN, RNN, GBC, RF, LR, DT, catboost, ADA, LDA, ET, LGB, QDA, NB, XGB, KNN"
    exit 1
fi

EXPERIMENT=$1
MODEL_NAME=$2
OVERWRITE=$3

RESULT_PATH="../results"
MODEL_SAVE_PATH="../models"

# Create necessary directories
mkdir -p logs
mkdir -p $RESULT_PATH
mkdir -p $MODEL_SAVE_PATH

echo "Submitting job for experiment: $EXPERIMENT, model: $MODEL_NAME"

if [ "$OVERWRITE" == "--overwrite" ]; then
    sbatch --account=rrg-ubcxzh --export=EXPERIMENT=$EXPERIMENT,MODEL_NAME=$MODEL_NAME,RESULT_PATH=$RESULT_PATH,MODEL_SAVE_PATH=$MODEL_SAVE_PATH,OVERWRITE="--overwrite" train_job.sh
else
    sbatch --account=rrg-ubcxzh --export=EXPERIMENT=$EXPERIMENT,MODEL_NAME=$MODEL_NAME,RESULT_PATH=$RESULT_PATH,MODEL_SAVE_PATH=$MODEL_SAVE_PATH train_job.sh
fi

echo "Job submitted successfully!"
echo "Monitor with: squeue -u \$USER"
