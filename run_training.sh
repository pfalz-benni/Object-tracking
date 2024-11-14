#!/bin/bash
#SBATCH --gres=gpu:1
srun singularity exec --bind /home/kannan/neuroTUM/github/Object-tracking:/home/kannan/ --nv singularity_events.sif python3 train.py --config yolo5n_mse.json
