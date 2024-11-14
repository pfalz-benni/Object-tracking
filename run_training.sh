#!/bin/bash
#SBATCH --gres=gpu:1
# srun singularity exec --bind /home/kannan/neuroTUM/obj_tracking:/home/kannan/ --nv singularity_events.sif python3 train.py --config sliced_baseline.json
#srun singularity exec --bind /home/kannan/neuroTUM/github/Object-tracking:/home/kannan/ --nv singularity_events.sif pip install dv==1.0.12
# srun singularity exec --bind /home/kannan/neuroTUM/github/Object-tracking:/home/kannan/ --nv singularity_events.sif pip install tonic
srun singularity exec --bind /home/kannan/neuroTUM/github/Object-tracking:/home/kannan/ --nv singularity_events.sif python3 yolov5/train.py --data data.yaml --weights '' --cfg models/yolov5n.yaml --img 416 --epochs 30
