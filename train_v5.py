from ultralytics import YOLO
import torch

model = YOLO('yolov5nu.pt') 

# Train the model
model.train(
    data='data.yaml',  # Path to data config
    epochs=30,                 # Number of training epochs
    imgsz=416,                 # Image size for Tiny YOLO
    batch=16,                  # Batch size
    name='tiny_yolo_banana_teddy',  # Experiment name
    device=0                   # Specify GPU (use 'cpu' if no GPU)
)