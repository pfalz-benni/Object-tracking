import argparse, json, os, mlflow, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.Yolo import TinyYOLOv2
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import ImageDraw
import torchvision
from pathlib import Path


def load_jsonl_labels(jsonl_path: Path):
    labels = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['detection']:
                for detection in data['detection']:
                    bbox = detection['bounding_box']
                    label = detection['label']
                    labels.append((bbox, label))
            else:
                labels.append(None)  # No detection case

    return labels


def main(args):
    
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file), 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")


    # Define your model, optimizer, and criterion
    model = eval(args.architecture)(args).to(args.device)

    file_paths = ['toy/14']
    data_test=[]
    for file_path in file_paths:
        path_event = Path('Event_frames/'+  file_path + '.npy')
        path_label = Path('Label/' + file_path + '/detections_updated.jsonl')  

        # Load event data and label bounding boxes
        event_data = np.load(path_event)
        label_bb = load_jsonl_labels(path_label)
        min_len = min(len(event_data), len(label_bb))

        for index in range(min_len):
            label = label_bb[index]

            if label is None:
                class_out = 0
                bbox = None
            else:
                bbox, class_label = label

                # Set class output based on label
                if class_label == 'banana':
                    class_out = 1
                elif class_label == 'teddy bear':
                    class_out = 2  # Class 2 for "toy"
                else:
                    class_out = 0
            data_test.append((event_data[index], bbox, class_out))
            
    assert args.batch_size == 1
    test_data = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, \
                                    num_workers=2 )

    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        raise ValueError("Please provide a checkpoint file.")

    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data):

            data = data.to(args.device)
            output = model(data)
            output_tensor = output.cpu()
            target = target.cpu()

            print(output_tensor.shape)
            print(target.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a config file 
    parser.add_argument("--config_file", type=str, default='test_config', \
                        help="path to JSON configuration file")
    # load weights from a checkpoint
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint")
    args = parser.parse_args()
    main(args)