import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import string
import random
import torch.nn as nn

import tqdm


def train_epoch(model, train_loader, criterion, optimizer, args):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    args = args
    for i, data in enumerate(train_loader):
    # for i, data in enumerate(tqdm.tqdm(train_loader, desc="Training Progress", unit="batch")):

        image, target = data
        output = model(image.to(args.device))
        target0 = target[0].to(args.device).to(torch.float32)

        # print(i, end=' ')

        # loss_val, box_l, obj_l, class_l  = criterion(output[0], target[0])
        loss_val = torch.zeros((1), dtype=torch.float, device=args.device)

        loss_val = criterion(output[0], target0)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()

        # if i >= 9:
        #     break

    return model, total_loss / len(train_loader), total_iou/len(train_loader)

def validate_epoch(model, val_loader, criterion, args):

    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    for i, data in enumerate(val_loader):
        image, target = data
        output = model(image.to(args.device))
        target0 = target[0].to(args.device).to(torch.float32)

        loss = torch.zeros((1), dtype=torch.float, device=args.device)
        
        # loss, box_l, obj_l, class_l = criterion(output, target)
        loss = criterion(output[0], target0)

        loss.backward()
        total_loss += loss.item()

        # # Convert tensors to numpy arrays for visualization
        # inputs_np = inputs.cpu().numpy()
        # output_np = output.detach().cpu().numpy()
        # targets_np = targets.cpu().numpy()

        # # Visualize the first image in the batch
        # visualize(inputs_np[0], output_np[0], targets_np[0])

        # if i >= 9:
        #     break

    return total_loss / len(val_loader), total_iou/len(val_loader)

def visualize(image, predicted_bbox, true_bbox):
    # Assuming image, predicted_bbox, true_bbox are numpy arrays
    plt.figure(figsize=(10, 5))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Assuming image is in (3, H, W) format
    plt.imshow(image)
    plt.axis('off')
    
    # Plot the true bounding box
    plt.subplot(1, 2, 2)
    plt.title('True Bounding Box')
    if image.shape[0] == 3:
        plt.imshow(np.transpose(image, (1, 2, 0)))
    else:
        plt.imshow(image)
    plot_bbox(true_bbox)
    plt.axis('off')
    
    # Plot the predicted bounding box
    plt.subplot(1, 2, 2)
    plt.title('Predicted Bounding Box')
    if image.shape[0] == 3:
        plt.imshow(np.transpose(image, (1, 2, 0)))
    else:
        plt.imshow(image)
    plot_bbox(predicted_bbox)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    save_path = os.path.join('valid_plot1/', f'image_{random_name}.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory

def plot_bbox(bbox, color='r'):
    # bbox should be in format [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = bbox
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2))

def top_k_checkpoints(args, artifact_uri):
    # list all files ends with .pth in artifact_uri
    model_checkpoints = [f for f in os.listdir(artifact_uri) if f.endswith(".pth")]

    # but only save at most args.save_k_best models checkpoints
    if len(model_checkpoints) > args.save_k_best:
        # sort all model checkpoints by validation loss in ascending order
        model_checkpoints = sorted([f for f in os.listdir(artifact_uri) if f.startswith("model_best_ep")], \
                                    key=lambda x: float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(artifact_uri, model_checkpoints[-1]))

def calculate_iou(box1, box2):
    # box1, box2: tensors of shape (N, 4) where N is the number of samples
    # box: [xmin, ymin, xmax, ymax]
    
    # Calculate intersection coordinates
    xmin_inter = torch.max(box1[:, 0], box2[:, 0])
    ymin_inter = torch.max(box1[:, 1], box2[:, 1])
    xmax_inter = torch.min(box1[:, 2], box2[:, 2])
    ymax_inter = torch.min(box1[:, 3], box2[:, 3])
    
    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)
    
    # Calculate area of both bounding boxes
    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / torch.clamp(union_area, min=1e-6)  # Avoid division by zero
    
    return iou

def convert_to_xyxy(bbox, grid_size, anchor_idx, anchors, img_size):
    """
    Convert YOLO bounding box format to [xmin, ymin, xmax, ymax].

    :param bbox: Bounding box tensor [bx, by, bw, bh]
    :param grid_size: Size of the grid (e.g., 8 for 8x8)
    :param anchor_idx: Index of the anchor box
    :param anchors: List of anchor box dimensions
    :param img_size: Tuple of (image_width, image_height)
    :return: Bounding box in [xmin, ymin, xmax, ymax] format
    """
    bx, by, bw, bh = bbox

    grid_x = (anchor_idx % grid_size)
    grid_y = (anchor_idx // grid_size)

    pw, ph = anchors

    # Calculate bounding box coordinates relative to image
    bx = (bx + grid_x) / grid_size
    by = (by + grid_y) / grid_size
    bw = bw * pw
    bh = bh * ph

    # Convert center coordinates to corners
    xmin = bx - bw / 2
    ymin = by - bh / 2
    xmax = bx + bw / 2
    ymax = by + bh / 2

    # Scale to image size
    xmin *= img_size[0]
    ymin *= img_size[1]
    xmax *= img_size[0]
    ymax *= img_size[1]

    return torch.tensor([xmin, ymin, xmax, ymax])

