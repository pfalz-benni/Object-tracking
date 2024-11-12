import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import glob
import albumentations as A
import torch
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from typing import Any, Tuple, List
from torchvision import transforms

class Data(Dataset):
    def __init__(
        self,
        split: str,  # 'train', 'valid', or 'test'
        transform=None
    ):
        super().__init__()

        # Load file paths based on split
        split_path = f"dataset/{split}_set.npy"
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_path} not found. Please ensure the dataset is split and saved.")

        # Load file paths from the saved .npy files
        self.file_paths = np.load(split_path, allow_pickle=True)
        self.transform_flag = split

        # Define transformations with Albumentations
        self.transform = transform or A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            # A.RandomSizedBBoxSafeCrop(width=80, height=80, p=0.5, erosion_rate=0.3),
            A.VerticalFlip(p=0.5),
            # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # Load the data
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        events, frame_path, label_bb, class_out = self.data[index]

        # Create a blank frame for visualization
        frame = np.zeros((260, 346, 3), dtype=np.uint8)
        frame[:, :, 0] = events.astype(np.uint8) * 255
        frame[:, :, 1] = events.astype(np.uint8) * 255
        frame[:, :, 2] = events.astype(np.uint8) * 255

        # Apply transformations if a bounding box is present
        if label_bb is not None and self.transform_flag:
            transformed = self.transform(image=frame, bboxes=[label_bb], class_labels=[class_out])
            transformed_image = transformed['image']
            transformed_bbox = transformed['bboxes'][0]
            resized_image, resized_label = self.resize_image_and_bounding_box_with_padding(
                transformed_image, transformed_bbox, class_out, (256, 256)
            )
        else:
            resized_image, resized_label = self.resize_image_and_bounding_box_with_padding(
                frame, label_bb, class_out, (256, 256)
            )

        # Convert resized image to PyTorch tensor
        image_tensor = transforms.ToTensor()(resized_image)

        return image_tensor, resized_label

    def _load_data(self):
        data_class_0 = []
        data_class_1 = []
        data_class_2 = []

        # Load data based on the file paths in self.file_paths
        for file_path in self.file_paths:
            object_type, file_id = file_path.split('/')
            file_id = file_id.split('.')[0]
            # print(object_type)
            # print(file_id)

            path_event = Path(f'Event_frames/{object_type}/{file_id}.npy')
            path_frame = Path(f'Data_frames/{object_type}/{file_id}/output_frames/')
            images_in_path = self.get_all_images(path_frame)
            path_label = Path(f'Label/{object_type}/{file_id}/detections_updated.jsonl')

            # Load event data and labels
            event_data = np.load(path_event)
            label_bb = self.load_jsonl_labels(path_label)

            # Use the minimum length to avoid index errors
            min_len = min(len(event_data), len(images_in_path), len(label_bb))
            for index in range(min_len):
                label = label_bb[index]
                if label is None:
                    class_out = 0
                    bbox = None
                else:
                    bbox, class_label = label
                    class_out = 1 if class_label == 'banana' else 2 if class_label == 'teddy bear' else 0

                # Balance the dataset by class
                if class_out == 0:
                    data_class_0.append((event_data[index], images_in_path[index], bbox, class_out))
                elif class_out == 1:
                    data_class_1.append((event_data[index], images_in_path[index], bbox, class_out))
                elif class_out == 2:
                    data_class_2.append((event_data[index], images_in_path[index], bbox, class_out))

        # Balance the number of samples between classes
        min_samples = min(len(data_class_0), len(data_class_1), len(data_class_2))
        if min_samples == 0:
            raise ValueError("No data available after balancing classes.")
        balanced_data = data_class_0[:min_samples] + data_class_1[:min_samples] + data_class_2[:min_samples]

        return balanced_data

    def load_jsonl_labels(self, jsonl_path: Path) -> List:
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

    def get_all_images(self, folder_path: Path) -> List[str]:
        image_paths = glob.glob(os.path.join(folder_path, '*.png'))
        return sorted(image_paths)

    def resize_image_and_bounding_box_with_padding(self, image: np.ndarray, bbox: Tuple, class_out,  new_size: Tuple[int, int], num_bboxes=1) -> Tuple[np.ndarray, Tuple]:
        original_height, original_width = image.shape[:2]
        target_width, target_height = new_size

        # Calculate the scale while maintaining aspect ratio
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image with the calculated scale
        resized_image = cv2.resize(image, (new_width, new_height))

        # Calculate padding to be applied to the resized image
        pad_width = (target_width - new_width) // 2
        pad_height = (target_height - new_height) // 2
        pad_top = pad_height
        pad_bottom = target_height - new_height - pad_top
        pad_left = pad_width
        pad_right = target_width - new_width - pad_left

        # Apply padding to the resized image
        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Padding color: black
        )

        grid_size = 8  # The size of the feature map grid (8x8)
        num_anchors = 5  
        num_classes = 2
        
       # Initialize ground truth array
        bboxes = np.zeros((num_anchors, grid_size, grid_size, 6), dtype=np.float32)
        
        if bbox is not None:
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            x_min, y_min, x_max, y_max = bbox

            # Convert bbox to center format
            x_min_resized = (x_min * scale_x) + pad_left
            y_min_resized = (y_min * scale_y) + pad_top
            x_max_resized = (x_max * scale_x) + pad_left
            y_max_resized = (y_max * scale_y) + pad_top

            # Clip bounding box coordinates to be within the valid range [0, 1]
            bbox_width = min(max(x_max_resized - x_min_resized, 0), target_width)
            bbox_height = min(max(y_max_resized - y_min_resized, 0), target_height)
            center_x = min(max((x_min_resized + x_max_resized) / 2, 0), target_width)
            center_y = min(max((y_min_resized + y_max_resized) / 2, 0), target_height)
            # bbox_width = x_max_resized - x_min_resized
            # bbox_height = y_max_resized - y_min_resized
            # center_x = (x_min_resized + x_max_resized) / 2
            # center_y = (y_min_resized + y_max_resized) / 2

            # Normalize bounding box coordinates to the grid size
            center_x /= target_width
            center_y /= target_height
            bbox_width /= target_width
            bbox_height /= target_height
            # print(bbox_height)
            # print(bbox_width)

            # Map normalized coordinates to grid cell
            grid_x = int(center_x * grid_size)
            grid_y = int(center_y * grid_size)

            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                # Create the ground truth entry for the first anchor box
                bboxes[0, grid_y, grid_x] = [
                    center_x,    # Center x (normalized to grid cell size)
                    center_y,    # Center y (normalized to grid cell size)
                    bbox_width,  # Width (normalized)
                    bbox_height, # Height (normalized)
                    1.0,         # Objectness score
                    class_out    # Class index
                ]
            
        return padded_image, bboxes


    # def plot_image_with_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    #     x_min, y_min, x_max, y_max = bbox
    #     width = x_max - x_min
    #     height = y_max - y_min
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor='red', facecolor='none', linewidth=2))
    #     plt.show()


def visualize_event_data_with_labels(dataloader, num_samples=5):
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_samples:
            break  # Stop after visualizing the specified number of samples

        # Convert the PyTorch tensor back to a NumPy array for visualization
        image_np = images[0].numpy().transpose(1, 2, 0)  # Get first sample in the batch and format it for plotting
        bboxes = labels[0]  # Get corresponding labels (bounding boxes and classes)

        # Plot the image with the bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        grid_size = bboxes.shape[1]  # Assume it's a square grid (e.g., 8x8 for grid_size=8)

        for y in range(grid_size):
            for x in range(grid_size):
                bbox = bboxes[0, y, x]  # Extract anchor box info (first anchor in this example)
                if bbox[4] > 0.5:  # Objectness score threshold
                    # Convert normalized bbox coordinates to pixel space
                    center_x = bbox[0] * 256
                    center_y = bbox[1] * 256
                    width = bbox[2] * 256
                    height = bbox[3] * 256
                    x_min = center_x - width / 2
                    y_min = center_y - height / 2

                    # Draw the bounding box
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x_min, y_min), width, height,
                            edgecolor='red', facecolor='none', linewidth=2
                        )
                    )
                    plt.text(
                        x_min, y_min - 10,
                        f"Class: {int(bbox[5])}",
                        color='red', fontsize=12, backgroundcolor='black'
                    )

        plt.show()

# # Initialize the datasets for train, validation, and test
# train_dataset = Data(split="train")
# valid_dataset = Data(split="valid")
# test_dataset = Data(split="test")

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
# visualize_event_data_with_labels(train_loader, num_samples=80)
# # # Example usage: Print shape of images and labels in the train set
# # for i, (image, target) in enumerate(train_loader):
# #     print(image.size(), target.size())
# #     # if i == 5:  # Limit output for demonstration purposes
# #     #     break