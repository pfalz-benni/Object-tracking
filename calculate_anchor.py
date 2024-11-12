import json
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_annotations_from_json(json_file_path):
    annotations = []
    
    # Open the JSON file and read each line
    with open(json_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            
            # Check for 'detection' field in the JSON object
            if data['detection']:
                boxes = []
                for detection in data['detection']:
                    # Extract the bounding box coordinates
                    xmin, ymin, xmax, ymax = detection['bounding_box']
                    boxes.append([xmin, ymin, xmax, ymax])
                
                # Append to annotations list
                annotations.append({'boxes': boxes})

    return annotations

def calculate_anchors(annotations, num_anchors=5):
    wh = []
    for anno in annotations:
        for box in anno['boxes']:
            # Calculate width and height
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            wh.append([w, h])
    wh = np.array(wh)

    # Use KMeans to find the anchor boxes
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(wh)
    anchors = kmeans.cluster_centers_
    return anchors

def process_multiple_json_files(json_file_paths, num_anchors=5):
    all_annotations = []
    
    # Iterate over each JSON file
    for json_file_path in json_file_paths:
        if not os.path.isfile(json_file_path):
            print(f"File not found: {json_file_path}")
            continue
        
        # Extract annotations from the current file
        annotations = extract_annotations_from_json(json_file_path)
        all_annotations.extend(annotations)
    
    # Calculate anchors from all collected annotations
    anchors = calculate_anchors(all_annotations, num_anchors)
    return anchors

# List of paths to your JSON files
json_file_paths = ['/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/7/detections_updated.jsonl',
                   '/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/8/detections_updated.jsonl',
                   '/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/15/detections_updated.jsonl',
                   '/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/17/detections_updated.jsonl',
                   '/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/23/detections_updated.jsonl',
                   '/home/kannan/Kannan_Workspace/obj_tracking/9_7_24/24/detections_updated.jsonl'


]

# Number of anchors to calculate
num_anchors = 5

# Process the JSON files and calculate anchors
anchors = process_multiple_json_files(json_file_paths, num_anchors)

# Print the calculated anchors
print("Calculated anchors (width, height):", anchors)
