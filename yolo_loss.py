import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=2, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Ensure predictions and targets are tensors of the correct shape
        batch_size, num_boxes, grid_size, grid_size, num_values = predictions.shape
        assert num_values == 7, "Expected 7 values per bounding box (cx, cy, w, h, confidence, class probability)"
        
        # Split predictions into components
        pred_cx = predictions[..., 0]
        pred_cy = predictions[..., 1]
        pred_w = predictions[..., 2]
        pred_h = predictions[..., 3]
        pred_conf = predictions[..., 4]
        pred_class = predictions[..., 5:]
        
        # Same split for targets
        target_cx = targets[..., 0]
        target_cy = targets[..., 1]
        target_w = targets[..., 2]
        target_h = targets[..., 3]
        target_conf = targets[..., 4]
        target_class = targets[..., 5:]
        
        # Localization loss (Bounding Box coordinates)
        box_loss = self.lambda_coord * torch.sum((pred_cx - target_cx) ** 2 +
                                                 (pred_cy - target_cy) ** 2 +
                                                 (pred_w - target_w) ** 2 +
                                                 (pred_h - target_h) ** 2)
        
        # Objectness loss (Confidence score)
        obj_loss = torch.sum((pred_conf - target_conf) ** 2)
        
        # Classification loss
        class_loss = torch.sum((pred_class - target_class) ** 2)
        
        # Total loss
        total_loss = box_loss + obj_loss + class_loss
        
        # Make sure the total loss is a scalar
        if total_loss.numel() > 1:
            total_loss = torch.sum(total_loss)

        return total_loss, box_loss, obj_loss, class_loss