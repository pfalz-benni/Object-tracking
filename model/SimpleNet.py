import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn

class SimpleObjectDetector(nn.Module):
    def __init__(self, args, num_classes=4):
        super(SimpleObjectDetector, self).__init__()

        # Load the pre-trained ResNet50 model
        backbone = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layers from ResNet50
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # The output channels of the ResNet50 backbone before the final layer are 2048
        self.feature_channels = 2048
        
        # Adding a Conv2d layer to reduce the number of channels
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.feature_channels, 512, kernel_size=1), 
            nn.ReLU()
        )
        
        # Define the RPN for bounding box regression
        self.rpn_bbox = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 9 * 4, kernel_size=1)  # 9 anchors, 4 bbox coords per anchor
        )
        
        # Define the RPN for object classification
        self.rpn_cls = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 9 * num_classes, kernel_size=1)  # 9 anchors, num_classes per anchor
        )
        
        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Extract features using the ResNet backbone
        features = self.backbone(x)
        # Further process features using the additional layers
        features = self.feature_extractor(features)
        
        # Generate proposals using the RPN
        bbox_proposals = self.rpn_bbox(features)
        class_proposals = self.rpn_cls(features)
        
        print("Bounding Box Proposals Size:", bbox_proposals.size())
        print("Class Proposals Size:", class_proposals.size())
        
        return bbox_proposals, class_proposals
    

class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 54 * 54, 128),  
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 4)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.float()
        x = x.view(1,3,216,216)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BB(nn.Module):
    def __init__(self, args, num_classes=4):
        super(BB, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        
        # Assuming you want to predict 4 bounding box coordinates
        self.bb = nn.Sequential(
            nn.InstanceNorm1d(512),
            nn.Linear(512, 4))
        
    def forward(self, x):
        x = x.view(1,3,216,216)
        x = x.float()
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        bbox = self.bb(x)
        return bbox
