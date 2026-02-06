import torch
import torch.nn as nn
from torchvision import models

class ConceptPredictor(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(ConceptPredictor, self).__init__()
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            # Remove the classification head
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Remove classification head
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone_name} not supported.")
            
        # Concept Head: 4 neurons for NO, NC, CO, PSC
        self.concept_head = nn.Linear(in_features, 4)
        
    def forward(self, x):
        features = self.backbone(x)
        raw_output = self.concept_head(features)
        
        # Clamp output to valid range [0, 5] as per requirements
        clamped_output = torch.clamp(raw_output, 0, 5)
        
        return clamped_output
