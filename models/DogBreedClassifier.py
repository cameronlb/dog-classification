import torch
import torch.nn as nn
import torchvision
from models.model_utils import initialize_feature_extractor


class DogBreedClassifier(nn.Module):
    def __init__(self):
        super(DogBreedClassifier, self).__init__()
        self.resnet = initialize_feature_extractor(torchvision.models.efficientnet_b0(pretrained=True))

        self.connected_layers = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Dropout(0.75),
            nn.Linear(256, 120),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.connected_layers(x)
        return x


