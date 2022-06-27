import torch
import torch.nn as nn

class OxfordPetsModel(nn.Module):
    def __init__(self, feature_maps, classes):
        super().__init__()
        # input channels rgb = 3, out_channels by convolution, kernel size
        self.layer1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(6, 6), stride=(1, 1), padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(nn.Linear(6400, 1600))
        self.layer4 = nn.Sequential(nn.Linear(1600, 400))
        self.layer5 = nn.Sequential(nn.Linear(400, 37))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
