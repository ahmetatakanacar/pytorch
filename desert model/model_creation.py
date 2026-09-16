import torch
from torch import nn

class DesertClassifier(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, 
                      out_channels=hidden_size, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, 
                      out_channels=hidden_size, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
                    nn.Conv2d(in_channels=hidden_size,
                              out_channels=hidden_size, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_size, 
                              out_channels=hidden_size, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, 
                                 stride=2)
                )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size * 16 * 16,
                      out_features=output_size)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))