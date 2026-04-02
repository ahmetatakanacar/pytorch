import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

torch.manual_seed(42)

images = torch.randn(size = (32,3,32,32)) # (batch_size, color_chanels, height, width)
test_image = images[0]
print(f"Image batch space: {images.shape}") # -> (batch_size, color_chanels, height, width)
print(f"Single image shape: {test_image.shape}") # -> (color_chanels, height, width)

conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

print(conv_layer(test_image).shape)
print(test_image.shape)

x = torch.randn(1, 3, 32, 32)   # [batch, channels, height, width]
print("Input:", x.shape)

conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

x = conv1(x)
print("After Conv1:", x.shape)

x = conv2(x)
print("After Conv2:", x.shape)

x = pool1(x)
print("After MaxPool1:", x.shape)

conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
pool2 = nn.MaxPool2d(2, 2)

x = conv3(x)
print("After Conv3:", x.shape)

x = conv4(x)
print("After Conv4:", x.shape)

x = pool2(x)
print("After MaxPool2:", x.shape)








