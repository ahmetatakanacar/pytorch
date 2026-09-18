import torch
from torch import nn
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.Food101(
    root="data",
    split="train",
    download=True,
    transform=data_transform
)

test_data = datasets.Food101(
    root="data",
    split="test",
    download=True,
    transform=data_transform
)