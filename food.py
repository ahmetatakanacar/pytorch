import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 1

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

class_names = train_data.classes

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True)

def create_model(num_classes: int, device: str):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = create_model(num_classes=len(class_names), device=device)