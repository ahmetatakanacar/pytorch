import os
from pathlib import Path
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_path = Path("data/")
image_path = data_path / "intel"

train_dir = image_path / "seg_train" / "seg_train"
test_dir = image_path / "seg_test" / "seg_test"


def check_data(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"# of directories: {len(dirnames)} and {len(filenames)} images in '{dirpath}'.")

manual_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 1

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       transform: transforms.Compose, 
                       batch_size: int, 
                       num_workers=NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers, 
                                  pin_memory=True,)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 num_workers=num_workers, 
                                 pin_memory=True,)

    return train_dataloader, test_dataloader, class_names

def create_model(num_classes, device):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )

    return model.to(device)

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (y_pred.argmax(dim=1) == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            test_acc += (y_pred.argmax(dim=1) == y).sum().item() / len(y_pred)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):

        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(f"epoch {epoch} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def main():
    check_data(train_dir)
    check_data(test_dir)

    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        transform=manual_transform,
                                                                        batch_size=32,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(42)
    model = create_model(num_classes=len(class_names), device=device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / total: {total:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    results = train(model, 
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    loss_fn,
                    epochs=5,
                    device=device,
    )

    torch.save({"model_state": model.state_dict(), "classes": class_names}, "intel_efficientnet_b0.pt")
    print("model saved: intel_efficientnet_b0.pt")


if __name__ == "__main__":
    main()