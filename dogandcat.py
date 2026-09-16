from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

data_path = Path("data/")
image_path = data_path / "pets"

def split_train_test(source_dir: Path, target_dir: Path, train_ratio: float = 0.8, seed: int = 42):
      if target_dir.exists():
            print(f"{target_dir} already exists, skipping split.")
            return

      class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

      for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg"))

            train_images, test_images = train_test_split(
                  images, train_size=train_ratio, random_state=seed, shuffle=True
            )
            splits = {"train": train_images, "test": test_images}

            for split_name, split_images in splits.items():
                  split_class_dir = target_dir / split_name / class_dir.name
                  split_class_dir.mkdir(parents=True, exist_ok=True)
                  for img_path in split_images:
                        shutil.copy2(img_path, split_class_dir / img_path.name)

split_train_test(image_path, data_path / "pets_split")

train_dir = data_path / "pets_split" / "train"
test_dir = data_path / "pets_split" / "test"

IMAGE_SIZE = 128
BATCH_SIZE = 32

train_transform = transforms.Compose([
      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

class_names = train_data.classes
print(class_names)

image, label = train_data[0]
print(image.shape, class_names[label])

train_dataloader = DataLoader(
      train_data,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=0
)
test_dataloader = DataLoader(
      test_data,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=0
)

print(len(train_dataloader), len(test_dataloader))
print(train_dataloader.dataset[0][0].shape)

class PetClassifierCNN(nn.Module):
      def __init__(self, input_shape: int, hidden_units: int, output_shape: int, image_size: int):
            super().__init__()
            self.block_1 = nn.Sequential(
                  nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_units),
                  nn.ReLU(),
                  nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_units),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block_2 = nn.Sequential(
                  nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_units),
                  nn.ReLU(),
                  nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                  nn.BatchNorm2d(hidden_units),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(hidden_units * 32 * 32, output_shape)
            )

      def forward(self, x: torch.Tensor):
            return self.classifier(self.block_2(self.block_1(x)))

def calculate_accuracy(y_true, y_pred):
      correct = torch.eq(y_true, y_pred).sum().item()
      return (correct / len(y_pred)) * 100

torch.manual_seed(42)
model = PetClassifierCNN(
      input_shape=3,
      hidden_units=32,
      output_shape=len(class_names),
      image_size=IMAGE_SIZE
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):

      train_loss = 0
      model.train()
      for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                  print(f"Epoch {epoch} | Batch {batch} | Loss: {loss.item():.5f}")

      train_loss /= len(train_dataloader)

      test_loss = 0
      test_acc = 0
      model.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  X, y = X.to(device), y.to(device)
                  test_pred = model(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

      print(f"Epoch {epoch} | Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")