import torch
from torch import nn
import numpy as np
from pathlib import Path
import os
from PIL import Image
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import torchvision


data_path = Path("data/")
image_path = data_path / "desert101"
print(image_path)

def check_data(dir_path):
      for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"# of directories: {len(dirnames)} and {len(filenames)} images in {dirpath}")
check_data(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image = random.choice(image_path_list)
img = Image.open(random_image)
#img.show()

data_transform = transforms.Compose([
      transforms.Resize(size = (64,64)),
      transforms.RandomHorizontalFlip(p = 0.2),
      transforms.TrivialAugmentWide(),
      transforms.ToTensor(),
      transforms.Normalize(mean = [0.6,0.6,0.6], std = [0.3,0.3,0.3])
])

train_data = datasets.ImageFolder(root = train_dir, transform = data_transform, target_transform = None)
test_data = datasets.ImageFolder(root = test_dir, transform = data_transform, target_transform = None)
class_names = train_data.classes

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)

test_dataloader = DataLoader(dataset = test_data,
                              batch_size = BATCH_SIZE,
                              shuffle = False)

class DesertClassifier(nn.Module):
      def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
      
            self.conv_block_1 = nn.Sequential(
                  nn.Conv2d(in_channels = input_shape,
                            out_channels = hidden_units,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = hidden_units,
                            out_channels = hidden_units,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size = 2,
                               stride = 2)

            )
            self.conv_block_2 = nn.Sequential(
                  nn.Conv2d(in_channels = hidden_units,
                            out_channels = hidden_units,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = hidden_units,
                            out_channels = hidden_units,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size = 2,
                               stride = 2)

            )

            self.classifier = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features = hidden_units * 16 * 16,
                            out_features = output_shape),
                  nn.Dropout(p=0.5)
            )

      def forward(self, x):
            return self.classifier(self.conv_block_2(self.conv_block_1(x)))
      
model_0 = DesertClassifier(input_shape = 3,
                           hidden_units = 32,
                           output_shape = len(class_names))

print(summary(model_0, input_size = [1,3,64,64]))

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
      
      model.train()

      train_loss = 0
      train_acc = 0

      for batch, (X, y) in enumerate(dataloader):

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
      
      train_loss = train_loss / len(dataloader)
      train_acc = train_acc / len(dataloader)
      return train_loss, train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,loss_fn: torch.nn.Module):
      model.eval()

      test_loss = 0
      test_acc = 0

      with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                  test_pred_logits = model(X)
                  loss = loss_fn(test_pred_logits, y)
                  test_loss += loss.item()

                  test_pred_labels = test_pred_logits.argmax(dim = 1)
                  test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
            
            test_loss = test_loss / len(dataloader)
            test_acc = test_acc / len(dataloader)
            return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 10):
      
      results = {"train_loss": [],
                 "train_acc": [],
                 "test_loss": [],
                 "test_acc": []}
      
      for epoch in range(epochs):
            train_loss, train_acc = train_step(model = model,
                                               dataloader = train_dataloader,
                                               loss_fn = loss_fn,
                                               optimizer = optimizer)
            test_loss, test_acc = test_step(model = model,
                                            dataloader = test_dataloader,
                                            loss_fn = loss_fn)
            
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc: {test_acc}")

            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
      return results

EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_0.parameters(), lr = 0.001)
model_0_results = train(model = model_0,
                        train_dataloader = train_dataloader,
                        test_dataloader = test_dataloader,
                        loss_fn = loss_fn,
                        optimizer = optimizer,
                        epochs = EPOCHS)
print(model_0_results)

def plot_loss_curves(results):
      loss = results["train_loss"]
      test_loss = results["test_loss"]
      accuracy = results["train_acc"]
      test_accuracy = results["test_acc"]

      epochs = range(len(results["train_loss"]))

      plt.figure(figsize = (16,8))
      plt.subplot(1,2,1)
      plt.plot(epochs, loss, label = "train_loss")
      plt.plot(epochs, test_loss, label = "test_loss")
      plt.title("loss")
      plt.xlabel("epochs")
      plt.legend()

      plt.subplot(1,2,2)
      plt.plot(epochs, accuracy, label = "train_accuracy")
      plt.plot(epochs, test_accuracy, label = "test_accuracy")
      plt.title("accuracy")
      plt.xlabel("epochs")
      plt.legend()
      plt.show()
plot_loss_curves(model_0_results)

def show_random_predictions(model, dataset, class_names, n=9):
    model.eval()
    
    plt.figure(figsize=(4, 4))

    indices = random.sample(range(len(dataset)), n)

    with torch.inference_mode():
        for i, idx in enumerate(indices):
            img, true_label = dataset[idx]

            img_input = img.unsqueeze(0)  
            logits = model(img_input)
            pred_label = logits.argmax(dim=1).item()

            img_show = img.permute(1, 2, 0)

            correct = (pred_label == true_label)
            color = "green" if correct else "red"

            plt.subplot(3, 3, i + 1)
            plt.imshow(img_show)
            plt.axis("off")

            plt.title(
                f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}",
                color=color,
                fontsize=10
            )

    plt.tight_layout()
    plt.show()
show_random_predictions(model = model_0, dataset = test_data, class_names = class_names)

online_image_path = data_path / "baklava-online.jpg"
single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
single_image = single_image / 255

plt.imshow(single_image.permute(1,2,0))
plt.title(single_image.shape)
plt.show()

single_image_transform = transforms.Compose([
      transforms.Resize(size = (64,64)),
      transforms.Normalize(mean = [0.6,0.6,0.6], std = [0.3,0.3,0.3])
])
single_image = single_image_transform(single_image)
single_image = single_image.unsqueeze(dim = 0)

model_0.eval()
with torch.inference_mode():
      logits = model_0(single_image)
      probs = torch.softmax(logits, dim = 1)
      pred_idx = probs.argmax(dim = 1).item()
print("Predicted class: ", class_names[pred_idx])