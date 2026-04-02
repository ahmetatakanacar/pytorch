import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530])
])

train_data = datasets.FashionMNIST(
      root = "data",
      train = True,
      download = True,
      transform = transform,
      target_transform = None
)

test_data = datasets.FashionMNIST(
      root = "data",
      train = False,
      download = True,
      transform = transform
)

print(train_data)
print(test_data)

image, label = train_data[0]
print(image)
print(label)

class_names = train_data.classes
print(class_names)

plt.figure(figsize=(3,3))
plt.imshow(image.squeeze(), cmap="gray") 
plt.title(f"{class_names[label]}")
#plt.show()


BATCH_SIZE = 32

train_dataloader = DataLoader(
      train_data,
      batch_size = BATCH_SIZE,
      shuffle = True
)
test_dataloader = DataLoader(
      test_data,
      batch_size = BATCH_SIZE,
      shuffle = False
)
print(len(train_dataloader), len(test_dataloader))
print(train_dataloader.dataset[0][0].shape)

flatten_model = nn.Flatten()
first_data = train_dataloader.dataset[0][0]
flattened_data = flatten_model(first_data)

print(first_data.shape, flattened_data.shape)

print(f"train values: {len(train_data)}")
print(f"test values: {len(test_data)}")
print("-"*30)
print(f"train dataload values: {len(train_dataloader)}")
print(f"test dataload values: {len(test_dataloader)}")

class FashionClassifier(nn.Module):
      def __init__(self, input_shape, hidden_units, output_shape):
            super().__init__()
            self.layer_stack = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features=input_shape, out_features= hidden_units),
                  nn.Linear(in_features=hidden_units, out_features= output_shape)
            )
      def forward(self, x):
            return self.layer_stack(x)
      
# MODEL 1
torch.manual_seed(42)
model = FashionClassifier(
      input_shape = 784,
      hidden_units = 32,
      output_shape = 10
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

def calculate_accuracy(y_true, y_pred):
      correct = torch.eq(y_true, y_pred).sum().item()
      acc = (correct / len(y_pred)) * 100
      return acc

epochs = 5
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                  print(f"Batch number: {batch}")
      
      train_loss /= len(train_dataloader)

      test_loss = 0
      test_acc = 0

      model.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.5f}")

def evaluate_model_performance(model, data_loader, loss_fn, accuracy_function):
      loss = 0
      acc = 0

      model.eval()
      with torch.inference_mode():
            for X, y in data_loader:
                  y_pred = model(X)
                  loss += loss_fn(y_pred, y)
                  acc += accuracy_function(y_true = y, y_pred = y_pred.argmax(dim = 1))
            
            loss /= len(data_loader)
            acc /= len(data_loader)
      
      return {"model_name": model.__class__.__name__,
              "model_loss": loss.item(),
              "model_accuracy": acc
              }

model_result = evaluate_model_performance(model = model, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print(model_result)

# MODEL 2
class FashionClassifierNonLinear(nn.Module):

      def __init__(self, input_shape, hidden_units, output_shape):
            super().__init__()
            self.layer_stack = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features=input_shape, out_features=hidden_units),
                  nn.ReLU(),
                  nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

      def forward(self, x):
            return self.layer_stack(x)

model_2 = FashionClassifierNonLinear(
      input_shape = 784,
      hidden_units = 32,
      output_shape = 10
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(), lr = 0.001)

epochs = 5
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model_2.train()
            y_pred = model_2(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                  print(f"Batch number: {batch}")
      
      train_loss /= len(train_dataloader)

      test_loss = 0
      test_acc = 0

      model_2.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model_2(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

model_2_result = evaluate_model_performance(model = model_2, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print(model_2_result)

# MODEL 3
class FashionClassifierCNN(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))

model_3 = FashionClassifierCNN(
      input_shape = 1,
      hidden_units = 32,
      output_shape = 10
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_3.parameters(), lr = 0.001)

epochs = 5
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model_3.train()
            y_pred = model_3(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                  print(f"Batch number: {batch}")
      
      train_loss /= len(train_dataloader)

      test_loss = 0
      test_acc = 0

      model_3.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model_3(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

model_3_result = evaluate_model_performance(model = model_3, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print(model_3_result)

# MODEL 4 WITH SGD OPTIMIZER
optimizer_sgd = torch.optim.SGD(params = model_3.parameters(), lr = 0.001, momentum=0.9)

epochs = 5
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model_3.train()
            y_pred = model_3(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer_sgd.step()

            if batch % 500 == 0:
                  print(f"Batch number: {batch}")
      
      train_loss /= len(train_dataloader)

      test_loss = 0
      test_acc = 0

      model_3.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model_3(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

model_3_result = evaluate_model_performance(model = model_3, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print("with sgd optimizer:", model_3_result)

y_preds = []
y_true = []

model_3.eval() 
with torch.inference_mode():
    for X, y in test_dataloader:
        y_logit = model_3(X)
        y_pred = torch.argmax(y_logit, dim=1)
      
        y_preds.append(y_pred)
        y_true.append(y)

y_preds = torch.cat(y_preds).cpu()
y_true = torch.cat(y_true).cpu()

cm = confusion_matrix(y_true.numpy(), y_preds.numpy())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Modelin Tahmini", fontsize=12, fontweight="bold")
plt.ylabel("Gerçek Değer", fontsize=12, fontweight="bold")
plt.title("FashionMNIST Karmaşıklık Matrisi (CNN - Model 3)", fontsize=14)
plt.xticks(rotation=45)
plt.show()

def make_predictions(model: torch.nn.Module, data: list):
    
    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:

            sample = sample.unsqueeze(0)

            pred_logit = model(sample)

            prob = torch.softmax(pred_logit, dim=1) 

            pred_probs.append(prob.squeeze(0))  

    return torch.stack(pred_probs)

import random
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
            plt.imshow(img_show, cmap = "gray")
            plt.axis("off")

            plt.title(
                f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}",
                color=color,
                fontsize=10
            )

    plt.tight_layout()
    plt.show()

show_random_predictions(model_3, test_data, class_names)