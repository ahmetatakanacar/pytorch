import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.CIFAR10(
      root = "data",
      train = True,
      download = True,
      transform = ToTensor(),
      target_transform = None,

)

test_data = datasets.CIFAR10(
      root = "data",
      train = False,
      download = True,
      transform = ToTensor()
)

print(train_data)

image , label = train_data[1]
print(image)
print(label)

print(image.shape)
print(len(train_data),len(test_data))

class_names = train_data.classes
print(class_names)

image = image.permute(1,2,0)
print(image.shape)
plt.figure(figsize=(2,2))
plt.imshow(image)
plt.title(f" first {class_names[label]}")
#plt.show()


from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                 std=[0.2470, 0.2435, 0.2616])
])

train_data = datasets.CIFAR10(
      root = "data",
      train = True,
      download = True,
      transform = transform,
      target_transform = None,

)

test_data = datasets.CIFAR10(
      root = "data",
      train = False,
      download = True,
      transform = transform
)

image,label = train_data[1]
print(image)
image = image.permute(1,2,0)
plt.figure(figsize=(2,2))
plt.imshow(image)
plt.title(f"second {class_names[label]}")
#plt.show()

from torch.utils.data import DataLoader

BATCH_SIZE = 32 #32 & 128 popular

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

class CIFAR10Classifier(nn.Module):

      def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.layer_stack = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features=input_shape, out_features=hidden_units),
                  nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

      def forward(self, x):
            return self.layer_stack(x)

torch.manual_seed(42)

model_0 = CIFAR10Classifier(
      input_shape=3072,
      hidden_units=32,
      output_shape=len(class_names)
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_0.parameters(), lr=0.01)

def calculate_accuracy(y_true, y_pred):
      correct = torch.eq(y_true, y_pred).sum().item()
      acc = (correct / len(y_pred)) * 100
      return acc



epochs = 2
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model_0.train()
            y_pred = model_0(X)
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

      model_0.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model_0(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

def evaluate_model_performance(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                               loss_fn: torch.nn.Module, accuracy_function):
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

model_0_result = evaluate_model_performance(model = model_0, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print(model_0_result)

class CIFAR10ClassifierNonLinear(nn.Module):

      def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            super().__init__()
            self.layer_stack = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features=input_shape, out_features=hidden_units),
                  nn.ReLU(),
                  nn.Linear(in_features=hidden_units, out_features=output_shape)
            )

      def forward(self, x):
            return self.layer_stack(x)

torch.manual_seed(42)
model_1 = CIFAR10ClassifierNonLinear(input_shape=3072,
                                     hidden_units=32,
                                     output_shape=len(class_names))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_1.parameters(), lr=0.01)

epochs = 2
for epoch in range(epochs):

      train_loss = 0

      for batch, (X,y) in enumerate(train_dataloader):
            model_1.train()
            y_pred = model_1(X)
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

      model_1.eval()
      with torch.inference_mode():
            for X, y in test_dataloader:
                  test_pred = model_1(X)
                  test_loss += loss_fn(test_pred, y)
                  test_acc += calculate_accuracy(y_true = y, y_pred = test_pred.argmax(dim = 1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
      print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

model_1_result = evaluate_model_performance(model = model_1, data_loader= test_dataloader,
                                            loss_fn = loss_fn, accuracy_function=calculate_accuracy)

print(model_1_result)

class CIFAR10ClassifierCNN(nn.Module):

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
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
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
            nn.Linear(in_features=hidden_units*8*8, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))

torch.manual_seed(42)
model_2 = CIFAR10ClassifierCNN(input_shape = 3,
                               hidden_units = 32,
                               output_shape = len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_2.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):

    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        model_2.train() 
        y_pred = model_2(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    
    train_loss /= len(train_dataloader)
    

    test_loss, test_acc = 0, 0 
    model_2.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model_2(X)
           
            test_loss += loss_fn(test_pred, y) 

            test_acc += calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

model_2_results = evaluate_model_performance(model=model_2, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_function=calculate_accuracy)
print(model_2_results)














