import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import MulticlassAccuracy

df = pd.read_csv("WineQT.csv")

print(df.head())
print(df.columns)
df.columns = df.columns.str.replace(" ", "_")
print(df.columns)

print(df.nunique)

df = df.drop("Id", axis = 1)
X = df.drop("quality", axis = 1).values
y = df["quality"].values

print(X)
print(y)


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)

y_train = torch.tensor(y_train, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

class WineQualityClassifier(nn.Module):

      def __init__(self, input, hidden, output):
            super().__init__()

            self.layer_stack = nn.Sequential(
                  nn.Linear(in_features = input, out_features = hidden),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(in_features = hidden, out_features = hidden),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(in_features = hidden, out_features = output)
            )
      
      def forward(self, x):
            return self.layer_stack(x)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

model = WineQualityClassifier(
      input = 11,
      hidden = 32,
      output = 6
)
loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

accuracy = MulticlassAccuracy(num_classes = 6)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

epochs = 250

for epoch in range(epochs):
      
      model.train()
      logits = model(X_train)
      loss = loss_fn(logits, y_train)

      pred = torch.softmax(logits, dim=1).argmax(dim=1)
      acc = accuracy(pred, y_train).item() * 100

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())
      train_accuracies.append(acc)

      model.eval()
      with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
            test_acc = accuracy(y_test, test_pred)
      test_losses.append(test_loss.item())
      test_accuracies.append(test_acc)

      if epoch % 25 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:4f}, Accuracy: {acc:4f}, Test Loss: {test_loss:4f}, Test Accuracy: {test_acc:4f}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test Loss Curve")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train/Test Accuracy Curve")
plt.legend()
plt.show()