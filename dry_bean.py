import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print(df.duplicated().sum())
df = df.drop_duplicates()

print(df.info())

X = df.drop("Class", axis = 1).values     
y = df["Class"].values

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

class DryBean(nn.Module):

      def __init__(self):
            super().__init__()

            self.layer_stack = nn.Sequential(
                  nn.Linear(in_features = 16, out_features = 32),
                  nn.ReLU(),
                  nn.Linear(in_features = 32, out_features = 32),
                  nn.ReLU(),
                  nn.Linear(in_features = 32, out_features = 7)
            )

      def forward(self, x):
            return self.layer_stack(x)

model = DryBean()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.002)
accuracy = MulticlassAccuracy(num_classes = 7)
confusion_matrix = MulticlassConfusionMatrix(num_classes = 7)

epochs = 400

for epoch in range(epochs):

      model.train()

      logits = model(X_train)
      loss = loss_fn(logits, y_train)

      pred = torch.softmax(logits, dim = 1).argmax(dim = 1)
      acc = accuracy(pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model.eval()
      with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
            test_acc = accuracy(test_pred, y_test)

            if epoch % 20 == 0:
                        print(f"Epoch: {epoch}, Loss: {loss:4f}, Accuracy: {acc:4f}, Test Loss: {test_loss:4f}, Test Accuracy: {test_acc:4f}")

print(confusion_matrix(test_pred, y_test))
