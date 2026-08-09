import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassConfusionMatrix

df = pd.read_csv("glass.csv")

print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Type"].unique())
print(df["Type"].value_counts())

df = df.drop_duplicates()
print(df.duplicated().sum())

X = df.drop("Type", axis = 1)
y = df["Type"]

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

class Glass(nn.Module):
      def __init__(self, input, hidden, output):
            super().__init__()

            self.layer = nn.Sequential(
                  nn.Linear(in_features=input, out_features=hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.3),
                  nn.Linear(in_features=hidden, out_features=hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.3),
                  nn.Linear(in_features=hidden, out_features=output)
            )

      def forward(self, x):
            return self.layer(x)

model = Glass(
      input = 9,
      hidden = 14,
      output = 6
)
class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()
loss_fn = nn.CrossEntropyLoss(weight = weights)
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01, weight_decay=1e-4)
accuracy = MulticlassAccuracy(num_classes = 6)
recall = MulticlassRecall(num_classes = 6)
confusion_matrix = MulticlassConfusionMatrix(num_classes=6)

torch.manual_seed(42)
epochs = 150

for epoch in range(epochs):
      model.train()

      logits = model(X_train)
      loss = loss_fn(logits, y_train)

      y_pred = torch.softmax(logits, dim = 1).argmax(dim = 1)
      acc = accuracy(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model.eval()
      with torch.inference_mode():
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
            test_acc = accuracy(test_pred, y_test)
            test_recall = recall(test_pred, y_test)

            if epoch % 5 == 0:
                  print(f"Epoch: {epoch} Loss: {loss:.4f} Accuracy: {acc:.4f} Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f} Test Recall: {test_recall:.4f}")

print(confusion_matrix(test_pred, y_test))