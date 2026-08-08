import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix

df = pd.read_csv("diabetes.csv")

print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["diabetes"].value_counts(normalize=True))

df = df.drop_duplicates()
df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.select_dtypes(include="object").columns:
      df_encoded[col] = le.fit_transform(df_encoded[col])

print(df_encoded.head())

X = df_encoded.drop("diabetes", axis = 1).values
y = df_encoded["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

class Diabetes(nn.Module):

      def __init__(self, input, hidden, output):
            super().__init__()

            self.layer = nn.Sequential(
                  nn.Linear(in_features = input, out_features = hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.2),
                  nn.Linear(in_features = hidden, out_features = hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.2),
                  nn.Linear(in_features = hidden, out_features = hidden),
                  nn.ReLU(),
                  nn.Dropout(p = 0.2),
                  nn.Linear(in_features = hidden, out_features = output)
            )

      def forward(self, x):
            return self.layer(x)

model = Diabetes(
      input = 8,
      hidden = 24,
      output = 1
)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)
accuracy = BinaryAccuracy()
confusion_matrix = BinaryConfusionMatrix()

torch.manual_seed(42)
epochs = 500

for epoch in range(epochs):
      model.train()

      logits = model(X_train)
      y_pred = torch.round(torch.sigmoid(logits))

      loss = loss_fn(logits, y_train)
      acc = accuracy(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model.eval()
      with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy(test_pred, y_test)

            if epoch % 20 == 0:
                  print(f"Epoch: {epoch} Loss: {loss:.4f} Accuracy: {acc:.4f} Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")