import torch
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("06-study_hours_grades.csv")

X = torch.tensor(df["study_hours"].values, dtype = torch.float32).unsqueeze(1)
y = torch.tensor(df["grade"].values, dtype = torch.float32).unsqueeze(1)

print(X.shape, X.ndim)

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(X_test))

import torch.nn as nn

class LinearRegressionModel(nn.Module):

      def __init__(self):
            super().__init__()

            self.linear_layer = nn.Linear(in_features = 1, out_features = 1)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_layer(x)  

torch.manual_seed(42)
model = LinearRegressionModel()
print(model)
print(model.state_dict())

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.001)
epochs = 120

for epoch in range(epochs):

      model.train()
      y_pred = model(X_train)

      loss = loss_fn(y_pred, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model.eval()
      with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

            if epoch % 5 == 0:
                  print(f"Epoch {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")

print(model.state_dict())

model.eval()
with torch.inference_mode():
      y_preds = model(X_test)

plt.scatter(X_train, y_train, c = "b", s = 5, label = "Train data")
plt.scatter(X_test, y_test, c = "y", s = 5, label = "Test data")
plt.scatter(X_test, y_preds, c = "r", s = 5, label = "Prediction")
plt.show()


















