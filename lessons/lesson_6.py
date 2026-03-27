import torch
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("06-study_hours_grades.csv")

print(df)
print(df.info())
print(df.describe())

y = torch.tensor(df["grade"].values)
X = torch.tensor(df["study_hours"].values)

print(X)
print(y)

train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#plt.scatter(X_train, y_train, c = "b", s = 5, label = "Training data")
#plt.scatter(X_test, y_test, c = "r", s = 5, label = "Test data")
#plt.show()

# pytorch ann

from torch import nn

class SimpleLinearRegressionModel(nn.Module):
      def __init__(self):
            super().__init__()

            self.weights = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)
            self.bias = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)
      
      def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias
      # x = data

torch.manual_seed(42)

model_0 =  SimpleLinearRegressionModel()
#model_0 = torch.compile(model_0)

print(model_0.parameters())
print(model_0.state_dict())

with torch.inference_mode():
      y_pred = model_0(X_test)


#plt.scatter(X_train, y_train, c = "b", s = 5, label = "Training data")
#plt.scatter(X_test, y_test, c = "r", s = 5, label = "Test data")
#plt.scatter(X_test, y_pred, c = "g", s = 5, label = "Pred data")
#plt.show()

loss_fn = nn.MSELoss()
#loss_fn = nn.L1Loss() -> MAE
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.001)
torch.manual_seed(42)
epochs = 120
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
      # train
      model_0.train()
      y_pred = model_0(X_train)
      loss = loss_fn(y_pred, y_train)

      #back propagation
      optimizer.zero_grad() # önceki adımdan kalan gradient değerlerini sıfırlar
      loss.backward() # model parametrelerine göre türevlerini hesaplar
      optimizer.step() # hesaplanan gradientleri kullanarak model ağırlıklarını günceller

      model_0.eval()

      with torch.inference_mode():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test) # loss_fn(test_pred, y_test.type(torch.float))

            if epoch % 5 == 0:
                  epoch_count.append(epoch)
                  train_loss_values.append(loss.detach().numpy())
                  test_loss_values.append(test_loss.detach().numpy())
                  print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")

#plt.plot(epoch_count, train_loss_values, label = "Train Loss")
#plt.plot(epoch_count, test_loss_values, label = "Test Loss")
#plt.ylabel("Loss")
#plt.xlabel("Epochs")
#plt.show()

print(model_0.state_dict())

model_0.eval()
with torch.inference_mode():
      y_preds = model_0(X_test)
print(X_test)
print(y_preds)

plt.scatter(X_train, y_train, c = "b", s = 5, label = "Training data")
plt.scatter(X_test, y_test, c = "r", s = 5, label = "Test data")
plt.scatter(X_test, y_preds, c = "g", s = 5, label = "Pred data")
plt.show()






