import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("09-iris.csv")

print(df.head())
print(df.info())

print(df["Species"].value_counts())


sns.scatterplot(x = df["PetalLengthCm"], y = df["PetalWidthCm"], hue = df["Species"])
#plt.show()

df = df.drop("Id", axis = 1)

X = df.drop("Species", axis = 1).values
y = df["Species"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)

y_train = torch.tensor(y_train, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

from torch import nn

class IrisClassifier(nn.Module):
      
      def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(in_features = 4, out_features = 16)
            self.layer2 = nn.Linear(in_features = 16, out_features = 16)
            self.layer3 = nn.Linear(in_features = 16, out_features = 3)

            self.relu = nn.ReLU()

      def forward(self, x):
            return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

#class IrisClassifierTwo(nn.Module):
      #def __init__(self):
            #super().__init_()

            #self.linear_layer_stack = nn.Sequential(
                  #nn.Linear(4, 16),
                  #nn.ReLU(),
                  #nn.Linear(16, 16),
                  #nn.ReLU(),
                  #nn.Linear(16, 3)
            #)
            

      #def forward(self, x):
           # return self.linear_layer_stack(x)
           
model = IrisClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

def calculate_accuracy(y_true, y_pred):
      correct = torch.eq(y_true, y_pred).sum().item()
      acc = (correct / len(y_pred)) * 100
      return acc

print(model(X_test)[:5])

y_logits = model(X_test)
y_pred_probs = torch.softmax(y_logits, dim = 1)

print(y_logits[:5])
print(y_pred_probs[:5])

print(torch.softmax(y_logits, dim = 1).argmax(dim = 1))

epochs = 200

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
      model.train()

      logits = model(X_train)
      loss = loss_fn(logits, y_train)

      pred = torch.softmax(logits, dim = 1).argmax(dim = 1)
      acc = calculate_accuracy(y_train, pred)

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
            test_acc = calculate_accuracy(y_test, test_pred)
      test_losses.append(test_loss.item())
      test_accuracies.append(test_acc)

      if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:4f}, Accuracy: {acc:4f}, Test Loss: {test_loss:4f}, Test Accuracy: {test_acc:4f}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test Loss Curve")
plt.legend()
#plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train/Test Accuracy Curve")
plt.legend()
#plt.show()

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_tensor = torch.tensor(new_sample, dtype = torch.float32)
print(new_sample_tensor, new_sample_tensor.shape)

model.eval()
with torch.inference_mode():
      logits = model(new_sample_tensor)
      probs = torch.softmax(logits, dim = 1)
      predicted_class = torch.argmax(probs, dim = 1)

predicted_label = le.inverse_transform([predicted_class.item()])[0]
print("Predicted Class Index:", predicted_class)
print("Predicted Species:", predicted_label)

from torchmetrics.classification import MulticlassAccuracy

accuracy = MulticlassAccuracy(num_classes = 3)

epochs = 200
model2 = IrisClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr = 0.01 )

for epoch in range(epochs):
      
      model2.train()
      logits = model2(X_train)
      loss = loss_fn(logits, y_train)

      pred = torch.softmax(logits, dim = 1).argmax(dim = 1)
      acc = accuracy(pred, y_train).item() * 100

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model2.eval()
      with torch.inference_mode():
            test_logits = model2(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
            test_acc = accuracy(test_pred, y_test).item() * 100

      if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

from torchmetrics.classification import MulticlassConfusionMatrix

cm = MulticlassConfusionMatrix(num_classes = 3)
matrix = cm(test_pred, y_test)
print(matrix)






















