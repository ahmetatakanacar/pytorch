import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("08-email_classification_svm.csv")

print(df.head())
print(df.info())
print(df["email_type"].unique())

sns.scatterplot(x = df["subject_formality_score"], y = df["sender_relationship_score"], hue = df["email_type"])
#plt.show()

X = df[["subject_formality_score" , "sender_relationship_score"]].values
y = df["email_type"].values
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(len(X_train),len(X_test))

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype = torch.float32).unsqueeze(1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from torch import nn

class ClassificationModel(nn.Module):
      def __init__(self):
            super().__init__()

            self.layer_1 = nn.Linear(in_features = 2, out_features = 5)
            self.layer_2 = nn.Linear(in_features = 5, out_features = 1)

      def forward(self, x):
            return self.layer_2(self.layer_1(x))

model_0 = ClassificationModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)

def calculate_accuracy(y_test, y_pred):
     correct = torch.eq(y_test, y_pred).sum().item()
     accuracy = correct / len(y_pred) * 100
     return accuracy

y_logits = model_0(X_test)[:5]
print(y_logits)
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
y_preds = torch.round(y_pred_probs)
print(y_preds)

print(calculate_accuracy(y_test[:5], y_preds))

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):

      model_0.train()

      y_logits = model_0(X_train)
      y_pred = torch.round(torch.sigmoid(y_logits))

      loss = loss_fn(y_logits, y_train)
      acc = calculate_accuracy(y_test = y_train, y_pred = y_pred)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      model_0.eval()
      with torch.inference_mode():
            test_logits = model_0(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)
            test_acc = calculate_accuracy(y_test = y_test, y_pred = test_pred)

            if epoch % 5 == 0:
                  print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}, Test Loss: {test_loss}, Test Acc: {test_acc}")



















