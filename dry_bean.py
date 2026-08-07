import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

print(df.duplicated().sum())
df = df.drop_duplicates()

print(df.info())

X = df.drop("Class", axis = 1).values     
y = df["Class"].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long).unsqueeze(1)
y_test = torch.tensor(y_test, dtype = torch.long).unsqueeze(1)

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
