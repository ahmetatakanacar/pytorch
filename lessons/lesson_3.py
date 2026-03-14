import torch
import numpy as np

tensor = torch.tensor([1,2,3])
print(tensor * tensor)

matrix = torch.tensor(
      [[1,2,3,],
      [4,5,6],
      [7,8,9]]
)
print(matrix * matrix)

print(matrix @ matrix) # matrix çarpımı
print(torch.matmul(matrix, matrix)) # matrix çarpımı




