import torch
import numpy as np

random_array = np.random.randn(3,5)
print(random_array)

random_matrix = torch.rand(size = (3,5))
print(random_matrix)
print(random_matrix.ndim)
print(random_matrix.shape)

random_image_tensor = torch.rand(size = (224,224,3))
print(random_image_tensor)
print(random_image_tensor.ndim)

zeros = torch.zeros(size = (3,2))
print(zeros)

ones = torch.ones(size = (3,2))
print(ones)

vector_with_arange = torch.arange(start = 0, end = 10, step = 1)
print(vector_with_arange)
vector_with_arange = torch.arange(start = 0, end = 10, step = 3)
print(vector_with_arange)

similar_ones = torch.ones_like(input = vector_with_arange)
print(similar_ones)

x_tensor = torch.tensor([1,2,3])
print(x_tensor + 15, x_tensor * 3.14)

print(x_tensor.subtract(10))
print(x_tensor.multiply(10))
print(x_tensor.divide(5))

print(x_tensor * x_tensor, x_tensor + x_tensor)
#boyutlar aynı olmalı tensor matematik işlemi yapılması için





