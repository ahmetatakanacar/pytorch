import torch
import numpy as np

# pytorch datatypes - tensors
# tensor = çok boyutlu matris
# pytorch tensor -> scalar=tensor, vector=tensor, matrix=tensor, tensor=tensor

print(torch.tensor(10))
print(torch.tensor([10,30]))

scalar = torch.tensor(10)
print(type(scalar))

print(scalar.dim()) # boyut
print(scalar.shape)
print(scalar.item())
print(scalar.dtype)

numpy_int = np.int32(10)
scalar_from_np = torch.tensor(numpy_int)
print(scalar_from_np)
print(scalar_from_np.dtype)

new_scalar = torch.tensor(30, dtype = torch.int32)
print(new_scalar)
print(new_scalar.dtype)

tensor_float = torch.tensor(3.14)
print(tensor_float)
print(tensor_float.dtype)

print("\n")

# vectors

vector = torch.tensor([3,4])
print(vector.dim())
print(vector.shape)

vector = torch.tensor([3,4,5])
print(vector.dim())
print(vector.shape)

# count # of [ -> dimension
# count elements -> shape

matrix = torch.tensor([[3,4],[5,6]])
print(matrix)
print(matrix.dim())
print(matrix.shape)

tensor_example = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(tensor_example)
print(tensor_example.dim())
print(tensor_example.shape)

float_16_tensor = torch.tensor([3.0,6.0,9.0], dtype = torch.float16)
print(float_16_tensor.dtype)