import torch

tensor = torch.arange(0, 50, 5)
print(tensor)

print(tensor.median())
tensor = tensor.type(torch.float)
print(tensor.mean())

#argmax, argmin
print(tensor.argmax()) # max değerin index
print(tensor.argmin())

# manipulating tensor: reshaping, stacking, squeezing, unsqueezing

x = torch.arange(1,10,1)
print(x)
print(x.shape, x.ndim)

x_reshaped = x.reshape((9,1))
print(x_reshaped)
print(x_reshaped.shape, x_reshaped.ndim)

# view -> contigious tensor

x_reshaped[0] = 30
print(x_reshaped)
print(x)

x_view = x.view(9,1)

x_view[0] = 10
print(x_view)
print(x)


# stacking

y = torch.tensor([[1,2,3],
                  [4,5,6]])
print(y)

print(torch.stack([y,y,y,y], dim = 0))
print(torch.stack([y,y,y,y], dim = 1))

# squeeze & unsqueeze

print(y)
print(y.squeeze())

y = torch.tensor([[1,2,3]])
print(y.shape, y.ndim)

print(y.squeeze(), y.squeeze().ndim) # gereksiz boyutları kaldırır

y = torch.tensor([[1,2,3],[4,5,6]])
print(y)
print(y.unsqueeze(dim = 0)) # boyut ekler

z = y.unsqueeze(dim = 0)
print(z)
print(z.ndim, z.shape)

# permute

x = torch.rand(size = (224,224,3))
print(x.shape)

x_permuted = x.permute(2,0,1) # axis  0 -> 1, 1 -> 2, 2 -> 0
print(x_permuted.shape)

# indexing & slicing

a = torch.arange(1,10,1)
a = a.reshape(1,3,3)
print(a.shape, a.ndim)
print(a[0][0][0])
print(a[:,0,0])
print(a[:,1,0])
print(a[:,:,0])
print(a[:,0,:])

# random seed

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor = torch.rand(3,4)
print(random_tensor)