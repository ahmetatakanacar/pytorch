import torch

if torch.cuda.is_available():
      device = "cuda"
elif torch.backends.mps.is_available():
      device = "mps"
else:
      device = "cpu"

print(device)

tensor = torch.tensor([1,2,3])
print(tensor.device)

# manual

tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# context manager
with torch.device(device):
      tensor2 = torch.tensor([1,2,3])
      layer = torch.nn.Linear(20,30)
print(tensor2.device)
print(layer.weight.device)

# torch.set_default_device

print(torch.set_default_device(device))
tensor3 = torch.tensor([1,2,3])
layer2 = torch.nn.Linear(30,40)
print(tensor3.device)
print(layer2.weight.device)