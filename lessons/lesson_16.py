import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms, datasets
from torchinfo import summary
from torch.utils.data import DataLoader, Subset

from torch.utils.data import random_split

if torch.cuda.is_available():
      device = "cuda"
elif torch.backends.mps.is_available():
      device = "mps"
else:
      device = "cpu"

IMG_SIZE = 224
seed = 42

def to_rgb(img):
      return img.convert("RGB")

train_tf = transforms.Compose([
      transforms.Lambda(to_rgb),
      transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
      transforms.RandomHorizontalFlip(p=0.4),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random")
])

test_tf = transforms.Compose([
      transforms.Lambda(to_rgb),
      transforms.Resize(256),
      transforms.CenterCrop(IMG_SIZE),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = "./data"
test_ratio = 0.2

ds_train_full = datasets.Caltech101(root=root, download=False, transform=train_tf)
ds_test_full = datasets.Caltech101(root=root, download=False, transform=test_tf)

n_total = len(ds_train_full)
n_test = int(n_total * test_ratio)
n_train = n_total - n_test

print(n_train, n_test, n_total)

g = torch.Generator().manual_seed(seed)
perm = torch.randperm(n_total, generator=g).tolist()

train_idx = perm[:n_train]
test_idx = perm[n_train:]
#print(len(train_idx), len(test_idx))

train_ds = Subset(ds_train_full, train_idx)
test_ds = Subset(ds_test_full, test_idx)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

img, label = ds_train_full[6000]
class_names = ds_train_full.categories

height = 224
width = 224
color_channels = 3
patch_size = 16

number_of_patches = int((height * width) / (patch_size **2))
print(number_of_patches)

embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)
#print(embedding_layer_input_shape)
#print(embedding_layer_output_shape)

image_batch, label_batch = train_dataloader.dataset[3]
image, label = image_batch, label_batch

conv2d = nn.Conv2d(in_channels=3,
                   out_channels= 768,
                   kernel_size= patch_size,
                   stride=patch_size,
                   padding=0)

image_out_of_conv = conv2d(image.unsqueeze(0))
#print(image_out_of_conv.shape)

flatten = nn.Flatten(start_dim=2, end_dim=3)
image_out_of_conv_flattened = flatten(image_out_of_conv)
#print(image_out_of_conv_flattened.shape)

class PatchEmbedding(nn.Module):
      
      def __init__(self,
                   in_channels: int = 3,
                   embedding_dim: int = 768,
                   patch_size: int = 16):
            super().__init__()

            self.patcher = nn.Conv2d(in_channels=in_channels,
                                     out_channels=embedding_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0)
            self.flatten = nn.Flatten(start_dim=2, end_dim=3)
      
      def forward(self, x):
            x_patcher = self.patcher(x)
            x_flattened = self.flatten(x_patcher)
            return x_flattened.permute(0,2,1)

patch_embedding = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
patch_embedded_image = patch_embedding(image.unsqueeze(0))
print(patch_embedded_image.shape, image.unsqueeze(0).shape)

random_input_image = (1,3,224,224)

summary(PatchEmbedding(), input_size = random_input_image, col_names = ["input_size", "output_size", "num_params", "trainable"])