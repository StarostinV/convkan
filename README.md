# Convolutional KAN layer

## Implementation of the Convolutional Kolmogorov-Arnold Network layer in PyTorch.

The KAN implementation is taken from the https://github.com/Blealtan/efficient-kan/ repository.

## Usage

Training a simple model on MNIST (97% accuracy after the first epoch):

```python

import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

from convkan import ConvKAN, LayerNorm2D

# Define the model
model = nn.Sequential(
    ConvKAN(1, 32, padding=1, kernel_size=3, stride=2),
    LayerNorm2D(32),
    ConvKAN(32, 32, padding=1, kernel_size=3, stride=1),
    LayerNorm2D(32),
    ConvKAN(32, 10, padding=1, kernel_size=3, stride=2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
).cuda()

# Define transformations and download the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
model.train()
for i, (x, y) in enumerate(train_loader):
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        y_hat = model(x)
        _, predicted = torch.max(y_hat, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

```