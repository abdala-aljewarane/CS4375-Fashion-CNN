import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm

from models.lenet import LeNet
from utils.data_loader import DatasetLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
dataset_name = 'fashion_mnist'
batch_size = 32
num_epochs = 1
num_iterations = 5  # Just do a few iterations

# Load a small subset of the dataset
print(f"\nLoading {dataset_name} dataset...")
data_loader = DatasetLoader(dataset_name, batch_size=batch_size, num_workers=0)
train_dataset, val_dataset, test_dataset = data_loader.get_dataset()
dataset_params = data_loader.get_dataset_params()

# Take only a small subset of the data (100 samples)
subset_indices = list(range(100))
train_subset = Subset(train_dataset, subset_indices)
val_subset = Subset(val_dataset, subset_indices[:50])

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Create model
print(f"\nCreating LeNet model...")
model = LeNet(
    num_channels=dataset_params['num_channels'],
    num_classes=dataset_params['num_classes']
)
model = model.to(device)

# Print model summary
print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simple training loop
print("\nStarting training...")
model.train()

for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Only train for a few iterations
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iterations:
            break
            
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        print(f'Batch {i+1}/{num_iterations}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

print("\nQuick test completed successfully!") 