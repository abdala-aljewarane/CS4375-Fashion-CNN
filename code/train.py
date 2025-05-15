# train.py - Model training script

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.lenet import LeNet
from models.vgg16 import VGG16
from models.resnet18 import ResNet18
from models.custom_cnn import CustomCNN
from utils.data_loader import DatasetLoader
from utils.trainer import Trainer
from utils.visualizer import DataVisualizer


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def get_model(model_name, num_channels, num_classes):
    """Get the specified model architecture"""
    models = {
        'lenet': LeNet,
        'vgg16': VGG16,
        'resnet18': ResNet18,
        'custom': CustomCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported")
    
    return models[model_name](num_channels=num_channels, num_classes=num_classes)


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', 
                          f'{args.dataset}_{args.model}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    data_loader = DatasetLoader(args.dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    dataset_params = data_loader.get_dataset_params()
    
    # Visualize dataset
    if not args.skip_visualization:
        print("\nAnalyzing dataset...")
        visualizer = DataVisualizer(train_loader.dataset,
                                  save_dir=os.path.join(exp_dir, 'visualizations'))
        visualizer.analyze_dataset()
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model,
                     num_channels=dataset_params['num_channels'],
                     num_classes=dataset_params['num_classes'])
    model = model.to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, device, train_loader, val_loader, test_loader,
                     learning_rate=args.learning_rate,
                     log_dir=os.path.join(exp_dir, 'logs'))
    
    # Train model
    print("\nStarting training...")
    trainer.train(args.epochs)
    
    print(f"\nTraining complete. Results saved in {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN models on various datasets')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
                        help='dataset to use')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'vgg16', 'resnet18', 'custom'],
                        help='model architecture to use')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    
    # Hardware parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of data loading workers')
    
    # Other parameters
    parser.add_argument('--skip-visualization', action='store_true', default=False,
                        help='skip dataset visualization')
    
    args = parser.parse_args()
    main(args)