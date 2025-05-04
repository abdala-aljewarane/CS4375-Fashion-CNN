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

from models.lenet import LeNet5
from models.modified_lenet import ModifiedLeNet5
from models.vgg16 import VGG16
from utils.data_loader import get_fashion_mnist_loaders, get_deepfashion2_loaders
from utils.metrics import calculate_metrics


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


def train_model(args):
    """Main training function."""
    # Create experiment directories
    experiment_name = f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join('experiments', 'runs', experiment_name)
    checkpoint_dir = os.path.join('experiments', 'checkpoints', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=experiment_dir)
    
    # Load data
    if args.dataset == 'fashion_mnist':
        train_loader, val_loader, class_names = get_fashion_mnist_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        num_classes = 10
        input_channels = 1
    else:  # deepfashion2
        train_loader, val_loader, class_names = get_deepfashion2_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        num_classes = 10
        input_channels = 3
    
    # Create model
    if args.model == 'lenet':
        model = LeNet5(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'modified_lenet':
        model = ModifiedLeNet5(num_classes=num_classes, input_channels=input_channels)
    else:  # vgg16
        model = VGG16(num_classes=num_classes, input_channels=input_channels, pretrained=args.pretrained)
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 30)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
        writer.add_scalar('Train/Accuracy_epoch', train_acc, epoch)
        writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/Accuracy_epoch', val_acc, epoch)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'args': args
        }
        
        # Save regular checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'New best model saved! Val Acc: {val_acc:.4f}')
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(checkpoint, final_checkpoint_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    history_path = os.path.join(experiment_dir, 'training_history.npy')
    np.save(history_path, history)
    
    writer.close()
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.4f}')
    print(f'Results saved to: {experiment_dir}')
    print(f'Checkpoints saved to: {checkpoint_dir}')


def main():
    parser = argparse.ArgumentParser(description='Train CNN models on fashion datasets')
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True, 
                       choices=['lenet', 'modified_lenet', 'vgg16'],
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['fashion_mnist', 'deepfashion2'],
                       help='Dataset to train on')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights for VGG16')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--lr_step_size', type=int, default=10,
                       help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Learning rate scheduler gamma')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()