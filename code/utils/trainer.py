import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

class Trainer:
    """
    Handles model training, validation, and testing
    """
    def __init__(self, model, device, train_loader, val_loader, test_loader,
                 learning_rate=0.001, log_dir='runs'):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(log_dir, 'best_model.pth')

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.best_model_path)
        
        # Adjust learning rate
        self.scheduler.step(val_loss)
        
        return val_loss, val_acc

    def test(self):
        """Test the model and compute metrics"""
        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        self.writer.add_figure('Confusion Matrix', plt.gcf())
        
        # Log metrics
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'Test/{name}', value)
        
        return metrics

    def train(self, num_epochs):
        """Train the model for specified number of epochs"""
        for epoch in range(1, num_epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(epoch)
            
            # Print epoch results
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Final test
        print('\nTesting best model:')
        metrics = self.test()
        
        # Print final metrics
        print('\nFinal Test Metrics:')
        for name, value in metrics.items():
            print(f'{name}: {value:.2f}') 