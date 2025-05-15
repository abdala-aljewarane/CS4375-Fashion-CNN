import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Debug GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

from models.lenet import LeNet
from models.vgg16 import VGG16
from models.resnet18 import ResNet18
from models.custom_cnn import CustomCNN
from utils.data_loader import DatasetLoader
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_tsne, extract_features

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

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def evaluate_model(model, test_loader, criterion, device, exp_dir, model_name, dataset_name, class_names):
    """Evaluate model on test set and generate metrics and visualizations"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted')
    
    # Print detailed report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Save report to file
    with open(os.path.join(exp_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, exp_dir, model_name, dataset_name, normalize=False)
    plot_confusion_matrix(cm, class_names, exp_dir, model_name, dataset_name, normalize=True, 
                         title='Normalized Confusion Matrix')
    
    # Extract features and generate t-SNE visualization
    # Use a subset of test data (max 1000 samples) for t-SNE to avoid excessive computation
    try:
        subset_size = min(1000, len(test_loader.dataset))
        subset_indices = np.random.choice(len(test_loader.dataset), subset_size, replace=False)
        subset_dataset = torch.utils.data.Subset(test_loader.dataset, subset_indices)
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=64, shuffle=False)
        
        features, labels = extract_features(model, subset_loader, device)
        plot_tsne(features, labels, class_names, exp_dir, model_name, dataset_name)
    except Exception as e:
        print(f"Warning: Could not generate t-SNE visualization. Error: {e}")
    
    return test_loss, test_acc, precision, recall, f1

def main(args):
    # Create experiment directory
    if args.results_dir:
        exp_dir = args.results_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"experiments/{args.dataset}_{args.model}_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Override certain parameters for deep_fashion dataset
    if args.dataset == 'deep_fashion':
        args.epochs = args.deep_fashion_epochs
        args.lr = args.deep_fashion_lr
        print(f"Using dataset-specific settings: epochs={args.epochs}, lr={args.lr}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    data_loader = DatasetLoader(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    params = data_loader.get_dataset_params()
    
    # Get class names for the dataset
    class_names = []
    if args.dataset == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif args.dataset == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        # Default class names based on indices
        class_names = [str(i) for i in range(params['num_classes'])]
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, params['num_channels'], params['num_classes'])
    
    # Move model to device first
    model = model.to(device)
    
    # Load pretrained model if specified
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        
        # If we have a state_dict inside a dictionary (newer format)
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            # Older format where the file contains just the state dict
            pretrained_dict = checkpoint
            
        # If target dataset has different number of classes than source dataset,
        # we need to handle the last layer(s) differently
        model_dict = model.state_dict()
        
        # Filter out final layer if number of classes is different
        if args.model == 'lenet':
            # For LeNet, we should keep all layers except fc3 if num_classes differs
            if 'fc3.weight' in pretrained_dict and pretrained_dict['fc3.weight'].size(0) != params['num_classes']:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if 'fc3' not in k}
        elif args.model == 'vgg16' or args.model == 'custom':
            # For VGG16 and custom, final layer is in classifier
            classifier_key = 'classifier.6.weight' if args.model == 'vgg16' else 'classifier.3.weight'
            if classifier_key in pretrained_dict and pretrained_dict[classifier_key].size(0) != params['num_classes']:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if not (k.startswith('classifier.6') if args.model == 'vgg16' 
                                       else k.startswith('classifier.3'))}
        
        # Update model with pretrained weights where they match
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print("Pretrained weights loaded successfully")
    
    # Print model details
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    else:
        scheduler = None
    
    # Lists to store metrics for visualization
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Train model
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update scheduler if using plateau
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()
        
        # Print epoch results
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Save training history plot
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         exp_dir, args.model, args.dataset)
    
    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    print(f"\nLoaded best model from epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc, precision, recall, f1 = evaluate_model(
        model, test_loader, criterion, device, exp_dir, args.model, args.dataset, class_names)
    
    # Save final metrics
    with open(os.path.join(exp_dir, 'final_metrics.txt'), 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1 Score (weighted): {f1:.4f}\n")
    
    # Save hyperparameters
    with open(os.path.join(exp_dir, 'hyperparameters.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Device: {device}\n")
    
    print(f"\nTraining complete. Results saved in {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN models on various datasets')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10'],
                        help='dataset to use')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'vgg16', 'custom'],
                        help='model architecture to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw'],
                        help='optimizer to use (default: adam)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine', 'plateau'],
                        help='learning rate scheduler (default: cosine)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for regularization (default: 1e-4)')
    
    # Hardware parameters
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of data loading workers')
    
    # Quick test parameters
    parser.add_argument('--subset-fraction', type=float, default=1.0,
                        help='Fraction of the dataset to use (0 < f ≤ 1). Use a small value, e.g. 0.1, for rapid testing')
    
    # Dataset-specific parameters
    parser.add_argument('--deep_fashion_epochs', type=int, default=20,
                        help='number of epochs for deep_fashion dataset (default: 20)')
    parser.add_argument('--deep_fashion_lr', type=float, default=0.0005,
                        help='learning rate for deep_fashion dataset (default: 0.0005)')
    
    # Experiment settings
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='path to pretrained model to use for transfer learning')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='directory to save results')
    
    args = parser.parse_args()
    
    main(args) 