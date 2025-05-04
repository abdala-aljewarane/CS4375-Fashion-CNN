# evaluate.py - Model evaluation script

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse

from models.lenet import LeNet5
from models.modified_lenet import ModifiedLeNet5
from models.vgg16 import VGG16
from utils.data_loader import get_fashion_mnist_loaders, get_deepfashion2_loaders
from utils.visualization import plot_confusion_matrix, plot_metrics_comparison
from utils.metrics import calculate_metrics


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate a trained model on the test dataset.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: List of class names
    
    Returns:
        dict: Evaluation metrics including accuracy, loss, and per-class metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    avg_loss = total_loss / len(test_loader)
    
    # Detailed classification report
    class_report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained CNN models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['lenet', 'modified_lenet', 'vgg16'],
                       help='Model architecture to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['fashion_mnist', 'deepfashion2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                       help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    if args.dataset == 'fashion_mnist':
        _, test_loader, class_names = get_fashion_mnist_loaders(
            batch_size=args.batch_size
        )
        num_classes = 10
        input_channels = 1
    else:  # deepfashion2
        _, test_loader, class_names = get_deepfashion2_loaders(
            batch_size=args.batch_size
        )
        num_classes = 10
        input_channels = 3
    
    # Load model
    if args.model == 'lenet':
        model = LeNet5(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'modified_lenet':
        model = ModifiedLeNet5(num_classes=num_classes, input_channels=input_channels)
    else:  # vgg16
        model = VGG16(num_classes=num_classes, input_channels=input_channels)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    print(f'\nEvaluation Results:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Loss: {metrics["loss"]:.4f}')
    
    # Save classification report
    report_path = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Accuracy: {metrics["accuracy"]:.4f}\n')
        f.write(f'Loss: {metrics["loss"]:.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(
            metrics["labels"], 
            metrics["predictions"], 
            target_names=class_names
        ))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        metrics["confusion_matrix"], 
        class_names,
        save_path=os.path.join(args.output_dir, f'{args.model}_{args.dataset}_confusion.png')
    )
    
    # Save metrics for comparison
    metrics_file = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_metrics.npy')
    np.save(metrics_file, metrics)
    
    print(f'\nResults saved to {args.output_dir}')


if __name__ == '__main__':
    main()