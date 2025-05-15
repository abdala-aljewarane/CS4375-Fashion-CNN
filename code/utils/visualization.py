import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir, model_name, dataset_name):
    """
    Plot training and validation loss/accuracy curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'{model_name} on {dataset_name} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title(f'{model_name} on {dataset_name} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_{model_name}_history.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_dir, model_name, dataset_name, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_tsne(features, labels, class_names, save_dir, model_name, dataset_name, perplexity=30, n_iter=1000):
    """
    Create t-SNE visualization of the features
    """
    # Reduce feature dimensionality with t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot the 2D points
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=class_names[i], alpha=0.7)
    
    plt.title(f't-SNE visualization of {model_name} features on {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_{model_name}_tsne.png'), dpi=300)
    plt.close()

def extract_features(model, dataloader, device):
    """
    Extract features from the second-to-last layer of the model for t-SNE visualization
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For LeNet
            if hasattr(model, 'fc2'):
                x = model.conv1(inputs)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                x = model.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                x = model.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = model.fc1(x)
                x = torch.relu(x)
                x = model.fc2(x)
                x = torch.relu(x)
                feats = x.cpu().numpy()
            
            # For VGG16
            elif hasattr(model, 'features'):
                x = model.features(inputs)
                x = x.view(x.size(0), -1)
                x = model.classifier[0](x)
                x = torch.relu(x)
                feats = x.cpu().numpy()
            
            # For Custom CNN
            elif hasattr(model, 'block3'):
                x = model.initial(inputs)
                x = model.block1(x)
                x = model.block2(x)
                x = model.block3(x)
                x = model.avgpool(x)
                x = x.view(x.size(0), -1)
                feats = x.cpu().numpy()
            
            else:
                # Fallback - use the output before last layer
                # This needs to be adjusted based on the specific model architecture
                outputs = model(inputs)
                feats = outputs.cpu().numpy()
            
            features.append(feats)
            labels.append(targets.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels
