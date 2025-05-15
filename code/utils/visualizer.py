import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
import umap
import os

class DataVisualizer:
    """
    Handles data visualization and dimensionality reduction
    """
    def __init__(self, dataset, save_dir='visualizations'):
        self.dataset = dataset
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a dataloader with batch_size=1 for visualization
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Get all data for dimensionality reduction
        self.data, self.labels = self._get_all_data()

    def _get_all_data(self):
        """Extract all data and labels from dataset"""
        all_data = []
        all_labels = []
        
        for data, label in self.dataloader:
            # Flatten the data
            flattened = data.squeeze().view(-1).numpy()
            all_data.append(flattened)
            all_labels.append(label.item())
            
        return np.array(all_data), np.array(all_labels)

    def show_sample_images(self, num_samples=25, classes=None):
        """Display sample images from the dataset"""
        plt.figure(figsize=(10, 10))
        
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            img, label = self.dataset[i]
            img = img.squeeze()
            
            if img.shape[0] == 3:  # RGB image
                img = img.permute(1, 2, 0)
            
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            plt.title(f'Class {label}' if classes is None else classes[label])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'sample_images.png'))
        plt.close()

    def visualize_pca(self, n_components=2):
        """Visualize data using PCA"""
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(self.data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                            c=self.labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('PCA Visualization')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.savefig(os.path.join(self.save_dir, 'pca_visualization.png'))
        plt.close()
        
        return pca.explained_variance_ratio_

    def visualize_lda(self, n_components=2):
        """Visualize data using LDA"""
        lda = LDA(n_components=n_components)
        data_lda = lda.fit_transform(self.data, self.labels)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_lda[:, 0], data_lda[:, 1], 
                            c=self.labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('LDA Visualization')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.savefig(os.path.join(self.save_dir, 'lda_visualization.png'))
        plt.close()

    def visualize_tsne(self, perplexity=30):
        """Visualize data using t-SNE"""
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        data_tsne = tsne.fit_transform(self.data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], 
                            c=self.labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(os.path.join(self.save_dir, 'tsne_visualization.png'))
        plt.close()

    def visualize_umap(self, n_neighbors=15, min_dist=0.1):
        """Visualize data using UMAP"""
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        data_umap = reducer.fit_transform(self.data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_umap[:, 0], data_umap[:, 1], 
                            c=self.labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(os.path.join(self.save_dir, 'umap_visualization.png'))
        plt.close()

    def plot_class_distribution(self, classes=None):
        """Plot the distribution of classes in the dataset"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(unique_labels, counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        
        if classes is not None:
            plt.xticks(unique_labels, classes, rotation=45)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'class_distribution.png'))
        plt.close()

    def analyze_dataset(self, classes=None):
        """Perform complete dataset analysis"""
        print("Performing dataset analysis...")
        
        # Show sample images
        print("Generating sample images visualization...")
        self.show_sample_images(classes=classes)
        
        # Plot class distribution
        print("Generating class distribution plot...")
        self.plot_class_distribution(classes=classes)
        
        # Dimensionality reduction visualizations
        print("Generating PCA visualization...")
        var_ratio = self.visualize_pca()
        print(f"PCA explained variance ratio: {var_ratio}")
        
        print("Generating LDA visualization...")
        self.visualize_lda()
        
        print("Generating t-SNE visualization...")
        self.visualize_tsne()
        
        print("Generating UMAP visualization...")
        self.visualize_umap()
        
        print("Dataset analysis complete. Visualizations saved in:", self.save_dir) 