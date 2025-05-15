import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import os
import glob
from PIL import Image

class KaggleClothingDataset(Dataset):
    """Custom dataset for loading Kaggle clothing dataset"""
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether to load training or test set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Determine which directory to use
        self.data_dir = os.path.join(root_dir, "train" if train else "test")
        
        # Get classes from directory names
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths and their labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_path in glob.glob(os.path.join(class_dir, "*.jpg")):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DatasetLoader:
    """
    Handles loading and preprocessing of different datasets
    """
    def __init__(self, dataset_name, batch_size=128, num_workers=2, subset_fraction: float = 1.0):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Keep only a fraction of the data if requested (useful for quick smoke-tests)
        # A value of 1.0 means use the full dataset (default behaviour)
        if not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be in the interval (0, 1].")
        self.subset_fraction = subset_fraction
        
        # Set dataset specific parameters
        self.dataset_params = {
            'mnist': {'num_channels': 1, 'num_classes': 10, 'size': 28},
            'fashion_mnist': {'num_channels': 1, 'num_classes': 10, 'size': 28},
            'cifar10': {'num_channels': 3, 'num_classes': 10, 'size': 32},
            'cifar100': {'num_channels': 3, 'num_classes': 100, 'size': 32},
            'kaggle_clothing': {'num_channels': 3, 'num_classes': 10, 'size': 224},  # Will be determined dynamically
            'deep_fashion': {'num_channels': 1, 'num_classes': 17, 'size': 28} # Processed to match Fashion-MNIST format
        }
        
        if self.dataset_name not in self.dataset_params:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        self.params = self.dataset_params[self.dataset_name]
        
        # Define transforms
        if self.dataset_name == 'kaggle_clothing':
            # For larger images like those in Kaggle dataset, use resizing
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
            ])
            
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
            ])
        elif self.dataset_name == 'deep_fashion':
            # Images already resized to 28x28 and grayscale - similar to Fashion-MNIST
            self.train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Explicitly convert to grayscale
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
            
            self.test_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Explicitly convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Standard transforms for MNIST/Fashion-MNIST
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * self.params['num_channels'], 
                                  (0.5,) * self.params['num_channels'])
            ])
            
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * self.params['num_channels'], 
                                  (0.5,) * self.params['num_channels'])
            ])

    def get_dataset(self):
        """Load the specified dataset"""
        if self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST('./data', train=True, download=True,
                                         transform=self.train_transform)
            test_dataset = datasets.MNIST('./data', train=False,
                                        transform=self.test_transform)
        
        elif self.dataset_name == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                                transform=self.train_transform)
            test_dataset = datasets.FashionMNIST('./data', train=False,
                                               transform=self.test_transform)
        
        elif self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                           transform=self.train_transform)
            test_dataset = datasets.CIFAR10('./data', train=False,
                                          transform=self.test_transform)
        
        elif self.dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                            transform=self.train_transform)
            test_dataset = datasets.CIFAR100('./data', train=False,
                                           transform=self.test_transform)
        
        elif self.dataset_name == 'kaggle_clothing':
            # Load the Kaggle clothing dataset
            kaggle_data_dir = './data/kaggle_clothing'
            
            # Check if dataset exists
            if not os.path.exists(kaggle_data_dir):
                raise FileNotFoundError(f"Kaggle clothing dataset not found at {kaggle_data_dir}. " 
                                       f"Please run 'python code/download_kaggle_data.py' first.")
            
            # Create datasets
            train_dataset = KaggleClothingDataset(kaggle_data_dir, transform=self.train_transform, train=True)
            test_dataset = KaggleClothingDataset(kaggle_data_dir, transform=self.test_transform, train=False)
            
            # Update num_classes based on actual dataset
            self.params['num_classes'] = len(train_dataset.classes)
        
        elif self.dataset_name == 'deep_fashion':
            # Load the processed Deep Fashion dataset
            deep_fashion_dir = './data/deep_fashion'
            
            # Check if dataset exists
            if not os.path.exists(deep_fashion_dir):
                raise FileNotFoundError(f"Deep Fashion dataset not found at {deep_fashion_dir}. " 
                                       f"Please run 'python code/prepare_deepfashion.py' first.")
            
            # Create datasets using ImageFolder
            train_dataset = datasets.ImageFolder(os.path.join(deep_fashion_dir, 'train'), 
                                               transform=self.train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(deep_fashion_dir, 'test'), 
                                              transform=self.test_transform)
            
            # Update num_classes based on actual dataset
            self.params['num_classes'] = len(train_dataset.classes)
        
        # Split training data into train and validation sets
        if self.dataset_name not in ['kaggle_clothing', 'deep_fashion']:
            # For standard datasets, split the train set
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            # For custom datasets, split the train set
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            # Use generator for deterministic behavior
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        
        # Optionally subsample the datasets for faster iteration during quick tests
        if self.subset_fraction < 1.0:
            import random
            def _get_subset(dataset):
                subset_len = max(1, int(len(dataset) * self.subset_fraction))
                indices = random.sample(range(len(dataset)), subset_len)
                return torch.utils.data.Subset(dataset, indices)

            train_dataset = _get_subset(train_dataset)
            val_dataset = _get_subset(val_dataset)
            # We usually don't need the test set during rapid iteration, but keep it consistent
            test_dataset = _get_subset(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self):
        """Create data loaders for train, validation and test sets"""
        train_dataset, val_dataset, test_dataset = self.get_dataset()
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader, test_loader

    def get_dataset_params(self):
        """Return dataset specific parameters"""
        return self.params
