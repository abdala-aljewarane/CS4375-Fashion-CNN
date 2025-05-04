# Comparing CNN Architectures for Fashion Item Classification

CS-4375 Course Project | Spring 2025

## 👥 Team Members
- [Abdala Aljewarane] ([axa210260]@utdallas.edu)
- [Lerich Osay] ([lerich.osay]@utdallas.edu) 


## 📝 Project Overview
This project implements and compares two CNN architectures (LeNet-5 and VGG16) for fashion item classification across two datasets of varying complexity. We investigate how model architecture affects performance when scaling from simple to complex fashion datasets.

### Research Questions
1. How does model complexity affect classification accuracy across different fashion datasets?
2. How well do models trained on simple datasets (Fashion-MNIST) generalize to complex real-world fashion images (DeepFashion2)?
3. What are the computational trade-offs between lightweight and deep architectures?

## 🏗️ Implemented Architectures
- **LeNet-5**: Classic lightweight CNN (62K parameters)
- **VGG16**: Deep convolutional network (138M parameters)
- **Modified LeNet-5**: Enhanced version with batch normalization and dropout (85K parameters)

## 📊 Datasets

### 1. Fashion-MNIST
- 70,000 grayscale images (28×28)
- 10 fashion categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Clean, centered, preprocessed images

### 2. DeepFashion2
- 491K images with 13 clothing categories
- Real-world images with varied poses, occlusion, and backgrounds
- For this project, we use a subset:
  - 50,000 images (5,000 per category)
  - Resized to 64×64 for computational efficiency
  - 10 categories selected to match Fashion-MNIST where possible

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (required for DeepFashion2)
- 10GB+ disk space

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/[abdala-aljewarane]/CS4375-Fashion-CNN.git
   cd CS4375-Fashion-CNN