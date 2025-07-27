import torch
from torchvision import transforms, datasets, models
import numpy as np

# Import dataset and transform
transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=r'C:\Users\Admin\OneDrive - Hanoi University of Science and Technology\Documents\Plant_Disease_Detection\MyAI\PlantVillage', transform=transform)

# Print the number of images, original position and transforms applied
dataset 

# Split the dataset
indices = list(range(len(dataset)))
split = int(np.floor(0.85 * len(dataset)))  # 85% of all for training and validation
train = int(np.floor(0.70 * split))   # 70% of split for training, the rest for validation

# Check number of images in each set
print("Number of images in training set:", train)
print("Number of images in validation set:", split - train)    
print("Number of images in test set:", len(dataset) - split)

np.random.shuffle(indices)

train_indices, validation_indices, test_indices = (
    indices[:train], 
    indices[train:split], 
    indices[split:]
)

# Create samplers
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

# Get number of classes
targets_size = len(dataset.class_to_idx)
