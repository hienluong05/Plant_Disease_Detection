import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torchvision import datasets, transforms, models  # datsets  , transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

import dataProcessor 


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024), # 50176 is number of chanels * width * height of image after convolutional layer, 
                                    # 1024 is number of nodes (neurals) of the first fully connected layer selected by designer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176) # convert feature tensor to flatten to take to fully connected

        # Fully connected
        out = self.dense_layers(out)

        return out
    
# Select device: cuda if GPU is available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
model = CNN(dataProcessor.targets_size) # Create model with number of classes
model.to(device) # Take model to device

# Loss function and optimizer
criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters())

# Train using batch
def train_model(model, criterion, train_loader, validation_loader, epochs):
    # Initialize 2 arrays to store lossses in each epoch for train and test
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)
    
    for epoch in range(epochs):
        t0 = datetime.now() # Variable to measure time taken for each epoch
        
        # Train on training set
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad() # remove previous gradient
            output = model(inputs)
            loss = criterion(output, targets) # calculate loss between predicted and actual
            train_loss.append(loss.item()) # store loss of batch to list
            loss.backward() # calculate gradient
            optimizer.step() # update arguments of model based on gradient
        
        train_loss = np.mean(train_loss) # get everage loss for the epoch
        
        # Validate on validation set: similar to tranning but without gradient calculation and optimizer step
        # because we only validate the model, not train it
        validation_loss = []
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device) # take data to calculation device
            output = model(inputs)
            loss = criterion(output, targets)
            validation_loss.append(loss.item())
            
        validation_loss = np.mean(validation_loss)
        
        # Store losses
        train_losses[epoch] = train_loss
        validation_losses[epoch] = validation_loss
        
        dt = datetime.now() - t0
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time: {dt}")
        
    return train_losses,validation_losses # to draw graph or validate the model later

device = "cpu"

# create DataLoader for training, validation and test sets
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset, 
    batch_size=batch_size, 
    sampler=dataProcessor.train_sampler
)

test_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset,      
    batch_size=batch_size,
    sampler=dataProcessor.test_sampler
)

validation_loader = torch.utils.data.DataLoader(
    dataProcessor.dataset,
    batch_size = batch_size,
    sampler = dataProcessor.validation_sampler
)

# train the model in 5 epochs
train_losses, validation_losses = train_model(model, criterion, train_loader, validation_loader, 5)

torch.save(model.state_dict(), "plant_disease_detection_model.pt")

# targets_size = 39
# model = CNN(targets_size)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

# # draw graph of losses to validate the learning process
# plt.plot(train_losses , label = 'train_loss')
# plt.plot(validation_losses , label = 'validation_loss')
# plt.xlabel('No of Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Calculate accuracy 
# def accuracy(loader):
#     n_correct = 0
#     n_total = 0

#     for inputs, targets in loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         _, predictions = torch.max(outputs, 1)
#         n_correct += (predictions == targets).sum().item()
#         n_total += targets.shape[0]

#     acc = n_correct / n_total
#     return acc

# # Print accuracy for training, validation and test sets
# train_acc = accuracy(train_loader)
# test_acc = accuracy(test_loader)
# validation_acc = accuracy(validation_loader)
# print(
#     f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}"
# )

# # single image prediction
# transform_index_to_disease = dataProcessor.dataset.class_to_idx
# transform_index_to_disease = dict(
#     [(value, key) for key, value in transform_index_to_disease.items()]
# )

# data = pd.read_csv("disease_info.csv", encoding="cp1252")
# from PIL import Image
# import torchvision.transforms.functional as TF

# def single_prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     print("Original : ", image_path[12:-4])
#     pred_csv = data["disease_name"][index]
#     print(pred_csv)
    
# single_prediction("C:/Users/Admin/OneDrive - Hanoi University of Science and Technology/Documents/Plant_Disease_Detection/MyAI/PlantVillage/Apple___Apple_scab/0a1f4c8b-3d2e-4f5c-9b6d-7e2c1f8b3c1a___RS_ApplScab 1006.JPG")
# # ... và các file test khác