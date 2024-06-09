import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_classes_Paul(dataset, classes):
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, filtered_indices)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 2)  # 3 * 32 * 32 corresponds to CIFAR10 image dimensions

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image to a vector
        x = self.fc(x)
        return x
def train_model_losses(model, train_loader, optimizer, criterion, num_epochs):
    train_losses = []  # Liste pour stocker les pertes
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        
        # Ajouter la perte moyenne de l'époque à la liste
        train_losses.append(running_loss / len(train_loader))
        
    
    return train_losses

def evaluate_model_accuracy(model, test_loader, num_epochs):
    accuracies = []  # Liste pour stocker les précisions
    
    model.eval()
    correct = 0
    total = 0
    for epoch in range(num_epochs):
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {accuracy/len(accuracy)}")
        # Calcul de la précision sur l'ensemble de test
            accuracy = correct / total
            
            accuracies.append(accuracy)
    
    return accuracies