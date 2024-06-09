# Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import torch
from torch import nn, save, load
import torchvision
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fonction pour désérialiser les données CIFAR-10


def filter_classes_Benjamin(dataset, classes):
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, filtered_indices)

# Définition de la classe Block
class Block(nn.Module):
    def __init__(self, num_classes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32*8*8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Fonction pour entraîner le modèle
def train_model_Benjamin(model, train_loader, optimizer, loss_function, num_epochs):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            # Calculate loss
            running_loss += loss.item()
            # Calculate accuracy per batch
            batch_acc = accuracy_Benjamin(outputs, labels)
            correct_predictions += batch_acc * inputs.size(0)  # Accumulate correct predictions
            total_predictions += inputs.size(0)  # Accumulate total predictions

        # Calculate epoch accuracy
        epoch_acc = correct_predictions / total_predictions
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return train_losses, train_accuracies

def accuracy_Benjamin(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    labels = labels.squeeze()  # Remove the extra dimension
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

def loss_function_Benjamin(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss

def get_optimizer_Benjamin(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer


def test_model_Benjamin(model, test_loader, loss_function):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        running_loss += loss.item()
        batch_acc = accuracy_Benjamin(outputs, labels)
        correct_predictions += batch_acc * inputs.size(0)
        total_predictions += inputs.size(0)

    test_loss = running_loss / len(test_loader)
    test_acc = correct_predictions / total_predictions

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc
