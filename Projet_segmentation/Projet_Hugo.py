import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch import nn, save, load
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def create_subset(dataset, classes, root='./data', train=True, download=True):
    """
    Crée une sous-base de données contenant uniquement les classes spécifiées.

    Args:
        dataset (str): Le nom du dataset ('cifar10' ou 'cifar100').
        classes (list): La liste des classes à inclure dans la sous-base de données.
        root (str): Le chemin du répertoire où stocker les données (défaut: './data').
        train (bool): Si True, la sous-base de données sera basée sur les données d'entraînement, sinon sur les données de test (défaut: True).
        download (bool): Si True, télécharge les données si elles ne sont pas présentes localement (défaut: True).

    Returns:
        torch.utils.data.Dataset: La sous-base de données créée.
    """
    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(root):
        os.makedirs(root)

    # Charger les données CIFAR-10 ou CIFAR-100 selon la valeur de 'dataset'
    if dataset == 'cifar10':
        cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
    elif dataset == 'cifar100':
        cifar_dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download)
    else:
        raise ValueError("Le nom du dataset doit être 'cifar10' ou 'cifar100'.")

    # Récupérer les indices des échantillons appartenant aux classes spécifiées
    indices = [i for i, (_, label) in enumerate(cifar_dataset) if label in classes]

    # Créer une sous-base de données basée sur les indices sélectionnés
    subset = torch.utils.data.Subset(cifar_dataset, indices)

    return subset


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_classes)  # Corrected this line

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x