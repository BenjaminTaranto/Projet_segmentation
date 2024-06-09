import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch import nn, save, load
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import Subset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def unpickle_Melvyn(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def select_dataset_classes_Melvyn(dataset, classes):
    """
    Crée une sous-base de données contenant seulement certaines classes choisies.

    Args:
        dataset (torch.utils.data.Dataset): Le jeu de données complet.
        classes (list[int]): Liste des indices des classes à inclure dans la sous-base de données.

    Returns:
        torch.utils.data.Subset: La sous-base de données contenant seulement les classes sélectionnées.
    """
    selected_indices = [idx for idx, (image, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, selected_indices)

class ImageClassifier(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 channels en entrée, 16 channels en sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.MaxPool2d(2),  # Max pooling avec une taille de noyau de 2x2 et un stride par défaut de 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 channels en entrée, 32 channels en sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.MaxPool2d(2),  # Max pooling avec une taille de noyau de 2x2 et un stride par défaut de 2
            nn.Flatten(),  # Aplatir la carte des caractéristiques en un vecteur pour la couche linéaire
            nn.Linear(32 * 8 * 8, 128),  # Couche linéaire avec 32*8*8 caractéristiques d'entrée et 128 caractéristiques de sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Linear(128, nb_classes)  # Couche linéaire avec 128 caractéristiques d'entrée et nb_classes caractéristiques de sortie
        )

    def forward(self, x):
        return self.model(x)
class ImageClassifier2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 channels en entrée, 16 channels en sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 channels en entrée, 32 channels en sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Flatten(),  # Aplatir la carte des caractéristiques en un vecteur pour la couche linéaire
            nn.Linear(32 * 32 * 32, 128),  # Couche linéaire avec 32*32*32 caractéristiques d'entrée et 128 caractéristiques de sortie
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Linear(128, 10)  # Couche linéaire avec 128 caractéristiques d'entrée et 10 caractéristiques de sortie
        )

    def forward(self, x):
        return self.model(x)
    # Initialisation du classificateur
