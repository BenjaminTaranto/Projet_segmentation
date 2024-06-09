import torchvision
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import ssl
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

#def unpickle(file):
    #import pickle
    #with open(file, 'rb') as fo:
    #    dict = pickle.load(fo, encoding='bytes')
    #return dict

#def main():
    #train_dataset = torchvision.datasets.CIFAR10( train=True, download=True)
    #test_dataset = torchvision.datasets.CIFAR10( train=False, download=True)

    #label_names = unpickle("data/cifar-10-batches-py/batches.meta")[b'label_names']
    #label_names = [name.decode('utf-8') for name in label_names]

    #transform = transforms.Compose([
        #transforms.ToTensor(),  # Convertir l'image en tenseur
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliser les valeurs des pixels
    #])

    #print("Nombre d'images dans l'ensemble d'entraînement:", len(train_dataset))
    #print("Nombre d'images dans l'ensemble de test:", len(test_dataset))

    #def filter_dataset(dataset, classes):
       # """
       # Crée une sous-base de données contenant seulement les classes spécifiées.

      #  Args:
        #    dataset (torch.utils.data.Dataset): L'ensemble de données d'origine.
         #   classes (list): Liste des classes à conserver.
#
       # Returns:
       #     torch.utils.data.Dataset: La sous-base de données contenant seulement les classes spécifiées.
      #  """
      #  indices = []
       # for idx, (data, target) in enumerate(dataset):
      #      if target in classes:
       #         indices.append(idx)
      #  return Subset(dataset, indices)

    # Classes choisies à conserver
    #chosen_classes = [0, 1]  # 0,1 (avion, automobile)

    # Créer une sous-base de données contenant seulement les classes choisies pour l'ensemble d'entraînement
   # train_subset = filter_dataset(train_dataset, chosen_classes)
   # test_subset = filter_dataset(test_dataset, chosen_classes)

    # Appliquer les transformations après le filtrage
   # train_subset.dataset.transform = transform
    #test_subset.dataset.transform = transform

    # DataLoader pour les sous-ensembles de données
    #trainloader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=2)
    #testloader = torch.utils.data.DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=2)

class CNN_C(nn.Module):
    def __init__(self, num_classes=10):  # Ajoutez num_classes ici
        super(CNN_C, self).__init__()
        # Définir les couches convolutionnelles, de pooling et entièrement connectées ici
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc = nn.Linear(32*8*8, num_classes)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Ajustez la taille de l'entrée en fonction de vos données (3 canaux, 32x32 pixels)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    