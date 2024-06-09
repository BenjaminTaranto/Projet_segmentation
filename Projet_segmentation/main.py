import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from Projet_Benjamin import *
from Projet_Ethan import *
import matplotlib.pyplot as plt
from Projet_Paul import *
from Graphique import *
from torch.optim import Adam
from Projet_Melvyn import *
from Projet_Hugo import *
from Projet_Younes import *
################ Variables globales ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Définir les transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger les données CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
label_names = unpickle("data/cifar-10-batches-py/batches.meta")[b'label_names']
label_names = [name.decode('utf-8') for name in label_names]
classes_to_include = [0,1,2,3,4,5,6,7,8,9]
criterion = nn.CrossEntropyLoss()
nb_epochs=10



def projet_Benjamin_main(classes_to_include,criterion,model,nb_epochs):
    
    train_binary_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_binary_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_binary_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary_subset, batch_size=64, shuffle=False)
    # Choix de la fonction de perte (loss function) et de l'optimiseur
    model.to(device)
    optimizer = get_optimizer_Benjamin(model) 
    train_losses_Benjamin,train_accuracies_Benjamin=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Benjamin, test_acc_Benjamin = test_model_Benjamin(model, test_loader, criterion)
    #print(test_loss_Benjamin, test_acc_Benjamin)
    return train_losses_Benjamin,train_accuracies_Benjamin,test_loss_Benjamin, test_acc_Benjamin
        
  

def projet_Paul_main(classes_to_include,criterion,model,nb_epochs):

    train_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    #train_losses_Paul=train_model_losses(model, train_loader, optimizer, criterion, nb_epochs)
    train_losses_Paul,train_accuracies_Paul=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Paul,test_accuracies_Paul = test_model_Benjamin(model, test_loader, criterion)
    return  train_losses_Paul,train_accuracies_Paul,test_loss_Paul,test_accuracies_Paul
def projet_Melvyn_main(classes_to_include,criterion,model,nb_epochs):
    
    train_binary_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_binary_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_binary_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary_subset, batch_size=64, shuffle=False)
    # Choix de la fonction de perte (loss function) et de l'optimiseur
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3) #Adam(clf.parameters(), lr=1e-3)
    train_losses_Melvyn,train_accuracies_Melvyn=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Melvyn, test_accuracies_Melvyn = test_model_Benjamin(model, test_loader, criterion)
    return train_losses_Melvyn,train_accuracies_Melvyn,test_loss_Melvyn,test_accuracies_Melvyn

def projet_Hugo_main(classes_to_include,criterion,model,nb_epochs):
    
    train_binary_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_binary_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_binary_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary_subset, batch_size=64, shuffle=False)
    # Choix de la fonction de perte (loss function) et de l'optimiseur
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    train_losses_Hugo,train_accuracies_Hugo=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Hugo, test_acc_Hugo = test_model_Benjamin(model, test_loader, criterion)
    return train_losses_Hugo,train_accuracies_Hugo,test_loss_Hugo, test_acc_Hugo

def projet_Ethan_main(classes_to_include,criterion,model,nb_epochs):
    
    train_binary_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_binary_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_binary_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary_subset, batch_size=64, shuffle=False)
    # Choix de la fonction de perte (loss function) et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    train_losses_Ethan,train_accuracies_Ethan=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Ethan, test_acc_Ethan = test_model_Benjamin(model,test_loader , criterion)
    #print(test_loss, test_acc)
    return train_losses_Ethan,train_accuracies_Ethan,test_loss_Ethan, test_acc_Ethan

def projet_Younes_main(classes_to_include,criterion,model,nb_epochs):
    
    train_binary_subset = filter_classes_Benjamin(train_dataset, classes_to_include)
    test_binary_subset = filter_classes_Benjamin(test_dataset, classes_to_include)
    # Création des DataLoader pour l'entraînement et le test
    train_loader = DataLoader(train_binary_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary_subset, batch_size=64, shuffle=False)
    # Choix de la fonction de perte (loss function) et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
    train_losses_Younes,train_accuracies_Younes=train_model_Benjamin(model, train_loader, optimizer, criterion, nb_epochs)
    test_loss_Younes, test_acc_Younes = test_model_Benjamin(model, test_loader, criterion)
    #print(test_loss, test_acc)
    return  train_losses_Younes,train_accuracies_Younes,test_loss_Younes, test_acc_Younes
        

# Exécuter les fonctions principales
data_accuracy_test=[]
data_loss_test=[]
train_losses_Benjamin, train_accuracies_Benjamin, test_loss_Benjamin, test_acc_Benjamin = projet_Benjamin_main(classes_to_include, criterion, Block(num_classes=10), nb_epochs)
data_accuracy_test.append(test_acc_Benjamin)
data_loss_test.append(test_loss_Benjamin)

train_losses_Paul,train_accuracies_Paul,test_loss_Paul,test_accuracies_Paul = projet_Paul_main(classes_to_include, criterion, SimpleCNN(10), nb_epochs)
data_accuracy_test.append(train_accuracies_Paul[-1])
data_loss_test.append(train_losses_Paul[-1])

train_losses_Melvyn,train_accuracies_Melvyn,test_loss_Melvyn,test_accuracies_Melvyn = projet_Melvyn_main(classes_to_include, criterion, ImageClassifier(10).to(device), nb_epochs)
data_accuracy_test.append(train_accuracies_Melvyn[-1])
data_loss_test.append(train_losses_Melvyn[-1])

train_losses_Hugo,train_accuracies_Hugo,test_loss_Hugo, test_acc_Hugo = projet_Hugo_main(classes_to_include, criterion,  Net(10), nb_epochs)
data_accuracy_test.append(train_accuracies_Hugo[-1])
data_loss_test.append(train_losses_Hugo[-1])

train_losses_Younes,train_accuracies_Younes,test_loss_Younes, test_acc_Younes = projet_Younes_main(classes_to_include, criterion, ConvNet(10), nb_epochs)
data_accuracy_test.append(train_accuracies_Younes[-1])
data_loss_test.append(train_losses_Younes[-1])

train_losses_Ethan,train_accuracies_Ethan,test_loss_Ethan, test_acc_Ethan = projet_Ethan_main(classes_to_include, criterion, CNN_C(10), nb_epochs)
data_accuracy_test.append(train_accuracies_Ethan[-1])
data_loss_test.append(train_losses_Ethan[-1])

plot_metrics2(data_accuracy_test,data_loss_test)
#plot_metrics(train_losses_Paul, train_accuracies_Paul,train_losses_Benjamin, train_accuracies_Benjamin, train_losses_Melvyn, train_accuracies_Melvyn,train_losses_Hugo, train_accuracies_Hugo,train_losses_Younes, train_accuracies_Younes,train_losses_Ethan, train_accuracies_Ethan)

#train_losses_Melvyn,train_accuracies_Melvyn=projet_Melvyn_main(classes_to_include,criterion,ImageClassifier(10).to(device),nb_epochs)
#train_lossetrain_losses_Ethan,train_accuracies_Ethan=projet_Ethan_main(classes_to_include,criterion,CNN_C(10),nb_epochs)s_Hugo,train_accuracies_Hugo=projet_Hugo_main(classes_to_include,criterion,Net(10).to(device),nb_epochs)

#train_losses_Ethan,train_accuracies_Ethan=projet_Ethan_main(classes_to_include,criterion,CNN_C(10),nb_epochs)
#train_losses_Younes,train_accuracies_Younes=projet_Younes_main(classes_to_include,criterion,ConvNet(10),nb_epochs)
#plot_metrics1(train_losses_Ethan, train_accuracies_Ethan)
#print(len(train_losses_Paul))
#print(len(train_accuracies_Paul))
#print(len(train_losses_Benjamin))
#print(len(train_accuracies_Benjamin))
#plot_metrics(train_losses_Paul, train_accuracies_Paul,train_losses_Benjamin, train_accuracies_Benjamin,train_losses_Melvyn, train_accuracies_Melvyn,train_losses_Hugo, train_accuracies_Hugo)
# Sauvegarder le modèle après l'entraînement
#model_path = "cifar10_model_state_dict.pth"
#torch.save(model.state_dict(), model_path)
#print(f"Model saved to {model_path}")

# Chargement du modèle pour l'inférence
#model = Block(num_classes=10)
#model.to(device)
#model.load_state_dict(torch.load(model_path))
#model.eval()
#print(f"Model loaded from {model_path}")