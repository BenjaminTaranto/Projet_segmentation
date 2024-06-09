import matplotlib.pyplot as plt
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def plot_metrics(train_losses_Paul, train_accuracies_Paul,train_losses_Benjamin, train_accuracies_Benjamin, train_losses_Melvyn, train_accuracies_Melvyn,train_losses_Hugo, train_accuracies_Hugo,train_losses_Younes, train_accuracies_Younes,train_losses_Ethan, train_accuracies_Ethan):    
    epochs = range(1, len(train_losses_Benjamin) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_Benjamin, 'b', label='Training loss Benjamin')
    plt.plot(epochs, train_losses_Paul, 'r', label='Training loss Paul')
    plt.plot(epochs, train_losses_Melvyn, 'g', label='Training loss Melvyn')
    plt.plot(epochs, train_losses_Hugo, 'y', label='Training loss Hugo')
    plt.plot(epochs, train_losses_Younes, 'm', label='Training loss Younes')
    plt.plot(epochs, train_losses_Ethan, 'c', label='Training loss Ethan')
    #plt.plot(epochs, train_losses2, label='cos(x)', color='red')  
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies_Benjamin, 'b', label='Training accuracy Benjamin')
    plt.plot(epochs, train_accuracies_Paul, 'r', label='Training accuracy Paul')
    plt.plot(epochs, train_accuracies_Melvyn, 'g', label='Training accuracy Melvyn')
    plt.plot(epochs, train_accuracies_Hugo, 'y', label='Training accuracy Hugo')
    plt.plot(epochs, train_accuracies_Younes, 'm', label='Training accuracy Younes')
    plt.plot(epochs, train_accuracies_Ethan, 'c', label='Training accuracy Ethan')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    

def plot_metrics2(data_accuracy_test,data_loss_test,):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(data_loss_test, bins=5, edgecolor='black', color='skyblue', alpha=0.7)
    plt.title('Test Loss')
    plt.xlabel('Différents modèles')
    plt.ylabel('Résultats')
    plt.subplot(1, 2, 2)
    plt.hist(data_accuracy_test, bins=5, edgecolor='black', color='skyblue', alpha=0.7)
    plt.title('Test accuracy')
    plt.xlabel('Différents modèles')
    plt.ylabel('Résultats')
    plt.show()

def plot_metrics1(train_losses_Benjamin, train_accuracies_Benjamin):    
    epochs = range(1, len(train_losses_Benjamin) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_Benjamin, 'b', label='Training loss Benjamin')
    
    #plt.plot(epochs, train_losses2, label='cos(x)', color='red')  
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies_Benjamin, 'b', label='Training accuracy Benjamin')

    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()