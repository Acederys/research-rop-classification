from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from manual_models import ResNetWithEmbeddings, EfficientNetWithEmbeddings


def calculate_metrics(model, val_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, f1, precision, recall, conf_matrix


def visualize_embeddings(embeddings, labels, save_path='../out_files/embeddings_visualization.png'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Embeddings')
    plt.savefig(save_path)
    plt.close()


def extract_and_visualize_embeddings(model, val_loader, device, save_path='../out_files/embeddings_visualization.png'):
    model.eval()
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model.get_embeddings(images)
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)

    visualize_embeddings(all_embeddings, all_labels, save_path)


def load_training_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    training_config = config['training']
    loss_function_name = training_config['loss_function']['name']
    optimizer_name = training_config['optimizer']['name']
    optimizer_params = training_config['optimizer']['params']
    num_epochs = training_config['num_epochs']
    batch_size = training_config['batch_size']
    k_folds = training_config['k_folds']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Загрузка модели
    num_classes = training_config['model']['num_classes']
    version = training_config['model']['version']
    if training_config['model']['type'] == 'resnet':
      model = ResNetWithEmbeddings()
      model = ResNetWithEmbeddings(num_classes = num_classes, version = version)
      model = model.to(device)
    elif training_config['model']['type'] == 'efficientnet':
      model = EfficientNetWithEmbeddings()
      model = EfficientNetWithEmbeddings(num_classes = num_classes, version = version)
      model = model.to(device)
    else:
      raise ValueError(f"Unknown model: {training_config['model']['name']}")

    # Загрузка функции потерь
    if loss_function_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif loss_function_name == 'HingeEmbeddingLoss':
        criterion = nn.HingeEmbeddingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function_name}")

    # Загрузка оптимизатора
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_params['lr'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_params['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return model, criterion, optimizer, device, num_epochs, batch_size, k_folds


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_model = None
    best_avg_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print('-' * 30)
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}, Accuracy: {100 * correct / total}%')

        avg_loss = (train_loss + val_loss) / 2
        if avg_loss < best_avg_loss:
            best_epoch = epoch + 1
            best_avg_loss = avg_loss
            best_model = model.state_dict()
            print(f'Epoch {epoch+1} is the new best epoch with average loss: {avg_loss}')

    model.load_state_dict(best_model)
    print(f'Loaded epoch {best_epoch} as the best epoch with average loss: {best_avg_loss}')
    print('-' * 30)
    return model