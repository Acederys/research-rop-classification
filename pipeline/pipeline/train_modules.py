from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import models

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

    return accuracy, f1, precision, recall


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
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    num_classes = training_config['model']['num_classes']
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Загрузка функции потерь
    if loss_function_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
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

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

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

        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    return model