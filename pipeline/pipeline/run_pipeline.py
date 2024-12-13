from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import data_loader
import train_modules

def run_pipeline():
    train_dataset, val_dataset = data_loader.main_dataloader('../config/data_loader_config.yaml')
    model, criterion, optimizer, device, num_epochs, batch_size, k_folds = train_modules.load_training_config_from_yaml('../config/train_modules_config.yaml')

    # Создаем DataLoader для тренировочного и валидационного датасетов
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,   
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
        )

    model = train_modules.train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    accuracy, f1, precision, recall = train_modules.calculate_metrics(model, val_loader)

    print(f'Final Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')


def run_kfold_pipeline():
    train_dataset, val_dataset = data_loader.main_dataloader('../config/data_loader_config.yaml')
    model, criterion, optimizer, device, num_epochs, batch_size, k_folds = train_modules.load_training_config_from_yaml('../config/train_modules_config.yaml')

    # Определяем количество фолдов
    k = k_folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold+1}')

        # Создаем тренировочный и валидационный датасеты
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)

        # Создаем DataLoader для тренировочного и валидационного датасетов
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        model = train_modules.train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

        accuracy, f1, precision, recall = train_modules.calculate_metrics(model, val_loader)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        print(f'Fold {fold+1} - Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')

    # Усредняем метрики
    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    print(f'Mean Accuracy: {mean_accuracy}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Mean Precision: {mean_precision}')
    print(f'Mean Recall: {mean_recall}')


if __name__ == "__main__":
    run_pipeline()