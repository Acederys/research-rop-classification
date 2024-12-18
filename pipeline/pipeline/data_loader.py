import s3fs
import os
import yaml
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from augmentation import apply_clahe, normalization_background

class CustomDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.image_paths = [x[0] for x in self.dataset.samples]

    def __len__(self):
        return len(self.dataset)  # Возвращаем исходное количество изображений

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        augmented_image = self.transform(image)  # Применяем аугментацию один раз
        return augmented_image, label, self.image_paths[idx]

class CustomDoubleDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.image_paths = [x[0] for x in self.dataset.samples]

    def __len__(self):
        return len(self.dataset) * 2  # Удваиваем количество изображений

    def __getitem__(self, idx):
        original_idx = idx // 2  # Определяем индекс исходного изображения
        image, label = self.dataset[original_idx]

        # Применяем аугментацию дважды
        augmented_image1 = self.transform(image)
        augmented_image2 = self.transform(image)

        if idx % 2 == 0:
            return augmented_image1, label, self.image_paths[original_idx]
        else:
            return augmented_image2, label, self.image_paths[original_idx]

# Функция для загрузки преобразований из YAML-файла
def load_transforms_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    transform_list = []
    for transform_config in config['transforms']:
        name = transform_config['name']
        params = transform_config['params']

        if name == 'normalization_background':
            transform_list.append(transforms.Lambda(normalization_background))
        if name == 'apply_clahe':
            transform_list.append(transforms.Lambda(apply_clahe))
        elif name == 'resize':
            transform_list.append(transforms.Resize(params['size']))
        elif name == 'random_horizontal_flip':
            transform_list.append(transforms.RandomHorizontalFlip(p=params['p']))
            
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

# Функция для загрузки файлов, если папка не существует
def download_files_if_not_exists(config):
    s3_config = config['s3']
    local_folder = s3_config['local_folder']
    print('Check if data already installed...')
    # Проверка существования локальной папки
    if not os.path.exists(local_folder):
        # Создание локальной папки
        os.makedirs(local_folder, exist_ok=True)

        # Создание файловой системы S3
        s3 = s3fs.S3FileSystem(key=s3_config['access_key'], secret=s3_config['secret_key'], client_kwargs={'endpoint_url': s3_config['endpoint_url']})

        # Скачивание папки
        s3.get(f"{s3_config['bucket_name']}/{s3_config['folder_name']}", local_folder, recursive=True)

        print(f"Downloaded {s3_config['folder_name']} to {local_folder}")
    else:
        print(f"Folder {local_folder} already exists. Skipping download.")

# Функция для создания датасетов
def create_datasets(config, transform):
    datasets_config = config['datasets']
    train_folder = os.path.join(config['s3']['local_folder'], datasets_config['train_folder'])
    val_folder = os.path.join(config['s3']['local_folder'], datasets_config['val_folder'])
    num = datasets_config['mult']

    if num:
        train_dataset = CustomDoubleDataset(
            root=train_folder,
            transform=transform
        )

        val_dataset = CustomDoubleDataset(
            root=val_folder,
            transform=transform
        )
    else:
        train_dataset = CustomDataset(
            root=train_folder,
            transform=transform
        )

        val_dataset = CustomDataset(
            root=val_folder,
            transform=transform
        )

    return train_dataset, val_dataset

# Основная функция для выполнения всех шагов
def main_dataloader(yaml_path):
    # Загрузка конфигурации из YAML-файла
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Загрузка преобразований
    transform = load_transforms_from_yaml(yaml_path)

    # Загрузка файлов, если папка не существует
    download_files_if_not_exists(config)

    # Создание датасетов
    train_dataset, val_dataset = create_datasets(config, transform)
    print('Dataloader finished his job')
    return train_dataset, val_dataset