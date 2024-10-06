# research-rop-classification
CNN Classifier for Recognizing Retinopathy of Premature

- Learning EfficientNet model from the Pytorch library – [rop-efficialnet-pytorch.ipynb]  
- Interpreting model using Grad-CAM – [rop-effnet-grad-cam (2).ipynb]


## Подсчет метрик для YOLO и EfficientNet

Загружаем две модели. Используем тестовый датасет с 80 изображениями. 

Демаем предсказание и записываем его в файл. 

Делаем посчет метрик:

### Precision, Recall, Accuracy

#### 1. **Precision** (Точность)
- **Определение:** Precision показывает, какая доля предсказанных положительных классов является правильной.
- **Формула:**
  
$$\text{Precision} = \frac{TP}{TP + FP}$$

  где:
  - **TP (True Positive)** — количество истинно положительных предсказаний,
  - **FP (False Positive)** — количество ложноположительных предсказаний.

#### 2. **Recall** (Полнота)
- **Определение:** Recall измеряет, какую долю всех истинных положительных классов модель смогла правильно классифицировать.
- **Формула:**
  
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  где:
  - **TP (True Positive)** — количество истинно положительных предсказаний,
  - **FN (False Negative)** — количество ложноотрицательных предсказаний.

#### 3. **Accuracy** (Доля правильных ответов)
- **Определение:** Accuracy показывает, какую долю всех предсказаний модель сделала правильно.
- **Формула:**
  
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  
  где:
  - **TP (True Positive)** — количество истинно положительных предсказаний,
  - **TN (True Negative)** — количество истинно отрицательных предсказаний,
  - **FP (False Positive)** — количество ложноположительных предсказаний,
  - **FN (False Negative)** — количество ложноотрицательных предсказаний.

### Матрица путаницы (Confusion Matrix)

**Матрица путаницы** — это таблица, которая используется для оценки качества работы классификационной модели. Она показывает, сколько предсказаний модель сделала правильно или ошибочно по каждому классу. В бинарной классификации матрица выглядит так:

|                    | **Предсказано: Позитив** | **Предсказано: Негатив** |
|--------------------|--------------------------|--------------------------|
| **Истинное: Позитив**  | True Positive (TP)        | False Negative (FN)       |
| **Истинное: Негатив**  | False Positive (FP)       | True Negative (TN)        |

- **TP (True Positive)** — количество правильно предсказанных положительных примеров.
- **TN (True Negative)** — количество правильно предсказанных отрицательных примеров.
- **FP (False Positive)** — количество неправильно предсказанных положительных примеров (ложноположительные).
- **FN (False Negative)** — количество неправильно предсказанных отрицательных примеров (ложноотрицательные).

Матрица путаницы позволяет вычислять метрики, такие как:
- **Precision (Точность)**
- **Recall (Полнота)**
- **Accuracy (Доля правильных ответов)**


### Доверительный интервал (Confidence Interval)

Вычисляем доверительный интервал 95% для уверенности модели в своих ответах.

# Обучение моделей
## Аугментации
Для аугментаций испольовалась библиотека [Albumentations](https://albumentations.ai/docs/)
### Функция для аугментаций 
Вход картинка и выход картинка, читаемая библиотекой cv2

Для использования в других библиотеках(например, matplotlib) необходимо преобразование, т.к. в cv2 каналы изображения в другом порядке: 
```
cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
```
Функция применяет аугментацию к картинке с определенной вероятностью, которая записана в prob_b, prob_m, prob_s
``` 
def augmentator(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prob_b = 0.7
    prob_m = 0.5
    prob_s = 0.3
    transform = A.Compose([
        A.SafeRotate(    limit=(-18, 18),  interpolation=2,  border_mode=2,  p=prob_s),
        A.HorizontalFlip(p=prob_b),
        A.VerticalFlip(  p=prob_b),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=prob_s),
        A.RandomBrightnessContrast(p=prob_m),
        A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.2, alpha_coef=0.08,p=prob_s),
        A.Sharpen(alpha=(0.1, 0.5), lightness=(0.5, 1.0), p=prob_b),
        A.RandomToneCurve(scale=0.1,p=prob_m),
        A.RingingOvershoot(blur_limit=(1, 111),cutoff=(0.5, 2.2),p=prob_m),
        ])

    transimage =  transform(image=image)['image']

    return transimage
```
### Структура аугментированного датасета
```
# dataset folders creation
!mkdir first_dataset_coco

# train folder creation
!mkdir 'first_dataset_coco/train'
!mkdir 'first_dataset_coco/train/healthy'
!mkdir 'first_dataset_coco/train/unhealthy'

# test folder creation
!mkdir 'first_dataset_coco/test'
!mkdir 'first_dataset_coco/test/healthy'
!mkdir 'first_dataset_coco/test/unhealthy'

# val folder creation
!mkdir 'first_dataset_coco/val'
!mkdir 'first_dataset_coco/val/healthy'
!mkdir 'first_dataset_coco/val/unhealthy'
```

## Ноутбуки с обучением
 - [EfficientNet](https://www.kaggle.com/code/artemsattarov/efficientnet-v2b1-for-article-train-e20-r224-embed)
 - [YOLOv8n](https://www.kaggle.com/code/artemsattarov/yolo-for-article-train-70)

