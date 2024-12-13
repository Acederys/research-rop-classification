import cv2
from PIL import Image
import numpy as np

def normalization_background(eye: Image.Image) -> Image.Image:
    eye = np.array(eye)
    eye = cv2.cvtColor(eye, cv2.COLOR_RGB2BGR)
    # Немного затемняем и осветляем и получаем чб
    contrast_eye_dark = cv2.convertScaleAbs(eye, alpha=0.5, beta=150)
    contrast_eye_light = cv2.convertScaleAbs(eye, alpha=1.5, beta=100)
    gray_eye_dark = cv2.cvtColor(contrast_eye_dark, cv2.COLOR_BGR2GRAY)
    gray_eye_light = cv2.cvtColor(contrast_eye_light, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    # Выделяем объекты с помощью THRESH_TRIANGLE
    triangle_eye = cv2.threshold(gray_eye_dark, 0, 255, cv2.THRESH_TRIANGLE)[1]

    # Выделяем объекты с помощью THRESH_TRIANGLE
    otsu_eye = cv2.threshold(gray_eye_light, 0, 255, cv2.THRESH_OTSU)[1]

    bin_eye = cv2.threshold(gray_eye, 20, 255, cv2.THRESH_BINARY)[1]

    # Используем морф, чтобы почистить от шумов и сделать контуры более гладкими
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    triangle_morph_eye = cv2.morphologyEx(triangle_eye, cv2.MORPH_CLOSE, kernel_ellipse, iterations=5)
    otsu_morph_eye = cv2.morphologyEx(otsu_eye, cv2.MORPH_CLOSE, kernel_ellipse, iterations=5)
    bin_morph_eye = cv2.morphologyEx(bin_eye, cv2.MORPH_CLOSE, kernel_ellipse, iterations=5)

    # На основе конура создаем маску
    mask_eye = np.zeros_like(eye)

    # Выделяем контуры на снимке, и выбираем самый большой (по идее это глаз)
    triangle_contours = cv2.findContours(triangle_morph_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    otsu_contours = cv2.findContours(otsu_morph_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    bin_contours = cv2.findContours(bin_morph_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if triangle_contours:
        max_contour = max(triangle_contours, key=cv2.contourArea)
        cv2.fillConvexPoly(mask_eye, max_contour, (255, 255, 255, 255))

    if otsu_contours:
        max_contour = max(otsu_contours, key=cv2.contourArea)
        cv2.fillConvexPoly(mask_eye, max_contour, (255, 255, 255, 255))

    if bin_contours:
        max_contour = max(bin_contours, key=cv2.contourArea)
        cv2.fillConvexPoly(mask_eye, max_contour, (255, 255, 255, 255))

    clear_eye = cv2.cvtColor(cv2.bitwise_and(eye, mask_eye), cv2.COLOR_BGR2RGB)

    return Image.fromarray(clear_eye)


def apply_clahe(image):
    # Преобразуем изображение из PIL в numpy array
    image_np = np.array(image)

    # Применяем CLAHE к каждому каналу (RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_np[:, :, 0] = clahe.apply(image_np[:, :, 0])
    image_np[:, :, 1] = clahe.apply(image_np[:, :, 1])
    image_np[:, :, 2] = clahe.apply(image_np[:, :, 2])

    # Преобразуем обратно в PIL Image
    image_pil = Image.fromarray(image_np)
    return image_pil