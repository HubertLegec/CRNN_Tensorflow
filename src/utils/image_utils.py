import cv2
import numpy as np


def load_and_resize_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 32))
    return np.expand_dims(image, axis=0).astype(np.float32)
