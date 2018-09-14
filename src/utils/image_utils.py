import cv2
import numpy as np


def load_and_resize_image(image_path: str):
    """
    Loads image as gray scale and scales to size `100 x 32`
    :param image_path: Path to image file
    :return: Numpy array with gray scale image of size `100 x 32`
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 32))
    equ = cv2.equalizeHist(image)
    img2 = np.zeros((32, 100, 3))
    img2[:, :, 0] = equ
    img2[:, :, 1] = equ
    img2[:, :, 2] = equ
    return np.expand_dims(img2, axis=0).astype(np.float32)

