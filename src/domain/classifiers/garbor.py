from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import cv2
import numpy as np


class GarborClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом Garbor.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом Garbor.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        filters = []
        ksize = 51
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
        accum = np.zeros_like(image)
        for kern in filters:
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum