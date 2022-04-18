from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import numpy as np
import cv2


class BriskClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом Brisk.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом Brisk.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        brisk = cv2.BRISK_create()
        _, des = brisk.detectAndCompute(image, None)
        des = cv2.resize(des, (250, 250))
        des = des.reshape(250, 250)

        return des
