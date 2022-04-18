from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import cv2
import numpy as np


class HarrisClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом Harris.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом Harris.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)

        dst = cv2.resize(dst, (250, 250))
        dst = dst.reshape(250, 250)
        return dst
