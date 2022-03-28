from typing import List

import cv2
import numpy as np
from domain.classifiers.base_classifier import BaseClassifier


class HistClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение признаков
    с помощью гистограммы цветов.
    """

    def get_features(self, image: np.ndarray) -> List:
        color = ('b', 'g', 'r')
        hists = []
        for i, _ in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [64], [0, 256])
            hists.append(hist)
        return hists
