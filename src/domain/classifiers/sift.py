from typing import List

import cv2
import numpy as np
from domain.classifiers.base_classifier import BaseClassifier


class SIFTClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение признаков
    с помощью гистограммы цветов.
    """

    def get_features(self, image: np.ndarray) -> List:
        #convert to grayscale image
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #initialize SIFT object
        sift = cv2.SIFT_create()

        #detect keypoints
        _, desc= sift.detectAndCompute(gray_scale, None)

        desc = cv2.resize(desc, (250, 250))
        desc = desc.reshape(250, 250)

        return desc
