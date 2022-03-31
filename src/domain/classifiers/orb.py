from typing import List
from domain.classifiers.base_classifier import BaseClassifier
import numpy as np
import cv2



class ORBClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом ORB.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом ORB.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(image,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(image, kp)

        des = cv2.resize(des, (250, 250))
        des = des.reshape(250, 250)

        return des