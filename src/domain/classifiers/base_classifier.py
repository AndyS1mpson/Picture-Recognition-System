from typing import Callable, List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    def __init__(self) -> None:
        """Инициализация.
        Args:
            method:
                функция, извлекающая признаки из изображения.
        """
        self.train_data: List = []

    @abstractmethod
    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признакового описания изображения.
        Args:
            image:
                изображение.
        Returns:
            List:
                признаковое описание.
        """


    def __dist(self, arr1: np.array, arr2: np.array) -> float:
        """
        Нахождение расстояния между двумя векторами
        в признаковом описании объектов.
        Args:
            arr1:
                вектор из признакового описания объектов.
            arr2:
                вектор из признакового описания объектов.
        Returns:
            float:
                расстояние между данными векторами
        """
        return np.linalg.norm(np.array(arr1) - np.array(arr2))

    def __search(self, test: np.array) -> Tuple:
        """
        Поиск эталона для тестового изображения
        по критерию минимального расстояния.
        Args:
            test:
                тестовое изображение.
        Returns:
            Tuple:
                изображение, наиболее 'схожее' с тестовым,
                номер группы которому принадлежит шаблон.
        """
        d_min = float("inf")
        template_min = None
        template_group = None
        for train, group in self.train_data:
            if (d := self.__dist(train, test)) < d_min:
                d_min = d
                template_min = train
                template_group = group
        return template_min, template_group

    def fit(self, X: List, y: List) -> None:
        train_data = []
        for index, image in enumerate(X):
            train_data.append((self.get_features(image), y[index]))
        self.train_data = train_data

    def predict(self, images: List) -> List:
        """Классификация изображений.
        Args:
            images:
                изображения которые надо проклассифицировать.
        Returns:
            List:
                список, содержащий назначенные классификатором группы.
        """
        predicted_groups = []
        for image in images:
            image_features = self.get_features(image)
            _, group_num = self.__search(image_features)
            predicted_groups.append(
                (
                    group_num,
                    image_features
                )
            )
        return predicted_groups

    def score(self, true_answers: List, predicted_answers: List) -> float:
        """Вычисление точности классификатора.
        Args:
            true_answers:
                вектор правильных меток изображений.
            predicted_answers:
                вектор предсказанных классификатором меток.
        Raises:
            Exception:
                вектора имеют разные размеры.
        Returns:
            float:
                точность.
        """
        if len(true_answers) != len(predicted_answers):
            raise Exception("Received arguments have different length")
        number_true = 0
        for index, ta in enumerate(true_answers):
            if ta == predicted_answers[index]:
                number_true += 1
        return number_true / len(predicted_answers)