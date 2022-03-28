from typing import List
from domain.classifiers.base_classifier import BaseClassifier
from skimage.feature import hog
import cv2
import numpy as np


class HoGClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом гистограммы градиента.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом HoG.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        cell_size = (8, 8)
        block_size = (2, 2)
        nbins = 9

        hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                                        image.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

        n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
        hog_feats = hog.compute(image) \
            .reshape(n_cells[1] - block_size[1] + 1,
                    n_cells[0] - block_size[0] + 1,
                    block_size[0], block_size[1], nbins) \
            .transpose((1, 0, 2, 3, 4))

        gradients = np.zeros((n_cells[0], n_cells[1], nbins))

        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

        gradients /= cell_count
        return gradients
