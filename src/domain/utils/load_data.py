
from typing import List, Tuple
from domain.config.settings import DATA_CONF
import os
import cv2


def load() -> List:
    """Выгрузка данных.

    Returns:
        List:
            список, содержащий изображения каждого стиля и метку стиля.
    """
    data = []
    for style in DATA_CONF:
        for filename in os.listdir(DATA_CONF[style]):
            f = os.path.join(DATA_CONF[style], filename)
            img = cv2.imread(f)
            data.append(
                (
                    img,
                    style
                )
            )
    return data



