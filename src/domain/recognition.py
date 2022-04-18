from collections import Counter
from typing import List, Tuple

from domain.classifiers import (GarborClassifier, HarrisClassifier,
                                HoGClassifier, ORBClassifier, SIFTClassifier,
                                BriskClassifier)
from domain.utils.load_data import load
from domain.utils.plots import plot_descriptor, plot_hog


def recognition(image: List) -> Tuple[str, List]:
    """Классификация стиля изображения.

    Args:
        image (List):
            входные изображения.

    Returns:
        Tuple[str, List]:
            метки классов,
            результаты работы дескрипторов.
    """
    # load train sample
    data = load()

    # create classifiers for voting classification 
    classifiers = [
        HoGClassifier(),
        ORBClassifier(),
        SIFTClassifier(),
        HarrisClassifier(),
        GarborClassifier(),
        BriskClassifier()
    ]

    # split data to X and y
    X_train = [img for img, _ in data]
    y_train = [style for _, style in data]

    # fit classifiers
    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    # predict
    class_marks = Counter()
    features = []
    for classifier in classifiers:
        mark_with_feature = classifier.predict(image)
        class_marks[mark_with_feature[0][0]] += 1
        features.append(mark_with_feature[0][1])

    # retrieve most common mark
    mark = class_marks.most_common(1)[0][0]

    descriptors = [
        plot_hog(features[0]),
        plot_descriptor(features[1]),
        plot_descriptor(features[2]),
        plot_descriptor(features[3]),
        plot_descriptor(features[4]),
        plot_descriptor(features[5])
    ]

    return mark, descriptors
