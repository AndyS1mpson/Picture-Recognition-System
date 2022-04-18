import matplotlib.pyplot as plt
import numpy as np
import cv2
import io


def plot_hog(hog: np.array) -> io.BytesIO:
    """
    Преобразование HoG дескриптора для отображения.
    """
    bin = 5
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.pcolor(hog[:, :, bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf



def plot_descriptor(orb: np.array) -> io.BytesIO:
    """
    Преобразование ORB дескриптора для отображения.
    """
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(orb)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf

