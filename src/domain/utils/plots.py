import matplotlib.pyplot as plt
import numpy as np
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



def plot_garbor(garbor: np.array) -> io.BytesIO:
    """
    Преобразование Garbor дескриптора для отображения.
    """
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(garbor)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf



def plot_hist(hist: np.array) -> io.BytesIO:
    """
    Преобразование Hist дескриптора для отображения.
    """
    fig = plt.figure(figsize=(4, 4))
    color = ('b', 'g', 'r')
    for j in range(0, 3):
        plt.plot(hist[j], color=color[j])
        plt.xlim([0, 64])
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return buf
