import numpy as np


def reordering(cm: np.ndarray, labels: list, order: list):

    cm2 = np.ndarray(cm.shape)

    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            nx = order.index(labels[x])
            ny = order.index(labels[y])
            cm2[nx, ny] = cm[x, y]

    return cm2
