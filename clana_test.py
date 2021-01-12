import glob
import numpy as np
from numpy.lib.function_base import average
import torch
import matplotlib.pyplot as plt

import clana.visualize_cm as v
import clana.optimize as o
import torch

from glob import glob


def get_filename_txt(text):
    return text[0:-2] + "txt"


def get_labels(fname):
    with open(fname) as f:
        lines = [line.rstrip() for line in f]
        return lines


def get_average_accuracy(matrix):
    h, w = matrix.size()
    sums = 0
    for i in range(h):
        sums += matrix[i, i]
    return sums / h


def get_average_matrix(matrix):
    sum_cm = (1.0 / matrix.sum(axis=1)).reshape(-1, 1)
    average = sum_cm * matrix
    return average


def load_data(rootname):
    labels = get_labels(get_filename_txt(rootname))

    tensor = torch.load(rootname)
    matrix = tensor.numpy()
    # average = get_average_matrix(matrix)
    average = matrix
    return average, labels

o.simulated_annealing
file_list = glob('raw_predictions/seen/*.pt')
matrix, labels = load_data(file_list[2])
v.plot_cm(cm=matrix, labels=labels, output="temp.png")
results = o.simulated_annealing(current_cm=matrix, current_perm=None)
v.plot_cm(cm=results.cm, labels=[labels[idx] for idx in results.perm], output="temp2.png")
