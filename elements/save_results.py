import numpy as np
from typing import Type, List, Any
from PIL import Image
import statistics
import requests
import io as b_io
import cv2
import unittest
import PIL
import sys
import os
import abc
from abc import ABC
import pandas
import pandas as pd
import shutil
import glob
import copy
import pickle
import albumentations
import seaborn as sns
import matplotlib.pyplot as plt
from albumentations import augmentations
from enum import Enum
from skimage import data
from typing import Iterable, Union, Set, Any, Optional, Any, Callable, Sized
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
NoneType = type(None)
import common
from common import get_logger, deprecated, DevNull, newline
from common.data.datasets_info import ABCDatasetInfo
from common.data.generic_dataset import GenericDataset, get_generic_dataset, get_generic_dataloader
from common.elements.model.basic import dynamic_load_weights_pt
from common.elements.visualize import hyperspectral_to_rgb,create_tb
logger = get_logger('notebook_logs')

from sklearn.svm import SVC
from skimage.feature import hog
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from common.elements.legacy.dataset import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever
from common.elements.legacy.classification import ClassifierType
from common import newline

def save_results(classifier_name: str, results_dict: dict[str, Any]):

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{classifier_name.replace(' ', '_')}_results.txt"
    with open(filename, 'w') as f:
        f.write(f"==================================================\n")
        f.write(f"{classifier_name} Performance:\n")
        f.write(f"==================================================\n")
        for key, value in results_dict.items():
            # Format times to two decimal places, and other floats to four
            if '_time' in key:
                f.write(f"{key.replace('_', ' ').title()}: {value:.2f}s\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
    print(f"Results for {classifier_name} saved to {filename}")

# def arrays_to_npz():
def save_boxes_to_textfile(boxes: dict[str, list[Iterable[Any]] | Iterable], filename: str, digits=4):
    """
    Save a dictionary of bounding boxes to a textfile seperated by spaces. Note: the filename should not contain any
    spaces.

    :param boxes: dictionary of a list of boxes. The keys represent a unique image id, and the boxes are in format
        [[y1, x1, y2, x2, class_id, class_prob], ...]
    :param filename: 
        The filename to save to boxes to. All boxes are written to single line starting with the key of the dict
        ::

            id y1 x1 y2 x2 class_id class_prob y1 x1 y2 x2 class_id class_prob ... \\newline
            id y1 x1 y2 x2 class_id class_prob y1 x1 y2 x2 class_id class_prob ... \\newline
            ...
    :param digits: round class_prob to this number of digits.

    >>> save_boxes_to_textfile({"image_id1": [[0, 0, 10, 10, 0, 0.9],
    ...                                       [5, 5, 15, 15, 1, 0.8]],
    ...                         "image_id2": [[7, 2, 14, 10, 1, 0.7],
    ...                                       [0, 0, 15, 15, 0, 0.6]]}, "boxes_test.txt")
    >>> lines = open("boxes_test.txt").readlines()
    >>> [str(line).strip() for line in lines]
    ['image_id1 0 0 10 10 0 0.9 5 5 15 15 1 0.8', 'image_id2 7 2 14 10 1 0.7 0 0 15 15 0 0.6']
    """
    assert filename.find(" ") == -1, "Filename should not contain space characters"
    dirname = os.path.dirname(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        lines = []
        for id, boxs in boxes.items():
            line = [id]
            for y1, x1, y2, x2, class_id, class_prob in boxs:
                line.extend([str(int(y1)), str(int(x1)), str(int(y2)), str(int(x2)), str(int(class_id)), str(round(float(class_prob), digits))])
            lines.append(" ".join(line) + newline)
        f.writelines(lines)
def save_class_names_to_textfile(class_names: list, filename: str):
    """
    This saves all strings in class_names to filename

    :param class_names: a list of classnames
    :param filename: a filename

    >>> save_class_names_to_textfile(['car', 'bike', 'truck'], "predictions.txt")
    >>> lines = open("predictions.txt").readlines()
    >>> [str(line).strip() for line in lines]
    ['car', 'bike', 'truck']
    """
    dirname = os.path.dirname(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        [f.write(str(c) + '\n') for c in class_names]
# def save_clusters_textfile():
def save_dict_to_file(list_dict: dict[str, str], filename: str):
    """
    Saves a list of dictionaries to a file.
    Each item in the list is a seperate line in the file.
    The key values are separated by a semicolon ';'.

    :param list_dict: the list of dictionaries to save
    :param filename: the filename to save to

    >>> data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    >>> save_dict_to_file(data, 'test.txt')
    >>> [s.strip() for s in open("test.txt", "r").readlines()]
    ['key1;value1', 'key2;value2', 'key3;value3']
    """
    lines = []
    for key, value in list_dict.items():
        lines.append(f"{key};{value}\n")

    with open(filename, "w") as f:
        f.writelines(lines)
def save_image(img: np.ndarray, filename: str):
    """
    Save an image to filename

    :param img: image numpy array with shape [HWC]
    :param filename: filename to store

    >>> import os
    >>> img = np.empty([5, 5, 3], dtype=np.uint8)
    >>> save_image(img, "test.png")
    >>> os.path.isfile("test.png")
    True
    """
    cv2.imwrite(filename, img)
# def save_image_ski():
# def save_key_value_list_to_csv():
# def save_numpy_array():
# def save_per_class_stats_to_textfile():
# def recombine_tiles():
# def untile_boxes():