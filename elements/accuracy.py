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
from common import get_logger, deprecated, DevNull
from common.data.datasets_info import ABCDatasetInfo
from common.data.generic_dataset import GenericDataset, get_generic_dataset, get_generic_dataloader
from common.elements.model.basic import dynamic_load_weights_pt
from common.elements.visualize import hyperspectral_to_rgb
logger = get_logger('notebook_logs')

from sklearn.svm import SVC
from skimage.feature import hog
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from common.elements.legacy.dataset import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever
from common.elements.legacy.classification import ClassifierType

def calc_accuracy(output: Iterable[Any], target: Iterable[Any], normalize: bool = True) -> float:
    """

    Calculate the accuracy based on the output and target.

    :param output: output of the model.
    :param target: list of classes of the ground-truth for each sample.
    :param normalize: normalize by the total number of sample.
    :return: accuracy

    :example:

    >>> calc_accuracy(['cat', 'cat', 'dog', 'dog'], ['cat', 'cat', 'cat', 'dog'])
    0.75
    >>> calc_accuracy(['cat', 'cat', 'dog', 'dog'], ['cat', 'cat', 'dog', 'dog'])
    1.0
    """
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    return accuracy_score(y_true=target, y_pred=output, normalize=normalize)