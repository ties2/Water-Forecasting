import numpy as np
from typing import Type, List, Any, Iterable, Union, Set, Optional, Callable
from PIL import Image
import statistics
import requests
import io
import cv2
import unittest
import os
import abc
from abc import ABC
import pandas as pd
import shutil
import glob
import copy
import pickle #joblib is better
import albumentations
import seaborn as sns
import matplotlib.pyplot as plt
from albumentations import augmentations
from enum import Enum
from skimage import data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from common.elements.utils import get_logger, deprecated, DevNull, get_tmp_dir
from common.data.datasets_info import ABCDatasetInfo
from common.data.generic_dataset import GenericDataset, get_generic_dataset, get_generic_dataloader
from common.elements.model.basic import dynamic_load_weights_pt
from common.elements.visualize import hyperspectral_to_rgb
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any
logger = get_logger('notebook_logs')

# def fit_model():
def fit_model_skl(model, input: Union[np.ndarray, list[np.ndarray]], target: Optional[np.ndarray]):
    """
    Fits a model using Scikit Learn

    :param model: the sklearn model
    :param input: the input to the model
    :param target: the targets of the model
    :return: the fitted model

    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> clf = svm.SVC()
    >>> X, y = datasets.load_iris(return_X_y=True)
    >>> type(clf.fit(X, y))
    <class 'sklearn.svm._classes.SVC'>
    """
    assert isinstance(input, np.ndarray) or isinstance(input, list)
    if target is not None:
        assert isinstance(target, np.ndarray)
    assert hasattr(model, 'fit')
    return model.fit(input, target)
# def calc_covariance_matrix_embedding_vectors():
# def calc_covariance_matrix_embeddings():
# def fit_multi_variate_gaussian_model():
# def fit_patchcore_model():