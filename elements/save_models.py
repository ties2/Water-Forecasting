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




# def save_model():
# def save_dict_pt():
def save_model_pt(model: nn.Module, path: str, loss: float = 0, epoch: float = 0, sub=0., div=1.):
    """
    Save a model to a path.
    Note: the model is saved as a dictionary where the key 'model' contains the model's state_dict.

    :param model: the model object to be saved.
    :param path: the path to save the model.
    :param loss: the loss of the model.
    :param epoch: the number of epochs elapsed when the model was saved.
    :param sub: the normalization value used to train the model. x' = (x - sub) / div
    :param div: the normalization value used to train the model. x' = (x - sub) / div


    >>> from common.elements.model.torch_models.lenet import LeNet
    >>> model = LeNet()
    >>> save_model_pt(model, "test.pth")
    >>> os.path.isfile("test.pth")
    True
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Save model to {os.path.abspath(path)}")
    save_dict = {'model': model.state_dict(), 'loss': loss, 'epoch': epoch, 'type': str(type(model)), 'sub': sub,
                 'div': div}
    torch.save(save_dict, path)
def save_model_skl(model, path: str, metric: float = 0):
    """
    Save a model to a path.
    Note: the model is saved as a dictionary where the key 'model' contains the SciKit learn model.

    :param model: the model object to be saved.
    :param path: the path to save the model.
    :param metric: some performance metric of the model.

    >>> from sklearn.cluster import KMeans
    >>> model = KMeans()
    >>> save_model_skl(model, "test.pth", 0.8)
    >>> os.path.isfile("test.pth")
    True
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Save model to {os.path.abspath(path)}")

    save_dict = {'model': model, 'metric': metric, 'type': str(type(model))}
    with open(path, "wb") as f:
        pickle.dump(save_dict, f)
# def save_pipeline_hf():
# def get_checkpoint_info_hf():
# def register_stable_diffusion_save_pre_hooks_hf():
# def save_diffusion_checkpoint_as_pipeline_hf():
# def save_state_ac():