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
#
# def optimize():
def back_prop_pt(loss: torch.Tensor, optimizer: torch.optim.Optimizer, model=None, clip_gradients=False,clip_value=0.1):
    """
    Perform back propagation on a model using the loss and an optimizer.

    :param loss: the loss output of the model (This should contain the whole lint through all parameters of the model).
    :param optimizer: the optimizer to use for backprop.
    :param model: the model that is being optimized.
    :param clip_gradients: set this to true to clip large gradientto prevent oscilations caused by rapid weight changes.
    :param clip_value: the value to use for clipping.

    >>> import torch
    >>> l = torch.tensor(5., requires_grad=True)
    >>> o = torch.optim.Adam([l], lr=3.)
    >>> back_prop_pt(l, o)
    >>> round(l.item(), 1)
    2.0
    """
    if clip_gradients:
        assert model is not None, "When clip gradients is True, the model should also be passed to back_prop_pt"

    loss.backward()
    if clip_gradients:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
def get_adam_optimizer_pt(model: Union[torch.nn.Module, dict[str, torch.Tensor]], learnrate: float = 0.001,weight_decay: float = 0) -> Union[torch.optim.Adam, torch.optim.AdamW]:
    """

    :param model: the model to be optimized by the model
    :param learnrate: the learning rate
    :param weight_decay: the weight decay for regularization
    :return: an instance of the ADAM optimizer.

    >>> from common.elements.model.torch_models.lenet import LeNet
    >>> o = get_adam_optimizer_pt(LeNet())
    >>> isinstance(o, torch.optim.AdamW) or isinstance(o, torch.optim.Adam)
    True
    """
    if hasattr(model, "parameters"):
        return torch.optim.AdamW(model.parameters(), lr=learnrate, weight_decay=weight_decay)
    elif hasattr(model, "values"):
        return torch.optim.Adam(model.values(), lr=learnrate, weight_decay=weight_decay)
    else:
        raise ValueError("Could not understand model.")
# def get_adamw_optimizer_pt():
# def get_sgd_optimizer_pt():