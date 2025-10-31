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

# def tune_params():
# def add_solution_to_evolver():
# def create_genetic_evolver():
# def get_lambda_lr_scheduler_pt():
def get_reduce_lr_on_plateau_pt(optimizer: torch.optim.Optimizer, patience: int = 5, factor: float = 0.1) -> Any:
    """
    Get the reduce learnrate on plateau scheduler. See PyTorch docs for more info on this scheduler.

    :param optimizer: optimizer to connect the scheduler to.
    :param patience: patience parameter of the scheduler.
    :param factor: factor parameter of the scheduler.
    :return: the scheduler object.

    >>> import torch.optim
    >>> import torchvision.models.alexnet
    >>> scheduler = get_reduce_lr_on_plateau_pt(torch.optim.Adam(torchvision.models.alexnet(pretrained=False).parameters(), lr=0.001))
    >>> isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    True
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
# def get_step_lr_pt():
# def run_evolver():
# def tune_lambda_learning_rate_pt():
def tune_learning_rate_pt(epoch_loss: float, scheduler, epoch=None):
    """
    Progress the PyTorch scheduler one step.

    :param epoch_loss: the loss at the current epoch.
    :param scheduler: the scheduler.
    :param epoch: the current epoch.

    >>> import torch.optim
    >>> l = torch.tensor([5.], requires_grad=True)
    >>> o = torch.optim.Adam([l], lr=0.001)
    >>> s = torch.optim.lr_scheduler.ReduceLROnPlateau(o, patience=5, factor=0.1)
    >>> tune_learning_rate_pt(l.item(), s)
    >>> round(l.item(), 1)
    5.0
    """
    scheduler.step(epoch_loss, epoch)
