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

from skimage import color
from scipy.ndimage import binary_fill_holes
from common.elements.legacy.detection.dataset import LabelImgDataset
from common.elements.legacy.detection.transforms import Normalizer, ToTensor, PaddingAndStacking, Stacking, Resizer
from common.elements.legacy.detection.collator import FixedTileCollator, RandomPositivesTilesCollator, FullCollator
from common.elements.legacy import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever



def fill_holes(img: np.ndarray) -> np.ndarray:
    """
    Fill holes of given image
    :param img: input image
    :return:

    >>> img = np.ones((10, 10))
    >>> img[4:6,4:6] = 0
    >>> filled = fill_holes(img=img)
    >>> np.sum(filled)
    100


    """
    return binary_fill_holes(img).astype('uint8')