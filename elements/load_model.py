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
from third_party.model_wrappers.yolov5_wrapper import YoloV5
from third_party.model_wrappers.unet import UNet



# def load_model():
# def create_channel_head():
# def create_model_skl():
# def load_dict_pt():

def load_model_pt(model: torch.nn.Module, path: str, strict: bool = True, force: bool = False) -> dict[str, Any]:
    """
    This method loads a model from path into the model object.

    :param model: a PyTorch model object with empty/random weights.
        Note: The weights from path are written this input parameter.
    :param path: path to the model saved with :meth:~`elements.basic.save_model.save_model_pt`.
    :param strict: ignores the weights for the head of the model if for example the num_classes don't match in the pretrained weights and the passed model
    :param force: if True handle potential mismatches by skipping incompatible layers while loading in the weights.
    :returns: the raw dictionary contents from path.

    >>> from common.elements.legacy.dataset import Dataset, DatasetInfo, get_dataset_info
    >>> from common.elements.model.torch_models.lenet import LeNet
    >>> from collections import OrderedDict
    >>> d = load_model_pt(LeNet(), get_dataset_info(Dataset.CIFAR10, DatasetInfo.MODEL))
    >>> isinstance(d['model'], OrderedDict)
    True
    """
    # Load the checkpoint data
    data = torch.load(path, map_location=torch.device('cpu'))

    # Ensure the checkpoint contains the required keys
    if 'model' not in data:
        logger.warning("The checkpoint file is missing the 'model' key.")
        model_weights = data
    else:
        model_weights = data['model']

    if 'type' not in data:
        logger.warning("The checkpoint file does not contain a 'type' key. Skipping type validation.")

    # Handle forced loading with dynamic mismatch resolution (https://stackoverflow.com/questions/67838192/size-mismatch-runtime-error-when-trying-to-load-a-pytorch-model)
    if force:
        dynamic_load_weights_pt(model, weights=data)
        logger.info(f"Loaded model with dynamic mismatch resolution from {os.path.abspath(path)}.")
    elif 'type' in data and not str(type(model)) == data['type']:
        # Validate model type if 'type' key exists
        logger.warning(f"The type in the file ({data['type']}) is different from the passed model ({type(model)}).")
        model.load_state_dict(model_weights, strict=strict)
        logger.info(f"Loaded model from {os.path.abspath(path)} with potential mismatches (strict={strict}).")
    else:
        # Normal weight loading
        model.load_state_dict(model_weights, strict=strict)
        logger.info(f"Loaded model from {os.path.abspath(path)} with strict={strict}.")

    return data
def load_model_skl(path: str):
    """
    Read an SciKit learn model from path.

    :param path: the path to the model.
    :returns: the 'model' key of the data.

    >>> with open("mymodel.skl", "wb") as f:
    ...   pickle.dump({"model": ["model x"], "metric": 1.0}, f)
    >>> load_model_skl("mymodel.skl")
    ['model x']
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
        model = data['model']
        logger.info(f"Loaded model from {os.path.abspath(path)} with metric {data['metric']}")
        return model
# def create_cifar10_lenet_pt():
# def create_mnist_lenet_pt():
# def create_projectionnet_pt():
# def create_prototypical_net_pt():
# def create_resnet18_pt():
# def create_siamese_net_pt():
# def create_vgg19_pt():
# def vgg19_gradcam_class_heatmap():
# def vgg19_single_image_classifier():
# def create_deim_pt():
# def create_detr_pt():
# def create_efficientdet_pt():
# def create_faster_rcnn_pt():
def create_yolov5_pt(num_classes: int, model_name='yolov5s', num_chans=3, device='cuda:0', pretrained=False) -> YoloV5:
    """
    Create a YoloV5 model

    :param num_classes: the number of classes.
    :param model_name: the name of the Yolo model.
    :param num_chans: the number of channels in the images.
    :param device: the device to run the model on.
    :param pretrained: True if pretrained on COCO dataset should be loaded.
    :return: model

    >>> model = create_yolov5_pt(num_classes=2)
    >>> isinstance(model, YoloV5)
    True
    """
    return YoloV5(model_name, num_classes, num_chans, device, pretrained)
# def create_yolov5_pt():
# def load_deim_pt():
# def create_keypoint_rcnn():
# def create_deepresunet_pt():
# def create_fewshot_matchingnet_spp_pt():
# def create_fewshot_resnet_pt():
# def create_plasticnet_pt():
# def create_resnetunet_pt():
# def create_ternausnet():
# def create_unet_latent_pt():
def create_unet_pt(num_classes: int, num_channels: int, depth=5, start_filters=64) -> UNet:
    """
    Create a UNet model

    :param num_classes: the number of classes to use for this model
    :param num_channels: the number of channels to use for this model
    :param depth: depth of the 'U' of the Unet
    :param start_filters: initial amount of filters
    :return: a UNet model

    >>> type(create_unet_pt(5, 3))
    <class 'third_party.model_wrappers.unet.UNet'>
    """
    model = UNet(num_classes=num_classes, in_channels=num_channels, depth=depth, start_filts=start_filters)
    return model
# def create_auto_enc_latent_pt():
# def create_auto_enc_latent_pt2():
# def create_kmeans():
# def load_distribution_padim():
# def load_embedding_indices():
# def load_fitted_features():
# def load_multi_variate_gaussian_model():
# def load_patchcore_model():
# def create_centroidnet_pt():
    