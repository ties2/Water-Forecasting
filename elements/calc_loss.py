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

import functools
from albumentations import Resize
from sklearn.metrics import confusion_matrix, accuracy_score
from common import reproduce_seed
from common.data.datatypes import ClassMask, SampleContainer
from common.data.datasets_info import supervised_instance_segmentation_datasets
from common.elements.utils import get_tmp_dir, static_var, wait_forever
from common.data.generic_dataset import get_generic_dataloader
from common.data.transforms import StackSC, ToTensorSC
from common.data.transforms.postprocess import TransposeClassMaskSC
from common.data.transforms import ClassMaskOneHotSC, BoundingBoxesToNumpySC
from common.data.transforms import NormaliseImageSC
from common.data.loaders.annotation import LoadClassMasksFileSC
from common.data.loaders import LoadImageSC, LoadImageSCBinary
from third_party.model_wrappers.unet import UNet
from third_party.model_wrappers.yolov5_wrapper import YoloV5
#
# def calc_loss()
# def calc_auto_encoding_loss_pt()
# def calc_efficiency_loss_pt()
# def calc_loss_pt()
# def calc_segmentation_loss_dice_pt()
def calc_segmentation_loss_pt(target_data: Union[torch.Tensor, list[dict]], loss=nn.CrossEntropyLoss,do_argmax=True, weights=False) -> torch.Tensor:
    """
    Calculates the loss between output_data and target_data for segmentation models (Unet, DeepLab, etc.).

    :param output_data: output data of the model (tensor of shape [b, nclasses, h, w] with class logits
        or a list of dictionaries containing the key 'class_masks' with the one-hot-encoded output: [b, nclasses, h, w].
    :param target_data: input data of the model (tensor of shape [b, 1, h, w] with class ids
        or a list of dictionaries containing the key 'class_masks' with the one-hot-encoded target: [b, nclasses, h, w].
    :param do_argmax: whether to return the argmax of the logits
    :param weights: optional user-given weights for the loss function

    :param loss: instance of a loss function
    :return: calculated loss

    >>> calc_segmentation_loss_pt(torch.Tensor([10]), torch.Tensor([15]), loss=lambda x, y: x - y)
    tensor([-5.])

    """
    # MSELoss requires target_data to stay one-hot-encoded
    if loss == nn.MSELoss:
        do_argmax = False

    # If a list is passed as target assume this is the list of dict target.
    if isinstance(output_data, list) or isinstance(output_data, tuple):
        output_data = torch.stack([t["class_masks"] for t in output_data])

    if isinstance(target_data, list) or isinstance(target_data, tuple):
        target_data = torch.stack([t["class_masks"] for t in target_data])

    if len(target_data.shape) > 1 and target_data.shape[1] > 1 and do_argmax:
        target_data = torch.argmax(target_data, dim=1, keepdim=False)
        target_data = target_data.long()
    else:
        target_data = target_data.float()

    if isinstance(loss, type):
        if weights:
            _, nc, h, w = output_data.size()
            ws = [1, h * w - (nc - 1), h * w - (nc - 1), h * w - (nc - 1), h * w - (nc - 1)]
            loss = loss(weight=torch.FloatTensor(ws).cuda())
        else:
            loss = loss()

    return loss(output_data, target_data)

# def calc_ssim_loss()
# def calc_centroid_loss_pt()
# def calc_dreambooth_loos_pt()
# def calc_loss_keypoints_rcnn()
def calc_yolov5_loss_pt(p: tuple[Any, ...], targets: torch.Tensor, model: YoloV5, image_height, image_width) -> torch.Tensor:
    """
    This calculates the loss for the YOLOv5 model

    Note: The convention is y,x but some models use class x_center y_center width height so here we convert them.
    Also, this implementation of YoloV3 requires the annotations to be scaled between 0 and 1
    Also add an id to each entry in targets

    :param p: predictions output from the model
    :param targets: targets from the ground-truth (loaded using the dataloader)
    :param model: the model
    :param image_height: the image height
    :param image_width: the image width
    :return:

    >>> from common import reproduce_seed
    >>> from third_party.model_wrappers.yolov5_wrapper import YoloV5
    >>> reproduce_seed(42)
    >>> model = YoloV5('yolov5s', num_classes=2)
    >>> input = torch.zeros((2, 3, 256, 256)).cuda()
    >>> predictions = model(input)
    >>> target = torch.zeros((2, 10, 5)).cuda()
    >>> loss = calc_yolov5_loss_pt(predictions, target, model, 256, 256)
    >>> isinstance( loss.cpu().item(), float)
    True
    """

    # convert pipeline targets: [y1, x1, y2, x2, c] to yolov5 targets: [i, class, ncx, ncy, nw, nh]
    targets2 = torch.tensor((), dtype=targets.dtype)
    targets2 = targets2.new_zeros(len(targets) * len(targets[0]), 6).to(targets.device)
    count = 0
    for i, target in enumerate(targets):
        for j, box in enumerate(target):
            if not torch.all(box.eq(torch.tensor([-1., -1., -1., -1., -1.]).to(targets.device))):  # skip padding tensors
                targets2[count, 0] = i  # sample_id
                targets2[count, 1] = box[4]  # class_id
                ymin = box[0]
                xmin = box[1]
                ymax = box[2]
                xmax = box[3]
                targets2[count, 2] = torch.round((xmin + xmax) / 2) / image_width  # x_center
                targets2[count, 3] = torch.round((ymin + ymax) / 2) / image_height  # y_center
                targets2[count, 4] = (xmax - xmin) / image_width  # width
                targets2[count, 5] = (ymax - ymin) / image_height  # height
                count += 1
    return model.loss(p, targets2[0:count])[0]
