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
from sklearn.svm import SVC
from skimage.feature import hog
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from common.elements.legacy.dataset import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever
from common.elements.legacy.classification import ClassifierType


# def preprocess():
# def fill_holes():
# def add_dimension():
# def assign_hs_bins():
# def calc_mean_per_class():
# def create_mask_from_boxes():
# def flatten_images_to_vector():
# def hwc_to_chw():
# def normalize_and_clip():
# def normalize_and_clip_pt():
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale

    :param img: an RGB image
    :return: a grayscale image with shape [h, w]

    >>> img = np.zeros((100, 200, 3))
    >>> img = rgb_to_gray(img)
    >>> img.shape
    (100, 200)
    """

    img_rgb = color.rgb2gray(img)
    return img_rgb
# def sample_wrt_bins():
# def standard_scale():
# def train_test_split_np():
def auto_threshold_image_cv(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Automatically threshold an image of type dtype.uint8 to a binary image using the OTSU method (see opencv documentation)

    :param image: input image in format [HxW]
    :param invert: invert the images after thresholding
    :return: thresholded
    :example: Threshold a mono-channel image

    >>> image = np.ones([5, 5], dtype=np.uint8)
    >>> image[1:4, 1:4] = 123
    >>> image
    array([[  1,   1,   1,   1,   1],
           [  1, 123, 123, 123,   1],
           [  1, 123, 123, 123,   1],
           [  1, 123, 123, 123,   1],
           [  1,   1,   1,   1,   1]], dtype=uint8)
    >>> auto_threshold_image_cv(image)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """
    image = image.astype(np.uint8)
    # Threshold the image using the OTSU auto threshold..
    _, bin_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        # Invert the binary image to make objects have the value 1.
        bin_image = np.logical_not(bin_image).astype(np.uint8)
    return bin_image
# def auto_threshold_images_cv():
# def get_auto_crop():
def resize_images_cv(images: list[np.ndarray], height: int, width: int) -> list[np.ndarray]:
    """
    Resize all image in the list of images to the supplied height, width

    :param images: list of input images
    :param height: the desired height
    :param width: the desired width
    :return: the list of resized images
    :example: Resize some three-channel images

    >>> images = [np.ones([10, 10, 3]),
    ...           np.ones([15, 15, 3])]
    >>> r = resize_images_cv(images, width=5, height=5)
    >>> r[0].shape
    (5, 5, 3)
    >>> r[1].shape
    (5, 5, 3)

    """
    return [cv2.resize(image, (width, height)) for image in images]
# def threshold_image_cv():
# def preprocess_pca():
# def clip_and_filter_annot():
# def create_tiles():
# def generate_fixed_tiles():
# def generate_random_tiles():
# def generate_random_tiles_annot():
# def generate_random_tiles_in_annot():
# def get_num_tiles():
# def get_padded_hw():
# def get_padding():
# def get_padding_tl():
# def pad_annotations():
# def pad_image():
# def tile_idx_to_coord():
# def tile_idx_to_ctr():
# def tile_idx_to_inner_rect():
# def tile_idx_to_rect():
# def unpad_boxes():
# def convolution_cv():
# def game_of_life():
# def maximum_filter_cv():
# def minimum_filter_cv():
# def canny_cv():
# def prewitt_cv():
# def sobel_cv():
# def corner_harris_cv():
# def difference_of_gaussian_cv():
# def laplacian_of_gaussian_cv():
# def hough_circle_transform_cv():
# def hough_line_transform_cv():
# def histogram_of_oriented_gradients():
# def sift_cv():
# def clip_peaks_pt():
# def cube_to_hh_pt():
# def cube_to_hhs_pt():
# def cube_to_hhsi_pt():
# def cube_to_hhsi_pt2():
# def cube_to_logderiv():
# def cube_to_pixels():
# def demosaic_np():
# def flatfield_line_pt():
# def flatfield_np():
# def pixels_to_cube_np():
# def pixels_to_cube_pt():
# def torch_to_hc():
# def torch_to_hhsi():
# def apply_supervised_reduction():
# def apply_t_sne_2d():
# def apply_unsupervised_reduction():
# def segmap_to_boxes():
# def add_blackout_noise_pt():