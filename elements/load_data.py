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
from common.data.loaders import LoadImageSC,LoadBoundingBoxesSC
from common.elements.model.basic import dynamic_load_weights_pt
from common.elements.visualize import hyperspectral_to_rgb
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any
from skimage import color
from scipy.ndimage import binary_fill_holes
from common.elements.legacy.detection.dataset import LabelImgDataset
from common.elements.legacy.detection.transforms import Normalizer, ToTensor, PaddingAndStacking, Stacking, Resizer
from common.elements.legacy.detection.collator import FixedTileCollator, RandomPositivesTilesCollator, FullCollator
from common.elements.legacy import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever
from common.data.transforms import StackSC, ToTensorSC, NormaliseImageSC,RemapLabels,BoundingBoxesToNumpySC

logger = get_logger('notebook_logs') #This line creates a logger object specifically for notebook logging purposes.

class DataPipeline:
    """Comprehensive data loading and preprocessing pipeline"""

    def __init__(self, dataset_name: str = 'CIFAR10', root: str = './data'):
        self.dataset_name = dataset_name
        self.root = root
        self.classes = None
        self.scaler = None
        self.pca = None

    def load_dataset(self,
                     train_size: int = 40000,
                     val_size: int = 10000,
                     normalize: bool = True,
                     scaling_method: str = 'standard',
                     apply_pca: bool = False,
                     pca_components: Optional[int] = None,
                     subsample: Optional[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Load and preprocess dataset with various options
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            normalize: Whether to normalize the data
            scaling_method: 'standard', 'minmax', or 'robust'
            apply_pca: Whether to apply PCA dimensionality reduction
            pca_components: Number of PCA components (None for automatic)
            subsample: If specified, subsample the data for faster experimentation
        Returns:
            (train_x, train_y), (val_x, val_y), (test_x, test_y)
        """
        print(f"Loading {self.dataset_name} dataset...")

        # Load dataset based on name
        if self.dataset_name == 'CIFAR10':
            transform = transforms.ToTensor()
            full_train_dataset = datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=transform
            )
            self.classes = full_train_dataset.classes

        elif self.dataset_name == 'MNIST':
            transform = transforms.ToTensor()
            full_train_dataset = datasets.MNIST(
                root=self.root, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root=self.root, train=False, download=True, transform=transform
            )
            self.classes = [str(i) for i in range(10)]

        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        # Split train into train/val
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Convert to numpy arrays
        print("Converting to numpy arrays...")
        train_x, train_y = self._dataset_to_numpy(train_dataset)
        val_x, val_y = self._dataset_to_numpy(val_dataset)
        test_x, test_y = self._dataset_to_numpy(test_dataset)

        # Subsample if requested (for faster experimentation)
        if subsample:
            print(f"Subsampling to {subsample} samples per set...")
            train_x, train_y = self._subsample(train_x, train_y, subsample)
            val_x, val_y = self._subsample(val_x, val_y, min(subsample // 4, len(val_x)))
            test_x, test_y = self._subsample(test_x, test_y, min(subsample // 4, len(test_x)))

        # Normalize/Scale the data
        if normalize:
            print(f"Applying {scaling_method} scaling...")
            train_x, val_x, test_x = self._normalize_data(
                train_x, val_x, test_x, method=scaling_method
            )

        # Apply PCA if requested
        if apply_pca:
            print(f"Applying PCA (components={pca_components or 'auto'})...")
            train_x, val_x, test_x = self._apply_pca(
                train_x, val_x, test_x, n_components=pca_components
            )

        # Print dataset info
        self._print_dataset_info(train_x, val_x, test_x)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def _dataset_to_numpy(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Convert PyTorch dataset to numpy arrays"""
        X, y = [], []
        for img, label in dataset:
            X.append(img.numpy().flatten())
            y.append(label)
        return np.array(X), np.array(y)

    def _subsample(self, X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample the data"""
        if n >= len(X):
            return X, y
        indices = np.random.choice(len(X), n, replace=False)
        return X[indices], y[indices]

    def _normalize_data(self, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray,
                        method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the data using specified method"""

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        train_x = self.scaler.fit_transform(train_x)
        val_x = self.scaler.transform(val_x)
        test_x = self.scaler.transform(test_x)

        return train_x, val_x, test_x

    def _apply_pca(self, train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray,
                   n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply PCA dimensionality reduction"""

        if n_components is None:
            # Keep 95% of variance
            self.pca = PCA(n_components=0.95)
        else:
            self.pca = PCA(n_components=n_components)

        train_x = self.pca.fit_transform(train_x)
        val_x = self.pca.transform(val_x)
        test_x = self.pca.transform(test_x)

        print(f"PCA reduced dimensions from {self.pca.n_features_in_} to {self.pca.n_components_}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return train_X, val_X, test_X

    def _print_dataset_info(self, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray):
        """Print dataset information"""
        print("\n" + "=" * 50)
        print("Dataset Information:")
        print("=" * 50)
        print(f"Classes: {self.classes}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Train set: {train_x.shape}")
        print(f"Validation set: {val_x.shape}")
        print(f"Test set: {test_x.shape}")
        print(f"Feature dimension: {train_x.shape[1]}")
        print("=" * 50 + "\n")

    def visualize_samples(self, x: np.ndarray, y: np.ndarray, n_samples: int = 5,
                          original_shape: Optional[tuple] = None):
        """Visualize random samples from the dataset"""

        if original_shape is None:
            if self.dataset_name == 'CIFAR10':
                original_shape = (3, 32, 32)
            elif self.dataset_name == 'MNIST':
                original_shape = (1, 28, 28)

        indices = np.random.choice(len(x), min(n_samples, len(x)), replace=False)

        fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
        if len(indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            # Reshape and denormalize if needed
            img = X[idx]
            if self.scaler:
                img = self.scaler.inverse_transform(img.reshape(1, -1)).flatten()

            # Reshape to original image shape
            if self.pca is None:
                img = img.reshape(original_shape)
                if original_shape[0] == 3:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                else:  # Grayscale
                    img = img.squeeze()
            else:
                # Can't reconstruct image after PCA
                ax.text(0.5, 0.5, f"Class: {self.classes[y[idx]]}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # Clip values to [0, 1] for display
            img = np.clip(img, 0, 1)

            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f"Class: {self.classes[y[idx]]}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_class_distribution(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Plot the distribution of classes in each dataset split"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (y, title) in zip(axes, [(y_train, 'Train'), (y_val, 'Validation'), (y_test, 'Test')]):
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'{title} Set Class Distribution')
            ax.set_xticks(unique)
            if self.classes:
                ax.set_xticklabels([self.classes[i] for i in unique], rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
# def load_data():
# def annots_are_empty():
# def convert_mnist_labels_to_classname():
def get_enabled_classes(loader, preprocesses: list):
    """
    Filters the classes from the dataloader using a RemapLabel object in the preprocess list. In RemapLabel if the value of a classname in the mapping is set to None, it won´t be included in the dataloader.

    :param loader: DataLoader object
    :param preprocesses: list of preprocessing tasks
    :return: a tuple with a list of classes and the number of classes
    """
    # Determine which classes are used and how many
    classes = None
    if any(isinstance(x, RemapLabels) for x in preprocesses):
        for x in preprocesses:
            if isinstance(x, RemapLabels):
                classes = [y for y in list(x.get_mapping().keys()) if x.get_mapping()[y] is not None]
                break
    else:
        classes = loader.dataset.get_class_names()
    num_classes = len(classes)
    return classes, num_classes
def get_labelimg_dataloader_pt(images_folder: str, annotations_folder: Optional[str] = None, batch_size: int = 1,fixed_height: Optional[int] = None, fixed_width: Optional[int] = None,pad_boxes=True, shuffle=True, normalize=True, norm_sub=128, norm_div=255,additional_transforms: Optional[list] = None) -> torch.utils.data.DataLoader:
        """
        Get a PyTorch dataloader for images that have been annotated by LabelImg.
        Note: the filenames of the images and annotations should correspond.

        :param images_folder: folder containing the images
        :param annotations_folder: folder containing the XML annotation files
        :param batch_size: the batch size
        :param fixed_height: each image is resized to this size
        :param fixed_width: each image is resized to this size
        :param pad_boxes: determines if the bounding boxes should be padded so they fit into a single tensor for the minibatch.
        :param shuffle: determines if samples needs to be shuffled
        :param normalize: determines if normalization is needed
        :param norm_sub: default subtract for normalization
        :param norm_div: default divide for normalization
        :param additional_transforms: additional custom transformations applied to the data.
            Note: These are put before the conversion to torch Tensor and therefor should accept numpy arrays.
        :return: an instance of :class:`common.detection.dataset.LabelImgDataset`

        :example: Add all images from Orbs dataset and check the return type.

        >>> from common.elements.legacy.dataset import Dataset, DatasetInfo, get_dataset_info
        >>> loader = get_labelimg_dataloader_pt(get_dataset_info(Dataset.ORBS_TRAINING, DatasetInfo.INPUT_DATA), get_dataset_info(Dataset.ORBS_TRAINING, DatasetInfo.ANNOTATION))
        >>> isinstance(loader, torch.utils.data.DataLoader)
        True
        """
        t = []
        if fixed_height is not None and fixed_width is not None:
            t.append(Resizer(height=fixed_height, width=fixed_width))
        if normalize:
            t.append(Normalizer(mean=norm_sub, std=norm_div))
        if additional_transforms is not None:
            t.extend(additional_transforms)
        t.append(ToTensor())

        dataset = LabelImgDataset(images_folder, annotations_folder, transforms=torchvision.transforms.Compose(t))

        collator_transforms = [PaddingAndStacking() if pad_boxes else Stacking()]
        collator = FullCollator(transforms=transforms.Compose(collator_transforms))

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collator)
        return dataloader
def get_mnist_torchvision_datasets(img_dir: str, n_samples: Union[int, float] = -1, download: bool = False, transform: torchvision.transforms = None) -> tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """
    Get the training and testing MNIST dataset from torchvision.

    :param img_dir: location of the MNIST dataset
    :param transform: transforms to perform on the dataset
    :param n_samples: number of samples. If integer select randomly n_samples from trainingset, if float select ratio of trainingset.
            (can be used to reduce amount of training data)
    :param download: download dataset from source
    :return: training and testing MNIST dataset from torchvision
    """

    trainset = torchvision.datasets.MNIST(root=img_dir, train=True, download=download, transform=transform)
    testset = torchvision.datasets.MNIST(root=img_dir, train=False, download=download, transform=transform)

    sampleids = []
    if n_samples == -1:
        return trainset, testset
    elif isinstance(n_samples, int):
        sampleids = np.random.choice(len(trainset), n_samples, replace=False)
    elif isinstance(n_samples, float):
        sampleids = np.random.choice(len(trainset), int(len(trainset) * n_samples), replace=False)

    trainset.data = trainset.data[sampleids]
    trainset.targets = trainset.targets[sampleids]
    return trainset, testset
# def get_random_affine_transform():
# def gray_to_rgb():
def load_image(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"{os.path.abspath(filename)}")
    return img
def load_image_ski(filename: str, scale=False) -> np.ndarray:
    """
    Read an image from disk. This method handles 16 bit image more correctly

    :param filename: filename of the image
    :param scale: scale pixels to RGB
    :return: numpy array with the image of shape [height, width, channels]

    >>> from common.elements.legacy.dataset import Dataset, DatasetInfo, get_dataset_info
    >>> img = load_image_ski(get_dataset_info(Dataset.POTATO_PLANT_TILES, DatasetInfo.PREVIEW))
    >>> img.shape
    (500, 600, 3)
    """
    img = io.imread(filename)
    if scale:
        img = ((img / np.max(img)) * 255).astype(np.uint8)
    return img
# def npz_to_arrays():
def split_dataset_pt(dataset: torch.utils.data.Dataset, training_frac: float = 0.5) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
# my code
    dataset_size = len(dataset)
    train_size = int(dataset_size * training_frac)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    return train_dataset, valid_dataset
# def get_hsi_dataloader():
# def get_tiled_hsi_dataloader():
# def get_bin_segm_dataset_pt():
# def get_segmentation_data_loader_pt():
# def binary_image_to_bounding_box():
# def bounding_box_to_center_box():
# def class_ids_to_class_names():
# def class_names_to_class_ids():
# def class_one_hot_to_ids():
# def convert_annotation_to_fasterrcnn_format():
# def denormalize_boxes():
# def normalize_boxes():
# def segm_one_hot_to_ids():
# def mono_flatfield_correction():
# def raw_to_bayer():
def get_cifar10_dataset_pt(folder: str="cifar10_download", transform: list = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), train: bool = True, suppress_stdout=False) -> tuple[torch.utils.data.Dataset, list[str]]:
    # Mine code
    dataset = torchvision.datasets.CIFAR10(root=folder, train=train, download=True, transform=transform)
    classes = dataset.classes
    return dataset, classes
# def load_imagenet_image():
# def get_centroidnet_dataloader_pt():
# def get_inst_segm_dataloader_pt():
# def get_inst_segm_dataset_pt():
# def get_tiled_inst_segm_dataloader():
def get_dataloader_segmentation(ds_info: ABCDatasetInfo, preprocessing: list,height: int = 224, width: int = 224, batch_size: int = 1):
    loader_funcs = [LoadImageSC(),
                    LoadClassMasksFileSC()]

    preprocessing.append(Resize(height=height, width=width))

    aug_transforms = []

    pre_transforms = preprocessing
    post_transforms = [NormaliseImageSC(mean=ds_info.mean, std=ds_info.std),
                       ClassMaskOneHotSC(class_ids=ds_info.class_ids),
                       ToTensorSC(),
                       TransposeClassMaskSC(axes=[2, 0, 1])
                       ]

    collate_fn = StackSC(concat_list=True, stack_array=True, stack_annotations=True)
    dataloader = get_generic_dataloader(dataset_info=ds_info, loader_funcs=loader_funcs,
                                        pre_transforms=pre_transforms,
                                        aug_transforms=aug_transforms,
                                        post_transforms=post_transforms,
                                        collate_fn=collate_fn,
                                        batch_size=batch_size,
                                        num_workers=0)
    return dataloader
def get_dataloader_object_detection(ds_info: ABCDatasetInfo, preprocessing: list,shuffle: bool = False, batch_size: int = 3, bbox_format: str = "tlbrc", relative_box: bool = False, pad_boxes: bool = True,):
    loader_funcs = [LoadImageSC(), LoadBoundingBoxesSC()]
    preprocess = preprocessing
    aug_transforms = [None]

    post_transforms = [BoundingBoxesToNumpySC(format=bbox_format, relative=relative_box, img_size=None),NormaliseImageSC(ds_info.mean, ds_info.std),ToTensorSC()]

    collate_fn = StackSC(stack_array=True, stack_annotations=True, pad_boxes=pad_boxes)

    dataloader = get_generic_dataloader(dataset_info=ds_info, loader_funcs=loader_funcs,
                                        pre_transforms=preprocess,
                                        aug_transforms=aug_transforms, post_transforms=post_transforms,
                                        collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

    return dataloader
