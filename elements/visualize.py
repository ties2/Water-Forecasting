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
from common.elements.visualize import draw_rect

from sklearn.svm import SVC
from skimage.feature import hog
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from common.elements.legacy.dataset import get_dataset_info, Dataset, DatasetInfo
from common import get_tmp_dir, wait_forever
from common.elements.legacy.classification import ClassifierType
from common.data.transforms import StackSC, ToTensorSC, NormaliseImageSC, BoundingBoxesToNumpySC


# def visualize():
# def draw_bounding_boxes():
# def hs_to_display():
# def hyperspectral_to_rgb():
def log_loss(loss, epoch, name="loss"):
    """
    Print the loss of the current epoch

    :param loss: loss to print .
    :param epoch: number of epochs elapsed.
    :param name: loss name (e.g. training, validation).

    >>> log_loss(10.0, 0)
    10.0
    """

    logger.info(f"{name} after epoch {epoch} is {loss:.5f}.")
    return loss

# def log_running_loss():
# def normalize_image():
# def put_text():
# def scale_for_display():
# def set_pixel_value_pt():
# def visualize_2d():
# def apply_color_map():
# def get_color_lut():
# def overlay_imgs():
# def show_image_cv():
# def add_image_name():
def create_tb(tb_name: str, delete_previous: bool = True, start_tb: bool = True) -> SummaryWriter:
    return common.elements.visualize.create_tb(tb_name, delete_previous=delete_previous, start_tb=start_tb)
def delete_tb(tb_name: str):
    return common.elements.visualize.delete_tb(tb_name=tb_name)
def plot_to_tensor(image):
    """ Convert a Matplotlib figure to a tensor for TensorBoard. """
    buf = b_io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(image)
    buf.seek(0)
    image = PIL.Image.open(buf)
    return np.array(image)
# def show_hyper_parameters_tb():
# def show_image_and_boxes_tb():
def show_image_and_boxes_tb(img: np.ndarray, boxes_result: Optional[Iterable] = None, image_name: str = None,dark_mode: bool = True,boxes_target: Optional[Iterable] = None, epoch=0, name: str = "images_and_boxes",dataformats: str = "HWC", normalize=False,result_color=(255, 255, 255), target_color=(0, 255, 0), writer: SummaryWriter = None,class_names=None, width: int = 2,result_width: int = None, showtext: bool = True):
    """
    Show images with bounding box overlay.
    If epoch == 0 the previous tensorboard is removed and a new tensorboard in launched, unless a SummaryWriter is passed to this function.

    :param img: image to display. The shape of this image is equal to dataformats
    :param boxes_result: [[y1, x1, y2, x2, class_id, class_prob]]
    :param boxes_target: [[y1, x1, y2, x2, class_id]]
    :param epoch: the tensorboard step
    :param image_name: text to display above the image
    :param dark_mode: image_name banner above the image blends into tensorboard given dark_mode or not
    :param name: the name of the image
    :param dataformats: the dataformat: HWC or CHW
    :param normalize: flag indicating if images should be rescaled between 0 an 255.
    :param result_color: the color of the result boxes
    :param target_color: the color of the target boxes
    :param writer: a SummaryWriter to write to
    :param width: width of the line used to draw rectangles for the target boxes
    :param result_width: width of the line used to draw rectangles for the result boxes
    :param showtext: show the classes of the boxes
    :param class_names: the class names of the class ids in the boxes_result and boxes_target

    >>> import os
    >>> wr = SummaryWriter()
    >>> image = np.zeros((10, 10, 3))
    >>> show_image_and_boxes_tb(img=image, writer=wr)
    """
    if result_width is None:
        result_width = width

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        dataformats = "HWC"
    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)
        dataformats = "HWC"

    # Take the mean of the channels if it's a hyperspectral cube
    h, w, c = img.shape
    if c > 3:
        img = hyperspectral_to_rgb(img)

    if normalize:
        min, max = np.min(img), np.max(img)
        if min != max:
            img = (((img - min) / (max - min)) * 255)

    img = np.ascontiguousarray(img.astype(np.uint8))

    if boxes_target is not None:
        for box_target in boxes_target:
            class_id = -1
            if len(box_target) == 4:
                y1, x1, y2, x2 = box_target
            elif len(box_target) == 5:
                if isinstance(box_target[4], str):
                    y1, x1, y2, x2, class_name = box_target
                else:
                    y1, x1, y2, x2, class_id = int(box_target[0]), int(box_target[1]), int(box_target[2]), int(
                        box_target[3]), int(box_target[4])
            else:
                raise RuntimeError("Invalid number of elements in target bounding box")
            class_name = None
            if showtext and class_id != -1:
                class_id = int(class_id)
                class_name = class_names[class_id] if class_names is not None else class_id
            draw_rect(img, y1, x1, y2, x2, class_name, target_color, width=width)

    if boxes_result is not None:
        for box_result in boxes_result:
            if len(box_result) == 4:
                y1, x1, y2, x2 = box_result
                class_id = -1
                prob = 0
            elif len(box_result) == 5:
                y1, x1, y2, x2, class_id = box_result
                prob = 0
            elif len(box_result) == 6:
                y1, x1, y2, x2, class_id, prob = box_result
            else:
                raise RuntimeError("Invalid number of elements in result bounding box")
            class_id = int(class_id)

            # determine what to show for class name
            class_name = None
            if class_id != -1 and showtext:
                class_name = class_names[class_id] if class_names is not None else class_id
                if type(prob) == int:
                    class_name = f"{class_name} {prob}"
                else:
                    class_name = f"{class_name} {prob:.2}"

            draw_rect(img, y1, x1, y2, x2, class_name, result_color, width=result_width)

    if image_name:
        img = common.elements.visualize.add_image_name(img=img, w=w, dark_mode=dark_mode, image_name=image_name)

    writer.add_image(name, img, epoch, dataformats=dataformats)
    writer.flush()
# def show_image_and_keypoints_tb():
def show_image_and_masks_tb(img: np.ndarray, writer: SummaryWriter, mask_result: np.ndarray = None,image_name: str = None, dark_mode: bool = True,mask_target: np.ndarray = None, color_map_style: int = cv2.COLORMAP_JET,epoch=0, name: str = "images_and_masks", dataformats: str = "HWC", normalize=False,class_names: list[str] = None):
    """
    Show images with bounding box overlay.
    If epoch == 0 the previous tensorboard is removed and a new tensorboard in launched, unless a SummaryWriter is passed to this function.

    :param img: shape == dataformats
    :param mask_result: result class masks of the image [h, w, 1] == class_id
    :param mask_target: target class masks of the image [h, w, 1] == class_id
    :param image_name: text to display above the image
    :param color_map_style: the colors cv2 uses for the masks
    :param dark_mode: image_name banner above the image blends into tensorboard given dark_mode or not
    :param epoch: epoch number
    :param name: name of section
    :param dataformats: the dataformat: HWC or CHW
    :param normalize: flag indicating if images should be rescaled between 0 an 255.
    :param writer: a SummaryWriter to write to
    :param class_names: names of the classes

    >>> image = np.random.rand(5, 5, 3)
    >>> wr = SummaryWriter()
    >>> m_result = np.random.rand(5, 5, 3)
    >>> m_target = np.random.rand(5, 5, 3)
    >>> show_image_and_masks_tb(img=image, mask_result=m_result, mask_target=m_target, writer=wr)
    """

    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = 0

    if dataformats == "CHW":
        img = np.moveaxis(img, 0, 2)
        mask_target = np.moveaxis(mask_target, 0, 2) if mask_target is not None else None
        mask_result = np.moveaxis(mask_result, 0, 2) if mask_result is not None else None
        dataformats = "HWC"

    # Take the mean of the channels if it's a hyperspectral cube
    h, w, c = img.shape
    if c > 3:
        img = hyperspectral_to_rgb(img)

    if normalize:
        img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
    img = np.ascontiguousarray(img.astype(np.uint8))

    # Reverse one hot encoding for correct color mask
    if mask_target is not None and len(mask_target.shape) > 2 and dataformats == "HWC":
        if mask_target.shape[2] == 1:
            mask_target = mask_target[:, :, 0]
        else:
            mask_target = np.argmax(mask_target, axis=2)

    if mask_result is not None and len(mask_result.shape) > 2 and dataformats == "HWC":
        if mask_result.shape[2] == 1:
            mask_result = mask_result[:, :, 0]
        else:
            mask_result = np.argmax(mask_result, axis=2)

    # Merge mask
    border = 20
    if mask_result is not None and mask_target is not None:
        mask = np.zeros((h, (w * 3) + (2 * border), 3), dtype=np.uint8)
    elif mask_result is not None or mask_target is not None:
        mask = np.zeros((h, (w * 2) + (1 * border), 3), dtype=np.uint8)
    else:
        mask = np.zeros([h, w, 3], dtype=np.uint8)

    mask[:, :,] = (255, 255, 255)

    mask[:, :w] = img
    cursor = w + border

    color_map = np.zeros([1, 1])
    # define color mapping
    if num_classes > 0:
        color_map = np.expand_dims(np.arange(num_classes), 1)
        color_map = ((color_map / num_classes) * 255).astype(np.uint8)
    else:
        if mask_target is not None:
            max_value_target = np.max(mask_target) + 1 if np.max(mask_target) else 1
            color_map = np.expand_dims(np.arange(max_value_target), 1)
            color_map = ((color_map / np.max(mask_target)) * 255).astype(np.uint8)
        elif mask_result is not None:
            max_value_result = np.max(mask_target) + 1 if np.max(mask_target) else 1
            color_map = np.expand_dims(np.arange(max_value_result), 1)
            color_map = ((color_map / np.max(mask_result)) * 255).astype(np.uint8)

    color_map = cv2.applyColorMap(np.ascontiguousarray(color_map), color_map_style)

    if mask_target is not None:
        mask_target_disp = np.zeros((mask_target.shape[0], mask_target.shape[1], 3), dtype=np.uint8)
        for n in np.unique(mask_target):
            mask_target_disp[mask_target == n] = color_map[n][0]

        mask_target_disp = cv2.cvtColor(mask_target_disp, cv2.COLOR_BGR2RGB)

        mask[:, cursor:cursor + w] = mask_target_disp
        cursor += w + border

    if mask_result is not None:
        mask_result_disp = np.zeros((mask_result.shape[0], mask_result.shape[1], 3), dtype=np.uint8)
        for n in np.unique(mask_result):
            mask_result_disp[mask_result == n] = color_map[n][0]

        mask_result_disp = cv2.cvtColor(mask_result_disp, cv2.COLOR_BGR2RGB)

        mask[:, cursor:cursor + w] = mask_result_disp

    if num_classes > 0:
        legenda = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        legenda = cv2.resize(legenda, None, fx=200, fy=30, interpolation=cv2.INTER_NEAREST)
        for idx, class_name in enumerate(class_names):
            cv2.putText(legenda, f"{class_name}", (5, (idx * 30) + 22), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    else:
        legenda = None

    # writer.add_image(name, img, epoch, dataformats=dataformats)
    # writer.add_image(name + "_target", img_target, epoch, dataformats=dataformats)
    # writer.add_image(name + "_result", img_result, epoch, dataformats=dataformats)
    if image_name:
        mask = common.elements.visualize.add_image_name(img=mask, w=mask.shape[1], dark_mode=dark_mode,
                                                        image_name=image_name)

    writer.add_image(name, mask, epoch, dataformats=dataformats)
    if legenda is not None:
        writer.add_image("legenda", legenda, epoch, dataformats="HWC")
    writer.flush()
def show_image_grid_tb(images: Union[torch.Tensor, np.ndarray], writer: SummaryWriter, epoch: int = 0,name: str = "image-grid", dark_mode: bool = True, image_name: str = None,normalize: bool = False, dataformats: str = "NCHW", invert: bool = False):
    """
    Shows images in a grid format

    :param images: images to be displayed. Can be a batch of images in PyTorch Tensor or NumPy ndarray format.
    :param writer: the SummaryWriter instance used for TensorBoard logging.
    :param epoch: the current epoch or step in training, used as a counter for TensorBoard.
    :param name: tag name to be used in TensorBoard for these images.
    :param dark_mode: if True, sets a dark theme for the image name banner.
    :param image_name: optional text to be displayed above the image grid.
    :param normalize: flag indicating if images should be rescaled between 0 an 255.
    :param invert: inverts the values for each channel (255 - values), can be useful for grayscale images
    :param dataformats: string indicating the format of the data. Should be 'NCHW' (batch, channels, height, width)
                        or 'NHW' (batch, height, width).
    """
    if dataformats == "NHW":
        images = np.expand_dims(images, axis=1)
    if isinstance(images, np.ndarray):
        images = torch.tensor(images)
    img_grid = torchvision.utils.make_grid(images)
    show_image_tb(img=img_grid.numpy(), name=name, dataformats="CHW", writer=writer, epoch=epoch,
                  normalize=normalize, dark_mode=dark_mode, image_name=image_name, invert=invert)
def show_image_tb(img: np.ndarray, writer: Optional[SummaryWriter] = None, epoch=0, name: str = "image", dataformats: str = "HWC",image_name: str = None, dark_mode: bool = True, invert: bool = False,normalize=False, clip_values: Optional[tuple[float, float]] = None):
    """
    Show an image TensorBoard.

    :param img: image to display. The shape of this image is equal to dataformats
    :param epoch: the tensorboard step
    :param invert: inverts the values for each channel (255 - values), can be useful for grayscale images
    :param name: the name of the image
    :param image_name: text to display above the image
    :param dark_mode: image_name banner above the image blends into tensorboard given dark_mode or not
    :param dataformats: the dataformat: HWC or CHW
    :param normalize: flag indicating if images should be rescaled between 0 an 255.
    :param writer: a SummaryWriter to write to
    :param clip_values: tuple of [a, b], clip values based on the statistics in the image [-a * sigma .. +b * sigma]
    """
    common.elements.visualize.show_image_tb(img=img, writer=writer, epoch=epoch, name=name, dataformats=dataformats, image_name=image_name, dark_mode=dark_mode, invert=invert, normalize=normalize, clip_values=clip_values)
# def show_images_and_labels_tb():

def show_loss_tb(loss, epoch, writer: SummaryWriter, name: str = "loss"):

    """
    Show loss on a TensorBoard.

    :param loss: current loss
    :param epoch: Tensorboard step
    :param name: name of the metric
    :param writer: writer to use

    >>> wr = SummaryWriter()
    >>> show_loss_tb(10, 0, wr)
    """
    if isinstance(loss, dict):
        writer.add_scalars(name, loss, epoch)
    else:
        writer.add_scalar(name, loss, epoch)
    writer.flush()
def show_metrics_graph_tb(metrics: dict[float, list[float]], writer: SummaryWriter = None, name: str = "Metrics over Probability Thresholds"):
    """
    Plots precision, recall, and F1-score as line plots over varying probability thresholds and logs it to TensorBoard. Metrics is the result from calc_overall_precision_recall_f1score_prob_range

    :param metrics: a dictionary with probability thresholds as keys and a list containing precision, recall, and F1-score as values.
    :param writer: TensorBoard SummaryWriter instance.
    :param name: title of the plot and the tag under which the image will be logged in TensorBoard.

    >>> metrics = {0.1: [0.8, 0.7, 0.75], 0.2: [0.82, 0.72, 0.77], 0.3: [0.85, 0.75, 0.8]}
    >>> writer = SummaryWriter()
    >>> show_metrics_graph_tb(metrics, writer)
    """
    if writer is None:
        writer = SummaryWriter()

    probabilities = list(metrics.keys())
    precisions = [metric[0] for metric in metrics.values()]
    recalls = [metric[1] for metric in metrics.values()]
    f1_scores = [metric[2] for metric in metrics.values()]

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=probabilities, y=precisions, label='Precision', marker='o')
    sns.lineplot(x=probabilities, y=recalls, label='Recall', marker='o')
    sns.lineplot(x=probabilities, y=f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Metrics')
    plt.title(f'{name}')
    plt.legend()
    plt.tight_layout()

    image_tensor = plot_to_tensor(plt.gcf())

    writer.add_image(f'{name}_metrics_plot', image_tensor, dataformats='HWC')

    writer.flush()
# def show_plot_tb():
# def show_rewards_tb():
# def show_scalar_tb():
def show_text_tb(name: str, writer: SummaryWriter, text: str = "", epoch: int = 0,confusion_matrix: pd.DataFrame = None, space: str = "."):
    """
    Show a text in tensorboard

    :param name: name of the text field
    :param text: text value (can use Markdown)
    :param confusion_matrix: confusion matrix which will be converted to html
    :param epoch: epoch the scalar was calculated.
    :param writer: writer to use
    :param space: what is a space represented by?

    >>> import os
    >>> from common.elements.visualize import launch_tb, delete_tb, plt_images
    >>> wr = SummaryWriter()
    >>> show_text_tb(name="caption", text="hello world", writer=wr)
    """

    if confusion_matrix is not None:
        content = confusion_matrix.to_html()
    elif text:
        content = "  \n".join([f"`{line.replace(' ', space)}`" for line in str(text).split("\n")])
    else:
        print("No input text or dataframe given")
        return

    writer.add_text(name, content, epoch)
    writer.flush()
# def show_video_tb():
def create_train_valid_plot() -> tuple[Any, Any, Any]:
    """
    Create a plot for graphing training and validation loss.
    The return values of this method can be passed to :meth:`visualize.matplotlib_elmts.plot_train_valid`

    return: The matplotlib figure, training loss axis and validation loss axis
    """
    fig = plt.figure()
    train_ax = fig.add_subplot(211, title="training loss")
    valid_ax = fig.add_subplot(212, title="validation loss")
    fig.tight_layout()
    train_ax.plot([0])
    valid_ax.plot([0])
    return fig, train_ax, valid_ax

# def gen_scatter_plt_np():
def plot_train_valid(train_loss_list: list[float], valid_loss_list: list[float], fig, train_ax, valid_ax):
    """
    Update the plot created by :meth:`visualize.matplotlib_elmts.create_train_valid_plot`

    :param train_loss_list: a list of values for the training loss
    :param valid_loss_list: a list of values for the validation loss
    :param fig: the matplotlib figure to plot on (needed to refresh the plot)
    :param train_ax: the training axis to plot the new data points to
    :param valid_ax: the validation axis to plot the new data points to
    """
    xs = list(range(len(train_loss_list)))
    # Update the data
    train_ax.lines[0].set_data(xs, train_loss_list)
    valid_ax.lines[0].set_data(xs, valid_loss_list)
    if len(train_loss_list) > 1 or len(valid_loss_list) > 1:
        # Resize axes
        train_ax.set_xlim(0, max(xs))
        valid_ax.set_xlim(0, max(xs))
        train_ax.set_ylim(min(train_loss_list), max(train_loss_list))
        valid_ax.set_ylim(min(valid_loss_list), max(valid_loss_list))
    # Draw plot
    fig.canvas.draw()
    fig.show()
# def show_image_plt():
# def show_images_and_labels_plt():
# def show_images_grid_plt():
def visualize_samples(samples, classes):
    """Visualizes list of (img_tensor, label) samples."""
    fig, axs = plt.subplots(1, len(samples), figsize=(15, 3))
    for i, (img, label) in enumerate(samples):
        axs[i].imshow(img.permute(1, 2, 0).numpy())  # Convert tensor to (H, W, C)
        axs[i].set_title(classes[label])
        axs[i].axis('off')
    plt.show()