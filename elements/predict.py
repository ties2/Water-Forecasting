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
from typing import Iterable, Union, Set, Any, Optional, Any, Callable, Sized, Tuple
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
import math

import csv
from skimage.draw import disk
from common.elements.legacy.dataset import get_dataset_info, DatasetInfo, Dataset
from common import get_tmp_dir
from third_party.model_wrappers.yolov5_wrapper import YoloV5

def run_predictions(classifier, test_x, test_y):
    """Runs prediction and evaluation."""
    acc = classifier.evaluate(test_x, test_y)
    return classifier.predict(test_x), acc
# def predict():
def blur(img: np.ndarray, kernel_r: int = 25) -> np.ndarray:
    """
    Blur an image with a circular kernel size of r.

    :param img: the image with shape [h,w] or [h,w,c]
    :param kernel_r: kernel radius
    :return: the blurred image

    >>> image = np.ones([3, 3])
    >>> image[1, 1] = 2
    >>> blur(img=image, kernel_r=2)
    array([[1.44444444, 1.22222222, 1.44444444],
           [1.22222222, 1.11111111, 1.22222222],
           [1.44444444, 1.22222222, 1.44444444]])
    """
    kernel = np.zeros((kernel_r * 2 + 1, kernel_r * 2 + 1), float)
    # rr, cc = disk([kernel_r, kernel_r], kernel_r)
    rr, cc = disk((kernel_r, kernel_r), kernel_r)
    kernel[rr, cc] = 1. / len(rr)
    dst = cv2.filter2D(img, -1, kernel)
    return dst
# def convolution():
# def flatten():
# def logits_to_class_ids():
# def logits_to_probabilities():
def predict_model_skl(model, input: Union[np.ndarray, list[np.ndarray]]):
    """
    Runs a prediction with a Scikit Learn model.

    :param model: the sklearn model
    :param input: the input to the model
    :return: the output of the model

    >>> class MyModel:
    ...   def predict(self, arr):
    ...     return arr.shape
    >>> predict_model_skl(MyModel(), np.zeros((1, 2, 3)))
    (1, 2, 3)
    """
    assert isinstance(input, np.ndarray) or isinstance(input, list)
    assert hasattr(model, 'predict')
    return model.predict(input)
# def relu():
# def set_num_classes():
def predict_class_pt(model: nn.Module, input_data: torch.Tensor, labels: Optional[list[str]] = None) -> tuple[List[int], Optional[List[str]]]:
    # my code
    """
    Predicts the class IDs and optionally class names for input data using a PyTorch model.

    :param model: The trained PyTorch model.
    :param input_data: The input tensor (e.g., a batch of images). Shape: [batch_size, channels, height, width].
    :param labels: An optional list of class names corresponding to the output indices.
    :return: A tuple containing:
             - A list of predicted class IDs for the input batch.
             - A list of predicted class names (if labels are provided), otherwise None.
    """
    model.eval()
    predicted_names = None
    with torch.no_grad():  # Disable gradient calculations during inference
        # Ensure input data is on the same device as the model
        # Check if model has parameters before accessing device
        try:
            device = next(model.parameters()).device
            input_data = input_data.to(device)
        except StopIteration:
            # Handle models without parameters or ensure device handling elsewhere
            print("Model has no parameters, using input device.")
            device = input_data.device
        outputs = model(input_data)  # Get model logits
        # Get the index of the highest score (predicted class ID)
        _, predicted_ids_tensor = torch.max(outputs.data, 1)
        # Move tensor to CPU before converting to list
        predicted_ids = predicted_ids_tensor.cpu().tolist()
    if labels:
        try:
            predicted_names = [labels[i] for i in predicted_ids]
        except IndexError as e:
            print(f"Error mapping predicted IDs to labels: {e}")
            print(f"Predicted IDs: {predicted_ids}")
            print(f"Number of labels: {len(labels)}")
            predicted_names = None  # Set to None or handle appropriately

    return predicted_ids, predicted_names

# def predict_logits_prototypical_pt():
# def predict_logits_pt():
# def decode_deim_boxes_pt():
# def decode_efficientdet_boxes2_pt():
# def decode_efficientdet_boxes_pt():
# def decode_relative_xywh_ground_truth():
# def decode_yolo_boxes_pt():
def predict_yolov5_pt(model: YoloV5, input_data: torch.Tensor) -> tuple[Any, ...]:
    """
    Do a prediction using the YoloV5 model

    :param model: the model
    :param input_data: the batch containing input data with shape [b, c, h, w]
    :return: the predictions
        Note: if model.training is True the raw boxes are None.

    >>> model = YoloV5('yolov5s', num_classes=2)
    >>> model = model.eval()
    >>> predictions = predict_yolov5_pt(model, torch.Tensor(2, 3, 256, 256).cuda())
    >>> isinstance(predictions, tuple)
    True
    """
    assert (isinstance(input_data, torch.Tensor) and isinstance(model, YoloV5))
    return model(input_data)
def decode_yolov5_boxes_pt(model: YoloV5, preds: tuple[Any, ...], image_batch: torch.Tensor, margin: int = 0) -> list[np.ndarray]:
    """
    Decode the output of YOLOv5 to create boxes

    :param model: the model
    :param preds: the predictions output of :meth:`elements.predict.detection.predict_yolov5_pt`
    :param image_batch: the original image_batch that provided the raw_boxes
    :param margin: boxes outside this margin will be rejected
    :return: decoded boxes containing [[y1, x1, y2, y2, class_id, class_prob]]

    >>> model = YoloV5('yolov5s', num_classes=2)
    >>> model = model.eval()
    >>> image_batch = torch.Tensor(2, 3, 256, 256).cuda()
    >>> predictions = model(image_batch)
    >>> boxes = decode_yolov5_boxes_pt(model, predictions, image_batch)
    >>> isinstance(boxes, list)
    True
    """
    assert (isinstance(image_batch, torch.Tensor) and isinstance(model, YoloV5))
    bboxes_batch = model.detections(preds)
    if margin > 0:  # Check if centroid is outside border margin
        h, w = image_batch.shape[-2:]
        res = []
        for bboxes in bboxes_batch:
            if len(bboxes):
                bboxes = bboxes[(bboxes[:, 0] + bboxes[:, 2]) / 2 > margin, :]
                bboxes = bboxes[(bboxes[:, 1] + bboxes[:, 3]) / 2 > margin, :]
                bboxes = bboxes[(bboxes[:, 0] + bboxes[:, 2]) / 2 < w - margin, :]
                bboxes = bboxes[(bboxes[:, 1] + bboxes[:, 3]) / 2 < h - margin, :]
            res.append(bboxes)
        bboxes_batch = res
    return bboxes_batch
def detect_circles(image: np.ndarray, min_r: int, max_r: int, min_dist: int, threshold: int) -> list[list[float]]:
    # Validate input image
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    # Convert CHW to HWC if needed
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW → HWC

    # Convert to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure dtype is uint8
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blur and detect circles
    blurred = cv2.medianBlur(image, 5)

    # --- Hough Circle Detection ---
    # Parameters need tuning for your specific image and object sizes
    # dp: Inverse ratio of accumulator resolution. 1 means same as input.
    # minDist: Minimum distance between centers of detected circles.
    # param1: Higher threshold for Canny edge detector.
    # param2: Accumulator threshold for circle centers. Lower = more circles.
    # minRadius: Minimum circle radius.
    # maxRadius: Maximum circle radius.
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=threshold,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r
    )

    # Convert circles to bounding boxes
    boxes = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            y_min = max(0, y - r)
            x_min = max(0, x - r)
            y_max = min(image.shape[0], y + r)
            x_max = min(image.shape[1], x + r)
            boxes.append([y_min, x_min, y_max, x_max, 1.0, 1.0])  # score=1.0, class_id=1.0

    return boxes
def detect_contours(bin_image: np.ndarray, min_area: int) -> list[list[float]]:
    # """
    # This method detects the circular coins of the Brazilian coins dataset and return a list of bounding box coordinates.
    #
    # :param bin_image: binary input image [HxW]
    # :param min_area: the minimum area of a bounding box
    # :return: list of bounding box coordinates in format [[y1, x1, y2, x2, class_id, probability],...,].
    #
    # >>> image = np.zeros((10, 10))
    # >>> image[2:5, 2:5] = 1
    # >>> detect_contours(bin_image=image, min_area=1)
    # [[2, 2, 4, 4, 0.0, 1.0]]
    # """
    assert len(bin_image.shape) == 2, "Image should have size HxW"
    bin_image = bin_image.astype(np.uint8)

    # Create a list of outer contours and return them as a simple chain approx.
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the contours to bounding boxes
    boxes = []
    for contour in contours:
        xs = contour[:, :, 0]
        ys = contour[:, :, 1]
        y1, x1, y2, x2 = np.min(ys), np.min(xs), np.max(ys), np.max(xs)
        area = (y2 - y1) * (x2 - x1)
        if area >= min_area:
            boxes.append([int(y1), int(x1), int(y2), int(x2), 0., 1.])
    return boxes
# def filter_box_margin():
# def is_box_within_margin():
# def parse_grounding_dino_server_response():
# def predict_deim_pt():
# def predict_detr_pt_boxes():
# def predict_detr_pt_loss():
# def predict_efficientdet_boxes_pt():
# def predict_efficientdet_pt():
# def predict_faster_rcnn_pt():
# def predict_grounding_dino_external():
# def predict_maskrcnn_pt():
# def predict_yolo_boxes_pt():
# def predict_yolo_pt():
def predict_yolov5_boxes_pt(model: YoloV5, image_batch: torch.Tensor, margin: int = 0) -> list[torch.Tensor | np.ndarray]:
    """
    Do a prediction on one image using the YoloV5 model and decode the boxes.

    :param model: the trained YOLOv5 model
    :param image_batch: the image data batch [(b, c, h, w)]
    :param margin: boxes outside this margin will be rejected
    :return: a list of Tensors with boxes of (N x 5) containing [[y1, x1, y2, y2, class_id, class_prob]]

    >>> model = YoloV5('yolov5s', num_classes=2)
    >>> model = model.eval()
    >>> boxes = predict_yolov5_boxes_pt(model, torch.Tensor(2, 3, 256, 256).cuda())
    >>> isinstance(boxes, list)
    True
    """
    assert (isinstance(image_batch, torch.Tensor) and isinstance(model, YoloV5))
    # run the minibatch
    predictions = predict_yolov5_pt(model=model, input_data=image_batch)

    # decode boxes
    boxes_list = decode_yolov5_boxes_pt(model, predictions, image_batch, margin)

    return boxes_list

# def decode_keypoint_rcnn_pt():
# def predict_keypoint_rcnn():
# def forward_mm_segmentation():
# def forward_mm_segmentation_loss():
def logits_to_ids(logits: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
    """
    converts a tensor with class logits to class_ids by taking the max value of the logits dimension

    :param logits: Tensor of dimension [b, nclasses, h, w] containing the logits per class
        The first dimension (b) can also be a list. If the tensor has three dimensions it is assumed that it has shape [nclasses, h, w]
    :return:  Tensor of dimension [b, 1, h, w] containing class ids (index of the logit with the maximum value)

    >>> tensor = torch.zeros(1, 3, 3, 3)
    >>> tensor[0, 0, 0:4, 0:4] = 1
    >>> tensor[0, 0:2, 1, 1] = 0
    >>> tensor[0, 2, 1, 1] = 1
    >>> logits_to_ids(tensor)
    tensor([[[[0, 0, 0],
              [0, 2, 0],
              [0, 0, 0]]]])
    """
    if isinstance(logits, list):
        logits = torch.stack(logits, 0)
    if len(logits.shape) == 4:
        dim = 1
    elif len(logits.shape) == 3:
        dim = 0
    else:
        raise RuntimeError("Unknown amount of dimensions for performing this operation")
    ids = torch.argmax(logits, dim=dim, keepdim=True)
    return ids
# def predict_cascadenet_pt():
# def predict_segmentation_fewshot_pt():
# def predict_segmentation_fewshot_with_mask_pt():
def predict_segmentation_pt(model: torch.nn.Module, input_data: Union[torch.Tensor, list[torch.Tensor], list[dict[str, torch.Tensor]]]) -> torch.Tensor:
    """
    Do a prediction using a segmentation model (UNet, DeepLab, etc.)
    Note: If the input is a list or tuple then the output will be a list of dicts.
    The following formats are supported: [{"image", image_tensor1}, {"image", image_tensor2}, ... ] or
    [image_tensor1, image_tensor2, ... ] or batch tensor.

    :param model: a segmentation model that returns class logits
    :param input_data: the minibatch of input data with shape [(b, c, h, w]
    :return: a [b, nclasses, h, w] class logit tensor

    >>> from third_party.model_wrappers.unet import UNet
    >>> tensor = torch.zeros(2, 3, 50, 50)
    >>> predict_segmentation_pt(UNet(5), tensor).shape
    torch.Size([2, 5, 50, 50])
    >>> predict_segmentation_pt(UNet(5), list(tensor))[0]["class_masks"].shape
    torch.Size([5, 50, 50])
    """
    output_as_dict = False
    if isinstance(input_data, list) or isinstance(input_data, tuple):
        if len(input_data) > 0 and isinstance(input_data[0], dict):
            input_data = [images["image"] for images in input_data]
        input_data = torch.stack(input_data)
        output_as_dict = True
    assert isinstance(input_data, torch.Tensor)
    output = model(input_data)
    if output_as_dict:
        # Output is the List of dict style.
        output = [{"class_masks": t} for t in output]
    return output
# def predict_auto_encoding_pt():
# def decode_centroidnet():
# def extract_centroidnet_images():
# def nms_centroidnet():
# def convert_data_to_maskrcnn_dataformat():
# def decode_class_masks_pt():
# def decode_instance_masks_pt():
# def decode_rcnn_boxes_pt():
# def predict_fasterrcnn_pt():
# def predict_maskrcnn_pt():
# def adjust_brightness_contrast_kornia():
# def apply_morphology_kornia():
# def detect_weeds():
# def normalize_kornia():
# def solarize_kornia():
# def weight_hsv_kornia():
# def diffusion_backward_pass_hf():
# def diffusion_forward_pass_hf():
# def diffusion_generate_image_hf():
# def prior_preservation_generate_class_images_hf():
