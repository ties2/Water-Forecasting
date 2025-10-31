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
from typing import Tuple, Dict, Any,Sized

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
from common.elements.validation import MatchLambda, MetricLambda, calc_confusion_metrics, calc_metrics_overall


# def calc_metrics():
# def calc_accuracy():
# def calc_confusion_matrix():
def calc_confusion_matrix_np(output: np.ndarray, target: np.ndarray, labels: Iterable[Any], normalize: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate the confusion matrix between the output and the target nd arrays. Each element of the array should correspond to a class_id.
        Note: The arrays are flattend before the confusion matrix is calculated.

        :param output: the class ids output of a model. The shape is HxW.
        :param target: class ids of the ground truth. The shape is HxW.
        :param labels: list of labels to translate class ids into class names. Each index in label corresponds to a class name (string).
        :param normalize: normalizes over the target (rows), predicted (columns) or all the population: {‘true’, ‘pred’, ‘all’}.
            Note: 'true' give precision, 'pred' recall and 'all' gives accuracy.
        :returns: dataframe containing the confusion between classes with row names and column names set to labels.

        >>> import functools
        >>> from sklearn.metrics import confusion_matrix, accuracy_score
        >>> import numpy as np
        >>> calc_confusion_matrix_np(output = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),
        ...                         target = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),
        ...                         labels = ['a', 'b', 'c'])
           a  b  c
        a  4  0  0
        b  0  4  0
        c  0  0  1
        >>> calc_confusion_matrix_np(output = np.array([[0, 1, 0], [1, 0, 2], [1, 1, 0]]),
        ...                         target = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),
        ...                         labels = ['a', 'b', 'c'])
           a  b  c
        a  3  1  0
        b  0  3  1
        c  1  0  0
        """
        y_true = list(np.reshape(target, [-1]))


        y_pred = list(np.reshape(output, [-1]))
        mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=list(range(len(labels))), normalize=normalize)
        return pd.DataFrame(mat, columns=labels, index=labels)
def calc_confusion_matrix_statistic(matrix: Union[pd.DataFrame, np.ndarray], statistic: str = 'accuracy',digits=3, labels: Optional[Iterable[Any]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Calculate the statistic per class from the given confusion matrix.
        Column headers represent predictions and row headers represent targets.

        :param matrix: input
        :param statistic: any of the following:
            'accuracy'  = P(R , T) = The joint probability of a result and a target = Each item divided by the sum of all items.
            'precision' = P(T | R) = The probability of target (T) given the result (R) = Each item divided by its row sum.
            'recall'    = P(R | T) = The probability of result (R) given the target (T) = Each item divided by its column sum.
            'f1'        = 2 * (precision * recall) / (precision * recall) or Dice
            'jaccard'   = TP / (TP + FP + FN) = Intersection over union
        :param digits: round to this amount of digits.
        :param labels: list of labels to translate class ids into class names. Each index in label corresponds to a class name (string).
        :return: the confusion matrix converted to the specified statistic.
            Note: For most metrics only the main diagonal of the resulting matrix should be interpreted as the actual metric.
            Technically, the other cells usually represent the metric if a different class-pair was considered a true positive.

        >>> import functools
        >>> from sklearn.metrics import confusion_matrix, accuracy_score
        >>> import numpy
        >>> calc_confusion_matrix_statistic(matrix=pd.DataFrame([[1, 0, 0], [0, 2, 0], [0, 0, 2]]), statistic='accuracy', digits=2)
             0    1    2
        0  0.2  0.0  0.0
        1  0.0  0.4  0.0
        2  0.0  0.0  0.4
        >>> calc_confusion_matrix_statistic(matrix=pd.DataFrame([[1, 1, 1], [0, 3, 0], [0, 0, 3]]), statistic='recall', digits=2)
              0     1     2
        0  0.33  0.33  0.33
        1  0.00  1.00  0.00
        2  0.00  0.00  1.00
        >>> calc_confusion_matrix_statistic(matrix=pd.DataFrame([[1, 1, 1], [0, 3, 0], [0, 0, 3]]), statistic='precision', digits=2)
             0     1     2
        0  1.0  0.25  0.25
        1  0.0  0.75  0.00
        2  0.0  0.00  0.75
        >>> calc_confusion_matrix_statistic(matrix=pd.DataFrame([[1, 1, 1],
        ...                                                      [0, 3, 0],
        ...                                                      [0, 0, 3]]), statistic='jaccard', labels=["a", "b", "c"], digits=2)
              a     b     c
        a  0.33  0.17  0.17
        b  0.00  0.75  0.00
        c  0.00  0.00  0.75
        """
        matrix = pd.DataFrame(matrix.astype(np.float32))
        if statistic == 'accuracy':
                result = np.round(matrix / matrix.values.sum(), decimals=digits)
        elif statistic == 'precision':
                result = np.round(matrix.div(matrix.sum(axis=0), axis=1), decimals=digits)
        elif statistic == 'recall':
                result = np.round(matrix.div(matrix.sum(axis=1), axis=0), decimals=digits)
        elif statistic == 'f1':
                pre = matrix.div(matrix.sum(axis=0), axis=1)
                rec = matrix.div(matrix.sum(axis=1), axis=0)
                denominator = pre + rec
                result = np.round(
                        2 * np.divide(pre * rec, denominator, out=np.zeros_like(pre), where=denominator != 0),
                        decimals=digits)
        elif statistic == 'jaccard':
                fp = np.sum(matrix, axis=0)
                fn = np.sum(matrix, axis=1)
                fp = np.resize(np.expand_dims(fp, axis=0), matrix.shape)
                fn = np.resize(np.expand_dims(fn, axis=1), matrix.shape)
                inter = matrix
                union = fp + fn - inter
                result = np.round(np.divide(inter, union, out=np.zeros_like(matrix), where=union != 0),
                                  decimals=digits)
        else:
                raise RuntimeError(f"Unknown statistic {statistic}")
        if labels is not None:
                result.index = labels
                result.columns = labels
        return result
def summate_dict_of_matrices(matrices=dict[Any, Union[pd.DataFrame, np.ndarray]]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Summate all matrices in dictionary. This can be used to integrate multiple confusion matrices into one. Usually when
        one matrix per image is calculated.

        :param matrices: a dictionary of matrices that need to be summed
        :return: the summed matrix

        >>> import functools
        >>> from sklearn.metrics import confusion_matrix, accuracy_score
        >>> import numpy
        >>> summate_dict_of_matrices(matrices={'matrix of sample 1': pd.DataFrame([[1, 0, 0], [0, 2, 0], [0, 0, 2]]),
        ...                                    'matrix of sample 2': pd.DataFrame([[2, 0, 0], [0, 1, 0], [0, 0, 1]])})
           0  1  2
        0  3  0  0
        1  0  3  0
        2  0  0  3
        >>> summate_dict_of_matrices(matrices={'a': np.array([[1, 1, 1], [1, 2, 1], [1, 1, 2]]),
        ...                                    'b': np.array([[2, 1, 1], [1, 1, 1], [1, 1, 1]])})
        array([[3, 2, 2],
               [2, 3, 2],
               [2, 2, 3]])
        """
        assert isinstance(matrices,
                          dict), "The first parameter should contain a dictionary of confusion matrices"
        return functools.reduce(lambda a, b: a + b, matrices.values())

# def calc_class_metrics():
# def calc_class_precision_recall_f1score():
# def calc_class_precision_recall_points():
def calc_mean_average_precision(result_boxes_per_image: dict[str, Iterable], target_boxes_per_image: dict[str, Iterable], iou_t=0.5, round_scores_to_digits=1) -> float:
    """
    Calculate the mean average precision for multiple images and multiple boxes (this implementation is very slow when there are a lot of matches).

    :param result_boxes_per_image: boxes per image returned by the model, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id, probability], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id, probability]]}
    :param target_boxes_per_image: boxes per image for the ground truth, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id]]}
    :param iou_t: the Intersection over Union threshold used for matching boxes
    :param prob_t: confidence threshold to filter out boxes with a confidence lower than this value
    :param round_scores_to_digits: round each score to this amount of digits. Because the amount of thresholds for the PR-curve depends on the amount of unique score, setting this to 0.1 gives 10 score thresholds.

    :return: the mean average precision
    """

    # Convert to a format that ge_avg_precision_at_iou() understands
    image_ids = set(result_boxes_per_image.keys()).union(set(target_boxes_per_image.keys()))
    result_boxes_per_class = {}
    target_boxes_per_class = {}
    class_ids = set()
    for image_id in image_ids:
        if image_id in result_boxes_per_image.keys():
            boxes_per_class = {}
            scores_per_class = {}

            # Create a dict for boxes and for scores
            for y1, x1, y2, x2, class_id, score in result_boxes_per_image[image_id]:
                y1, x1, y2, x2, class_id, score = int(y1), int(x1), int(y2), int(x2), int(class_id), round(float(score), round_scores_to_digits)
                if class_id not in boxes_per_class.keys():
                    boxes_per_class[class_id] = []
                    scores_per_class[class_id] = []
                boxes_per_class[class_id].append([x1, y1, x2, y2])
                scores_per_class[class_id].append(score)
                class_ids.add(class_id)

            # fill up result_boxes_per_class with the boxes and scores
            for class_id in boxes_per_class.keys():
                if class_id not in result_boxes_per_class:
                    result_boxes_per_class[class_id] = {}
                result_boxes_per_class[class_id][image_id] = {'boxes': boxes_per_class[class_id], 'scores': scores_per_class[class_id]}

        if image_id in target_boxes_per_image.keys():
            boxes_per_class = {}
            scores_per_class = {}

            # Create a dict for boxes and for scores
            for y1, x1, y2, x2, class_id in target_boxes_per_image[image_id]:
                y1, x1, y2, x2, class_id = int(y1), int(x1), int(y2), int(x2), int(class_id)
                if class_id not in boxes_per_class.keys():
                    boxes_per_class[class_id] = []
                    scores_per_class[class_id] = []
                boxes_per_class[class_id].append([x1, y1, x2, y2])
                class_ids.add(class_id)

            # fill up target_boxes_per_class with the boxes and scores
            for class_id in boxes_per_class.keys():
                if class_id not in target_boxes_per_class:
                    target_boxes_per_class[class_id] = {}
                target_boxes_per_class[class_id][image_id] = boxes_per_class[class_id]

    # Check completeness and add empty results
    for class_id in class_ids:
        if class_id not in result_boxes_per_class:
            result_boxes_per_class[class_id] = {}
        for image_id in image_ids:
            if image_id not in result_boxes_per_class[class_id].keys():
                result_boxes_per_class[class_id][image_id] = {'boxes': [], 'scores': []}
        if class_id not in target_boxes_per_class:
            target_boxes_per_class[class_id] = {}
        for image_id in image_ids:
            if image_id not in target_boxes_per_class[class_id].keys():
                target_boxes_per_class[class_id][image_id] = []

    avg_prec_list = []
    for class_id in class_ids:
        result_boxes_per_image = result_boxes_per_class[class_id]
        target_boxes_per_image = target_boxes_per_class[class_id]
        data = get_avg_precision_at_iou(target_boxes_per_image, result_boxes_per_image, iou_thr=iou_t)
        avg_prec_list.append(float(data['avg_prec']))

    map = statistics.mean(avg_prec_list)
    return map

def calc_overall_precision_recall_f1score(result_boxes_per_image: dict[str, Union[Sized, Iterable]], target_boxes_per_image: dict[str, Union[Sized, Iterable]], num_classes: int, iou_t=0.5,prob_t=0.5) -> list[float]:
    """
    Calculates the overall precision, recall and f1 score for a single iou and match probability threshold.
    This function calls :meth:`calc_confusion_metrics` and :meth:`calc_metrics_overall`.

    :param result_boxes_per_image: boxes per image returned by the model, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id, probability], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id, probability]]}
    :param target_boxes_per_image: boxes per image for the ground truth, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id]]}
    :param num_classes: number of classes the model is trained on.
    :param iou_t: the overlap threshold above which to accept boxes as a match
    :param prob_t: the probability threshold above which to accept boxes for matching
    :return: [Precision, recall and f1score]

    >>> target = {"image1": [[10, 15, 18, 22, 1], [25, 17, 30, 21, 1]],
    ...           "image2": [[50, 66, 59, 72, 1], [90, 34, 110, 51, 1]]}
    >>> result = {"image1": [[10, 15, 18, 22, 1, 0.90], [25, 17, 30, 21, 1, 0.6]],
    ...           "image2": [[50, 66, 59, 72, 1, 0.90], [90, 34, 110, 51, 1, 0.6]]}
    >>> calc_overall_precision_recall_f1score(result, target, num_classes=2, iou_t=0.5, prob_t=1.0)
    [nan, 0.0, 0.0]
    >>> calc_overall_precision_recall_f1score(result, target, num_classes=2, iou_t=0.5, prob_t=0.1)
    [1.0, 1.0, 1.0]
    """

    tp, tn, fp, fn = calc_confusion_metrics(result_boxes_per_image=result_boxes_per_image, target_boxes_per_image=target_boxes_per_image,
                                            num_classes=num_classes, match_t=iou_t, prob_t=prob_t)
    return calc_metrics_overall(tp=tp, tn=tn, fp=fp, fn=fn, metrics_funcs=[MetricLambda.Precision, MetricLambda.Recall, MetricLambda.F1])
def calc_overall_precision_recall_f1score_prob_range(result_boxes_per_image: dict[str, Union[Sized, Iterable]], target_boxes_per_image: dict[str, Union[Sized, Iterable]], num_classes: int, iou_t=0.5,prob_step: float = 0.1) -> dict[float, list[float]]:
    """
    Calculates the overall precision, recall and f1 score for a single iou and mutliple probability thresholds.
    This function calls :meth:`calc_confusion_metrics` and :meth:`calc_metrics_overall`.

    :param result_boxes_per_image: boxes per image returned by the model, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id, probability], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id, probability]]}
    :param target_boxes_per_image: boxes per image for the ground truth, with keys as image identifiers.
        For example: {'img_0.png': [[box1_y1, box1_x1, box1_y2, box1_x2, class_id], ... , [boxn_y1, boxn_x1, boxn_y2, boxn_x2, class_id]]}
    :param num_classes: number of classes the model is trained on.
    :param iou_t: the overlap threshold above which to accept boxes as a match
    :param prob_step: the step in probability thresholds to test the metrics for
    :return: a dictionary with the probabilities as the keys and [Precision, recall and f1score] as the values

    >>> target = {"image1": [[10, 15, 18, 22, 1], [25, 17, 30, 21, 1]],
    ...           "image2": [[50, 66, 59, 72, 1], [90, 34, 110, 51, 1]]}
    >>> result = {"image1": [[10, 15, 18, 22, 1, 0.90], [25, 17, 30, 21, 1, 0.6]],
    ...           "image2": [[50, 66, 59, 72, 1, 0.90], [90, 34, 110, 51, 1, 0.6]]}
    >>> calc_overall_precision_recall_f1score_prob_range(result, target, num_classes=2, iou_t=0.5, prob_step=0.2)
    {0.0: [1.0, 1.0, 1.0], 0.2: [1.0, 1.0, 1.0], 0.4: [1.0, 1.0, 1.0], 0.6000000000000001: [1.0, 0.5, 0.6666666666666666], 0.8: [1.0, 0.5, 0.6666666666666666], 1.0: [nan, 0.0, 0.0]}
    """
    metrics = {}

    for prob in np.arange(0, 1 + prob_step, prob_step):
        tp, tn, fp, fn = calc_confusion_metrics(result_boxes_per_image=result_boxes_per_image, target_boxes_per_image=target_boxes_per_image,
                                                num_classes=num_classes, match_t=iou_t, prob_t=prob)
        metrics[prob] = calc_metrics_overall(tp=tp, tn=tn, fp=fp, fn=fn, metrics_funcs=[MetricLambda.Precision, MetricLambda.Recall, MetricLambda.F1])

    return metrics
# def cluster_score_skl():
# def calc_anomaly_score():