# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
import py360convert
import random as rd

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class PanoTrans(BaseTransform):

    def __init__(self, ignore_index=255, shuffle=False, crop_size=(640, 640)):
        self.ignore_index = ignore_index
        self.shuffle = shuffle
        self.crop_size = crop_size

    @staticmethod
    def _shuffle(results, key, order):
        img_array = results[key]
        split_height, split_width = img_array.shape[0] // 2, img_array.shape[1] // 2
        img_blocks = [
            img_array[:split_height, :split_width],
            img_array[:split_height, split_width:],
            img_array[split_height:, :split_width],
            img_array[split_height:, split_width:]
        ]
        new_img_array = np.empty_like(img_array)
        new_img_array[:split_height, :split_width] = img_blocks[order[0]]
        new_img_array[:split_height, split_width:] = img_blocks[order[1]]
        new_img_array[split_height:, :split_width] = img_blocks[order[2]]
        new_img_array[split_height:, split_width:] = img_blocks[order[3]]
        return new_img_array

    @staticmethod
    def _cube_to_pano(results, key, ignore_index=255):
        if len(results[key].shape) == 2:
            h, w = results[key].shape
            img = np.full((h, w * 6), fill_value=ignore_index, dtype=np.uint8)
            img[:, :w] = results[key]  # cube image
            img = img[:, :, np.newaxis]
            img = py360convert.c2e(img, h=2*h, w=4*w, cube_format='horizon', mode='nearest')
            img = img[:, :, 0]
        else:
            h, w, _ = results[key].shape
            img = np.full((h, w * 6, 3), fill_value=ignore_index, dtype=np.uint8)
            img[:, :w] = results[key]  # cube image
            img = py360convert.c2e(img, h=2*h, w=4*w, cube_format='horizon', mode='bilinear')
        h_start = int(0.5 * h)
        h_end = int(0.5 * h + h)
        w_start = int(1.5 * w)
        w_end = int(1.5 * w + w)
        img = img[h_start:h_end, w_start:w_end]
        return img

    def transform(self, results: dict) -> dict:
        if self.shuffle:
            order = [0, 1, 2, 3]
            rd.shuffle(order)
            results['img'] = self._shuffle(results, 'img', order)
            results['gt_seg_map'] = self._shuffle(results, 'gt_seg_map', order)
        pad_h = self.crop_size[0] - results['img'].shape[0]
        pad_w = self.crop_size[1] - results['img'].shape[1]
        results['img'] = np.pad(results['img'], ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=(0, 0))
        results['gt_seg_map'] = np.pad(results['gt_seg_map'], ((0, pad_h), (0, pad_w)), 'constant', constant_values=(self.ignore_index, self.ignore_index))
        results['img_shape'] = results['img'].shape[:2]
        results['img'] = self._cube_to_pano(results, 'img', ignore_index=0).astype(np.uint8)
        results['gt_seg_map'] = self._cube_to_pano(results, 'gt_seg_map', ignore_index=self.ignore_index).astype(np.uint8)
        return results

