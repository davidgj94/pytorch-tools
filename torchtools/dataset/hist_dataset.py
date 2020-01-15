import matplotlib; matplotlib.use('tkagg')
import random
import numpy as np
import torch
from torch.utils import data
import os.path
import os
import pdb
from torchvision import transforms
from skimage.io import imread
import torchvision.transforms.functional as TF
from torchtools.augmentation import Compose
from .register import register
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import torchtools.lines as lines
import pickle
import torchtools.vis as vis
from scipy import ndimage
import pickle
from pathlib import Path
from skimage import measure
from .angle_detect import AngleDetectDataset

@register.attach('hist_dataset')
class HistDataset(AngleDetectDataset):
	
	def __init__(self, **kwargs):
		self.combine =  kwargs.pop("combine", False)
		super(HistDataset, self).__init__(**kwargs)

	def debug(self, label):
		for i in np.arange(label.shape[0]):
			plt.figure()
			plt.imshow(label[i])
	
	def __getitem__(self, index):

		data = super(HistDataset, self).__getitem__(index)
		label_test = data['label_test']
		label = data['label']
		weights = data['weights']
		angle_range_label = data['angle_range_label']

		if angle_range_label == 255:
			return data

		###############################################################################
		
		lines_v, _rot_angle = lines.extract_lines((label_test == 1), self.angle_range_v)
		lines_h, _ = lines.extract_lines((label_test == 1), self.angle_range_h)

		lines_v_mask = lines.create_grid(label.shape, lines_v, width=16) * (label == 0).astype(int)
		lines_h_mask = lines.create_grid(label.shape, lines_h, width=16) * (label == 0).astype(int)

		""" plt.figure()
		plt.imshow(lines_v_mask)
		plt.figure()
		plt.imshow(lines_h_mask)
		plt.show() """

		_rot_angle = np.rad2deg(_rot_angle)
		angle_dist = np.abs(self.rot_angles - _rot_angle)

		###############################################################################

		if self.combine:
			idx = angle_range_label
			n_angles = len(self.rot_angles) - 1
		else:
			idx = np.argmin(angle_dist)
			n_angles = len(self.rot_angles)

		sz = (n_angles,) + label.shape

		bin_label_v = np.zeros(sz, dtype=np.float32)
		bin_label_v[idx] = lines_v_mask.astype(np.float32)

		bin_label_h = np.zeros(sz, dtype=np.float32)
		bin_label_h[idx] = lines_h_mask.astype(np.float32)

		bin_label = np.stack((bin_label_v, bin_label_h), 0)
		
		lines_v_mask_inv = (lines_v_mask != 1).astype(np.float32)
		lines_h_mask_inv = (lines_h_mask != 1).astype(np.float32)
		weights_v = np.repeat(weights[np.newaxis,...] * lines_v_mask_inv, n_angles, 0)
		weights_v[idx] = weights
		weights_h = np.repeat(weights[np.newaxis,...] * lines_h_mask_inv, n_angles, 0)
		weights_h[idx] = weights

		weights = np.stack((weights_v, weights_h), 0)

		###############################################################################

		softmax_label_v = 255 * np.ones(label.shape, dtype=np.int64)
		softmax_label_v[lines_v_mask.astype(bool)] = angle_range_label

		softmax_label_h = 255 * np.ones(label.shape, dtype=np.int64)
		softmax_label_h[lines_h_mask.astype(bool)] = angle_range_label

		softmax_label = np.stack((softmax_label_v, softmax_label_h), 0)

		###############################################################################

		data.update(bin_label=bin_label, softmax_label=softmax_label, weights=weights)
		
		return data