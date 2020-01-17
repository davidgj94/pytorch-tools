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

@register.attach('angle_detect_dataset')
class AngleDetectDataset(data.Dataset):

	def __init__(self, root, id_list_path, 
		angle_step=15.0, min_angle=-45.0, max_angle=45.0, augmentations=[]):
	
		self.root = root
		self.id_list = np.loadtxt(id_list_path, dtype=str)
		self.mean = [0.485, 0.456, 0.406]
		self.var = [0.229, 0.224, 0.225]
		self.augmentations = Compose(augmentations)
		self.angle_range_v = np.array((min_angle, max_angle))
		self.rot_angles = np.arange(self.angle_range_v[0], self.angle_range_v[1] + angle_step, angle_step)
		self.angle_range_h = self.angle_range_v + 90.0
		self.angle_step = angle_step

	def _load_data(self, idx):
		"""
		Load the image and label in numpy.ndarray
		"""
		image_id = self.id_list[idx] + '.png'
		img_path = os.path.join(self.root, "images", image_id)
		label_path = os.path.join(self.root, "masks", image_id)
		label_test_path = os.path.join(self.root, "masks_test", image_id)

		img = np.asarray(imread(img_path))
		label = np.asarray(imread(label_path))[..., 0]
		label_test = np.asarray(imread(label_test_path))[..., 0]
		label_test[label_test == 255] = 0

		return image_id, img, label, label_test


	def __getitem__(self, index):

		image_id, img, label, label_test = self._load_data(index)

		image, _label = self.augmentations(img, np.dstack((label, label_test)))
		vis_image = image.copy()
		label, label_test = np.dsplit(_label, 2)
		label = np.squeeze(label).astype(np.int64)
		label_test = np.squeeze(label_test).astype(np.int64)

		not_ignore = (label != 255)
		label_3c = label.copy()
		label_3c[not_ignore] = np.clip(label_3c[not_ignore] - 1, a_min=0, a_max=None)

		label_2c = (label == 1).astype(np.float32)
		weights = (label_3c == 0).astype(np.float32)

		image = TF.to_tensor(image)
		image = TF.normalize(image, self.mean, self.var)
		image = image.numpy()

		angle_range_label = 255
		margin_label = np.ones(len(self.rot_angles) - 2, dtype=np.float32)
		if np.any(label_test == 1):
			_, _rot_angle_v = lines.extract_lines((label_test == 1), self.angle_range_v)
			_rot_angle = np.rad2deg(_rot_angle_v)
			angle_dist = np.abs(self.rot_angles - _rot_angle)
			margin_label = np.sign(angle_dist[1:-1] - angle_dist[:-2]).astype(np.float32)
			angle_range_label = np.argsort(angle_dist)[:2].min()
			""" print(_rot_angle)
			plt.figure()
			plt.imshow((label == 1)) """

		return dict(image_id=image_id, 
			image=image, 
			vis_image=vis_image, 
			angle_range_label=angle_range_label,
			margin_label=margin_label,
			label_test=label_test, 
			label=label,
			label_2c=label_2c,
			weights=weights
			)

	def __len__(self):
		return len(self.id_list)

	def __repr__(self):
		fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
		fmt_str += "    Root: {}".format(self.root)
		return fmt_str

@register.attach('ori_dataset')
class OriDataset(AngleDetectDataset):

	def __init__(self, root, id_list_path, 
		angle_step=15.0, min_angle=-45.0, max_angle=45.0, augmentations=[]):

		super(OriDataset, self).__init__(root, id_list_path, 
										angle_step=angle_step,
										min_angle=min_angle,
										max_angle=max_angle,
										augmentations=augmentations)

		self.id_list = [img_id for img_id in self.id_list.tolist() if "APR" in img_id]
		
	
	def __getitem__(self, index):

		data = super(OriDataset, self).__getitem__(index)
		weight = data['weights']
		label_test = data['label_test']

		lines_v, _rot_angle_v = lines.extract_lines((label_test == 1), self.angle_range_v)
		mask_v = lines.create_grid(label_test.shape, lines_v, width=16).astype(np.float32) * weight

		lines_h, _ = lines.extract_lines((label_test == 1), self.angle_range_h)
		mask_h = lines.create_grid(label_test.shape, lines_h, width=16).astype(np.float32) * weight

		mask = np.clip(mask_v + mask_h, a_min=0.0, a_max=1.0)

		""" print("Rotation angle: {}".format(np.rad2deg(_rot_angle_v)))
		plt.figure()
		plt.imshow(mask_v)
		plt.figure()
		plt.imshow(mask_h)
		plt.figure()
		plt.imshow(mask) """

		data.update(mask_v=mask_v, mask_h=mask_h, mask=mask)

		return data





@register.attach('angle_detect_dataset_v2')
class AngleDetectDatataset_v2(AngleDetectDataset):

	def __init__(self, **kwargs):
		kernel = kwargs.pop('kernel_size')
		self.margin = int((kernel - 1) // 2)
		self.theta_step = kwargs.pop('theta_step', 1.5)
		self.rho_step = kwargs.pop('rho_step', 10)
		self.tresh_h = kwargs.pop('tresh_h', 0.5)
		self.tresh_l = kwargs.pop('tresh_l', 0.25)
		self.debug = kwargs.pop('debug', False)
		super(AngleDetectDatataset_v2, self).__init__(**kwargs)
		self.id_list = [img_id for img_id in self.id_list.tolist() if "APR" in img_id]

	# def get_line_gt(self, true_lines, proposed_lines):

	#     true_lines = np.array(true_lines)
	#     proposed_lines = np.array(proposed_lines)

	#     n_lines = proposed_lines.shape[0]
	#     lines_gt = np.zeros(n_lines, dtype=np.float32)
	#     for idx in np.arange(n_lines):
	#         distance = np.abs(proposed_lines[idx] - true_lines)
	#         close_lines = np.logical_and(distance[:,0] < 1, distance[:,1] < np.deg2rad(0.5))
	#         lines_gt[idx] = np.any(close_lines).astype(np.float32)
	#     return lines_gt


	def get_lines_iou(self, true_lines, proposed_lines, label):

		mask = np.invert(label.astype(bool))

		def _compute_iou(true_mask, proposed_mask):
			not_ignored = np.logical_and(true_mask > 0, true_mask < 2)
			hist = np.bincount(2 * true_mask[not_ignored].flatten() + proposed_mask[not_ignored].flatten(), minlength=4).reshape((2,2))
			iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
			return iou[1]

		def _create_mask(_line, width=8 , guard=False):
			sz = mask.shape
			line_mask = lines.create_grid(sz, [_line], width=width).astype(int)
			if guard:
				line_mask = 2  * line_mask - lines.create_grid(sz, [_line], width=2*width)
			return line_mask * mask


		true_masks = np.array([_create_mask(_line, width=8, guard=False) for _line in true_lines], dtype=int)
		proposed_masks = np.array([_create_mask(_line, width=16, guard=False) for _line in proposed_lines], dtype=int)

		true_lines = np.array(true_lines)
		proposed_lines = np.array(proposed_lines)

		n_p = proposed_lines.shape[0]
		n_t = true_lines.shape[0]
		lines_iou = []
		for i in np.arange(n_p):
			iou = []
			for j in np.arange(n_t):
				_true_mask = true_masks[j]
				_proposed_mask = proposed_masks[i]
				if np.isnan(_compute_iou(_true_mask, _proposed_mask)):
					pdb.set_trace()
				iou.append(_compute_iou(_true_mask, _proposed_mask))
			lines_iou.append(max(iou))

		return np.array(lines_iou)


	def extract_edges(self, mask):

		if np.any(mask > 0):
			mask[mask > 0] = 255
			edges = cv2.Laplacian(mask.astype(np.uint8), cv2.CV_8U)
			cc = measure.label(edges, background=0)
			cc_coords = []
			for cc_idx in np.unique(cc).tolist()[1:]:
				y, x = np.where(cc == cc_idx)
				cc_coords.append((x, y))
			return cc_coords
		else:
			return None

	def plot_gt(self, true_lines, proposed_lines, label):

		mask = np.invert(label.astype(bool))

		def _create_mask(_lines, width=8):
			sz = mask.shape
			line_mask = lines.create_grid(sz, _lines, width=width).astype(int)
			return line_mask * mask

		vis_img = np.zeros(mask.shape, dtype=int)
		vis_img[_create_mask(proposed_lines, width=3).astype(bool)] = 2
		vis_img[_create_mask(true_lines, width=3).astype(bool)] = 1
		plt.figure()
		plt.imshow(vis_img)

	def get_lines_gt(self, lines_endpoints, lines_iou, return_is=False):
		
		is_positive = lines_iou > self.tresh_h
		is_negative = lines_iou < self.tresh_l

		n_pos = int(is_positive.sum())
		n_neg = int(is_negative.sum())
		lines_gt = np.append(np.ones(n_pos), np.zeros(n_neg)).astype(np.float32)
		lines_endpoints = np.vstack((lines_endpoints[is_positive], lines_endpoints[is_negative]))

		if return_is:
			return lines_endpoints, lines_gt, (is_positive, is_negative)
		else:
			return lines_endpoints, lines_gt



	def __getitem__(self, index):

		data = super(AngleDetectDatataset_v2, self).__getitem__(index)

		label = data['label']
		label_test = (data['label_test'] == 1)

		not_ignore = (label != 255)
		label_multiclass = label.copy()
		label_multiclass[not_ignore] = np.clip(label_multiclass[not_ignore] - 1, a_min=0, a_max=2)
		data.update(label_multiclass=label_multiclass)

		label = label_multiclass[self.margin:-self.margin, self.margin:-self.margin]
		label_test = label_test[self.margin:-self.margin, self.margin:-self.margin]
		vis_image = data['vis_image'][self.margin:-self.margin, self.margin:-self.margin]

		idx = data['angle_range_label']
		if idx != 255:

			edges_coords = self.extract_edges(label)
			sz = label_test.shape

			_rot_angle = self.rot_angles[idx]
			angle_range_v = np.array((_rot_angle, _rot_angle + self.angle_step))
			angle_range_h = angle_range_v + 90.0
			
			proposed_lines_v, lines_endpoints_v = lines.get_line_proposals(angle_range_v, sz, 
				angle_step=self.theta_step, 
				rho_step=self.rho_step, 
				edges_coords=edges_coords, 
				label=label)
			true_lines_v, _ = lines.extract_lines(label_test, angle_range_v)
			lines_v_iou = self.get_lines_iou(true_lines_v, proposed_lines_v, label)

			proposed_lines_h, lines_endpoints_h = lines.get_line_proposals(angle_range_h, sz, 
				angle_step=self.theta_step, 
				rho_step=self.rho_step, 
				edges_coords=edges_coords, 
				label=label)
			true_lines_h, _ = lines.extract_lines(label_test, angle_range_h)
			lines_h_iou = self.get_lines_iou(true_lines_h, proposed_lines_h, label)

			lines_endpoints_v, lines_gt_v, (is_positive_v, is_negative_v) = self.get_lines_gt(np.array(lines_endpoints_v), lines_v_iou, return_is=True)
			if lines_endpoints_v.shape[0] == 0:
				pdb.set_trace()
			if self.debug:
				self.plot_gt(true_lines_v, np.array(proposed_lines_v)[is_positive_v].tolist(), label)
				self.plot_gt(true_lines_v, np.array(proposed_lines_v)[is_negative_v].tolist(), label)
				plt.show()
			proposed_lines_v = np.array(proposed_lines_v)
			proposed_lines_v = np.vstack((proposed_lines_v[is_positive_v], proposed_lines_v[is_negative_v]))
			
			lines_endpoints_h, lines_gt_h, (is_positive_h, is_negative_h) = self.get_lines_gt(np.array(lines_endpoints_h), lines_h_iou, return_is=True)
			if lines_endpoints_h.shape[0] == 0:
				pdb.set_trace()
			if self.debug:
				self.plot_gt(true_lines_h, np.array(proposed_lines_h)[is_positive_h].tolist(), label)
				self.plot_gt(true_lines_h, np.array(proposed_lines_h)[is_negative_h].tolist(), label)
				plt.show()
			proposed_lines_h = np.array(proposed_lines_h)
			proposed_lines_h = np.vstack((proposed_lines_h[is_positive_h], proposed_lines_h[is_negative_h]))
		   
			
			lines_gt = np.append(lines_gt_v, lines_gt_h)
			data.update(lines_endpoints_v=lines_endpoints_v, 
				lines_endpoints_h=lines_endpoints_h, 
				lines_gt=lines_gt, 
				proposed_lines_v=proposed_lines_v,
				proposed_lines_h=proposed_lines_h,
				vis_image=vis_image,
				)
				
		return data



