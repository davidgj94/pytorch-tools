import matplotlib
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
import pickle
import torchtools.vis as vis
from scipy import ndimage
import pickle
from pathlib import Path
from torchtools.road_utils.junction import compute_junction_gt, find_branch_points, extract_coords, test_branch_points
from torchtools.road_utils.affinity_utils import getKeypoints, getVectorMapsAngles, getVectorMapsAngles_v2
from skimage.morphology import medial_axis as skeletonize
from sklearn.cluster import MeanShift
from torchtools.utils import timeit
from tqdm import tqdm

@register.attach('roads_dataset')
class RoadsDataset(data.Dataset):
	"""
	Base dataset class
	"""
	def __init__(self, root, id_list_path, augmentations=[], training=True, train_ori=True, down_label=False, angle_step=15.0, treshold=0.76):

		self.id_list = np.loadtxt(id_list_path, dtype=str)
		self.mean = [0.485, 0.456, 0.406]
		self.var = [0.229, 0.224, 0.225]
		self.augmentations = Compose(augmentations)
		self.training = training
		if self.training:
			self.root = os.path.join(root, 'train')
		else:
			self.root = os.path.join(root, 'val')
		self.root = os.path.expandvars(self.root)
		self.treshold = treshold

		self.train_ori = train_ori
		self.down_label = down_label
		if self.train_ori:
			self.angle_step = angle_step

	def _load_data(self, idx):
		"""
		Load the image and label in numpy.ndarray
		"""
		image_id = self.id_list[idx] + '.png'
		img_path = os.path.join(self.root, "images", image_id)
		label_path = os.path.join(self.root, "gt", image_id)

		img = np.asarray(imread(img_path))
		label = (np.asarray(imread(label_path)) / 255.0) > self.treshold
		label = label.astype(np.float32)
		return image_id, img, label


	def __getitem__(self, index):

		image_id, image, label = self._load_data(index)
		image, label = self.augmentations(image, label)
		vis_image = image.copy()
		image = TF.to_tensor(image)
		image = TF.normalize(image, self.mean, self.var)

		label = label.astype(np.float32)
		image = image.numpy()

		data = dict(image_id=image_id, 
					image=image, 
					label=label, 
					vis_image=vis_image,)

		data.update(binary_seg=dict(label=label, weights=np.ones_like(label, dtype=np.float32)))

		if self.training and self.train_ori:

			plt.figure()
			plt.imshow(label)

			if self.down_label:
				width, height = label.shape
				label_down = cv2.resize(label.astype(np.uint8), (int(width / 4), int(height / 4)), interpolation=cv2.INTER_NEAREST,)
				keypoints = getKeypoints(label_down, is_gaussian=False, smooth_dist=10)
				getVectorMapsAngles_v2(label_down.shape, keypoints, theta=3.5, bin_size=self.angle_step)
				ori_gt, ori_weights = getVectorMapsAngles(label_down.shape, keypoints, theta=3.5, bin_size=self.angle_step)
			else:
				keypoints = getKeypoints(label, is_gaussian=False)
				ori_gt, ori_weights = getVectorMapsAngles(label.shape, keypoints, theta=10, bin_size=self.angle_step)

			data.update(ori_seg=dict(label=ori_gt, weights=ori_weights))

			# ori_gt_sum = np.clip(ori_gt.sum(0), a_min=0.0, a_max=1.0)
			# plt.figure()
			# plt.imshow(label_down)
			# plt.figure()
			# plt.imshow(ori_gt_sum)
			# plt.show()
			# for idx in np.arange(len(ori_weights)):
			# 	_ori_weights = ori_weights[idx]
			# 	_ori_gt = ori_gt[idx]
			# 	if not np.all(_ori_weights < 1.0):
			# 		plt.figure()
			# 		plt.imshow(_ori_gt)
			# 		plt.figure()
			# 		plt.imshow(_ori_weights)
			# 		plt.figure()
			# 		plt.imshow(label)
			# 		plt.show()

		return data

	def __len__(self):
		return len(self.id_list)

	def __repr__(self):
		fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
		fmt_str += "    Root: {}".format(self.root)
		return fmt_str

@register.attach('roads_dataset_v2')
class RoadsDataset_v2(RoadsDataset):
	def __init__(self, root, id_list_path, augmentations=[], training=True, train_ori=False, down_label=False, angle_step=15.0, treshold=0.76, filter_gt=False):
		super(RoadsDataset_v2, self).__init__(root, id_list_path, augmentations=augmentations, 
																	training=training, 
																	train_ori=False, 
																	down_label=down_label,
																	treshold=treshold)
		self.angle_step = angle_step
		self.filter_gt = filter_gt

	def __getitem__(self, index):
		data = super(RoadsDataset_v2, self).__getitem__(index)
		label = data['binary_seg']['label']
		# plt.figure()
		# plt.imshow(label)
		if self.training:
			if self.down_label:
				width, height = label.shape
				label_down = cv2.resize(label.astype(np.uint8), (int(width / 4), int(height / 4)), interpolation=cv2.INTER_NEAREST,)
				keypoints = getKeypoints(label_down, is_gaussian=False, smooth_dist=3.5)
				ori_gt, ori_weights = getVectorMapsAngles_v2(label_down.shape, keypoints, theta=3.5, bin_size=self.angle_step, filter_gt=self.filter_gt)
			else:
				keypoints = getKeypoints(label, is_gaussian=False)
				ori_gt, ori_weights = getVectorMapsAngles_v2(label.shape, keypoints, theta=10, bin_size=self.angle_step, filter_gt=self.filter_gt)
			data.update(ori_seg=dict(label=ori_gt, weights=ori_weights))
		return data



# @register.attach('roads_dataset_balanced')
# class RoadsDatasetBalanced(RoadsDataset):
# 	def __init__(self, root, id_list_path, augmentations=[], training=True, treshold=0.76):
# 		super(RoadsDatasetBalanced, self).__init__(root, id_list_path, augmentations=augmentations, training=training, treshold=treshold)
# 		self.weights_path = os.path.join(str(Path(id_list_path).parent), 'sampling_weights.npy')
# 		self.sampling_weights = self.compute_weights()
# 		self.id_list_all = self.id_list.copy()
# 		self.create_epoch_list()

# 	def create_epoch_list(self,):
# 		indices = torch.multinomial(self.sampling_weights, len(self), replacement=True).numpy()
# 		self.id_list = self.id_list_all[indices]

# 	def compute_weights(self,):

# 		if os.path.exists(self.weights_path):
# 			sampling_weights = np.load(self.weights_path)
# 			return torch.Tensor(sampling_weights)

# 		sampling_weights = []
# 		_, _, label = self._load_data(0)
# 		H, W = label.shape
# 		bandwidth = 100 / np.sqrt(H * W)
# 		ms = MeanShift(bandwidth=bandwidth)
# 		margin_mask = np.zeros_like(label, dtype=np.uint8)
# 		margin_h = int(0.1 * H)
# 		margin_w = int(0.1 * W)
# 		margin_mask[margin_h:-margin_h, margin_w:-margin_w] = 1
# 		for idx in tqdm(np.arange(len(self)), ncols=100, total=len(self)):
# 			_, _, label = self._load_data(idx)
# 			skel = skeletonize(label)
# 			bp = find_branch_points(skel)
# 			bp *= margin_mask
# 			if np.all(bp < 1):
# 				sampling_weights.append(1)
# 				continue
# 			coords = extract_coords(bp, normalize=True)
# 			ms.fit(coords)
# 			n_junctions = len(ms.cluster_centers_)
# 			sampling_weights.append(n_junctions)
# 		sampling_weights = np.array(sampling_weights, dtype=np.float32)
# 		sampling_weights = np.clip(sampling_weights, a_min=1.0, a_max=15.0)
# 		sampling_weights /= sampling_weights.sum()
# 		np.save(self.weights_path, sampling_weights)

# 		return torch.Tensor(sampling_weights)
	
# 	def __getitem__(self, index):
# 		data = super(RoadsDatasetBalanced, self).__getitem__(index)
# 		if index == (len(self) - 1):
# 			self.create_epoch_list()
# 		return data