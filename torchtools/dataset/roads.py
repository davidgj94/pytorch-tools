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
import pickle
import torchtools.vis as vis
from scipy import ndimage
import pickle
from pathlib import Path
from torchtools.road_utils.junction import compute_junction_gt, find_branch_points, extract_coords, test_branch_points
from skimage.morphology import medial_axis as skeletonize
from sklearn.cluster import MeanShift
from torchtools.utils import timeit
from tqdm import tqdm

@register.attach('roads_dataset')
class RoadsDataset(data.Dataset):
	"""
	Base dataset class
	"""
	def __init__(self, root, id_list_path, augmentations=[], training=True, treshold=0.76):

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
					vis_image=vis_image,
					binary_seg=dict(label=label, 
									weights=np.ones_like(label, dtype=np.float32)))
		
		return data

	def __len__(self):
		return len(self.id_list)

	def __repr__(self):
		fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
		fmt_str += "    Root: {}".format(self.root)
		return fmt_str


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