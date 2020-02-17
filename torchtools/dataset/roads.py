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
from torchtools.road_utils.junction import compute_junction_gt

@register.attach('roads_dataset')
class BaseDataset(data.Dataset):
	"""
	Base dataset class
	"""
	def __init__(self, root, id_list_path, augmentations=[], train_junctions=False):

		self.root = root
		self.id_list = np.loadtxt(id_list_path, dtype=str)
		self.mean = [0.485, 0.456, 0.406]
		self.var = [0.229, 0.224, 0.225]
		self.augmentations = Compose(augmentations)
		self.train_junctions = train_junctions

	def _load_data(self, idx):
		"""
		Load the image and label in numpy.ndarray
		"""
		image_id = self.id_list[idx] + '.png'
		img_path = os.path.join(self.root, "images", image_id)
		label_path = os.path.join(self.root, "label", image_id)

		img = np.asarray(imread(img_path))
		pdb.set_trace() # ver si hay que quedarse con canal
		label = np.asarray(imread(label_path))
		return image_id, img, label


	def __getitem__(self, index):

		image_id, image, label = self._load_data(index)
		image, label = self.augmentations(image, label)
		image = TF.to_tensor(image)
		image = TF.normalize(image, self.mean, self.var)

		label = label.astype(np.float32)
		image = image.numpy()

		data = dict(image_id=image_id, image=image, label=label)

		if self.train_junctions:
			junction_gt, junction_weights = compute_junction_gt(label)
			data.update(junction_gt=junction_gt, junction_weights=junction_weights)
		
		return data


	def __len__(self):
		return len(self.id_list)

	def __repr__(self):
		fmt_str = "     Dataset: " + self.__class__.__name__ + "\n"
		fmt_str += "    Root: {}".format(self.root)
		return fmt_str