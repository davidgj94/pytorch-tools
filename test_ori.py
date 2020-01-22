import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchtools.dataset import get_dataset
from torchtools.model import get_model
from torchtools.loss import get_loss
import torchtools.utils as utils
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
from torchsummary import summary
from torchtools.save import CheckpointSaver
from pathlib import Path
from val import validate
from train import train
import matplotlib.pyplot as plt

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	return parser.parse_args()

def _test(dataset_params):
	id_list_path = os.path.join('test', 'list', 'train.txt')
	dataset = get_dataset("ori_dataset")(**dataset_params)
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
	model = get_model(4, dict(name="test_ori_v4", stride=16))
	for _iter, data in tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True):
		idx = data['angle_range_label'].item()
		model(idx)
		plt.show()

if __name__ == "__main__":
	args = parse_args()
	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	dataset_params = training_cfg["dataset"]["params"]
	id_list_path = os.path.join('test', 'list', 'train.txt')
	dataset_params.update(id_list_path=id_list_path)
	_test(dataset_params)
	print("Done")