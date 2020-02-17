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

def _test(dataset, model):
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
	for _iter, data in tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True):
		model(data)

if __name__ == "__main__":
	args = parse_args()
	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	dataset_params = training_cfg["dataset"]["params"]
	id_list_path = os.path.join('list', 'partition_0', 'train.txt')
	dataset_params.update(id_list_path=id_list_path)
	dataset = get_dataset(training_cfg["dataset"]['name'])(**dataset_params)
	model = get_model(num_classes, training_cfg['model'])
	_test(dataset, model)
	print("Done")