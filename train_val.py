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

def get_dataloader(id_list_path, dataset_cfg, batch_size, shuffle=True):

	dataset_params = dict(dataset_cfg['params'])
	dataset_params.update(id_list_path=id_list_path)
	dataset = get_dataset(dataset_cfg['name'])(**dataset_params)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def get_last_checkpoint(checkpoint_dir):

	if not os.path.exists(checkpoint_dir):
		return None
	else:
		checkpoints_globs = list(Path(checkpoint_dir).glob('*.pth'))
		if len(checkpoints_globs) == 0:
			return None
		key = lambda x: int(os.path.basename(str(x)).split('.')[0].split('_')[-1])
		last_checkpoint_path = str(sorted(checkpoints_globs, key=key, )[-1])
		return last_checkpoint_path

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--num_epochs', type=int, default=1)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()

	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	id_list_path = os.path.join('test', 'list', 'train.txt')
	model_train = get_model(num_classes, training_cfg["model"], training_cfg.get('aux_loss', False)).to(device)
	model_train.grids_v = model_train.grids_v.to(device)
	model_train.grids_h = model_train.grids_h.to(device)
	train_dataloader = get_dataloader(id_list_path, training_cfg['dataset'], training_cfg['batch_size'], shuffle=True)

	criterion = get_loss(training_cfg['loss'])

	optimizer = optim.SGD(model_train.trainable_parameters(False), lr=0.000075, momentum=0.9, weight_decay=1e-5)
	exper_name = os.path.basename(args.config).split(".")[0]
	checkpoint_dir = os.path.join('test', 'checkpoint', exper_name)
	last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
	if last_checkpoint_path is not None:
		print("CHECKPOINT: {}".format(last_checkpoint_path))
		last_checkpoint = torch.load(last_checkpoint_path)
		model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
		current_epoch = last_checkpoint["epoch"]
	else:
		current_epoch = 0
	checkpoint_saver = CheckpointSaver(checkpoint_dir, current_epoch)

	for epoch in range(args.num_epochs):

		print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:

			if phase == 'train':
				train(model_train, 
					train_dataloader, 
					criterion, 
					optimizer,
					training_cfg)

			elif (epoch + 1) % val_cfg['val_epochs'] == 0:
				checkpoint_saver(epoch, model_train, optimizer)