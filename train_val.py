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
	parser.add_argument('--config', type=str, required=True, nargs='+')
	parser.add_argument('--num_epochs', type=int, required=True, nargs='+')
	parser.add_argument('--parts', type=int, required=True, nargs='+')
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()

def main(config, num_epochs, use_cpu, part):

	id_list_path = os.path.join('list', 'partition_{}'.format(part), 'train.txt')
	exper_name = os.path.basename(config).split(".")[0]
	root_checkpoint_dir = os.path.join('checkpoint', 'partition_{}'.format(part))
	checkpoint_dir = os.path.join(root_checkpoint_dir, exper_name)

	num_classes, training_cfg, val_cfg = utils.get_cfgs(config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

	model_train = get_model(num_classes, training_cfg["model"]).to(device)
	pdb.set_trace()
	train_dataloader = get_dataloader(id_list_path, training_cfg['dataset'], training_cfg['batch_size'], shuffle=True)
	criterion = get_loss(training_cfg['loss'])

	current_epoch = 0
	learning_rate = training_cfg.get("learning_rate", 0.0005)

	last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
	if (last_checkpoint_path is None) and ("init" in training_cfg):
		last_checkpoint_path = os.path.join(root_checkpoint_dir, training_cfg["init"])

	if last_checkpoint_path is not None:
		last_checkpoint = torch.load(last_checkpoint_path)
		model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
		current_epoch = last_checkpoint["epoch"]
		learning_rate = last_checkpoint.get("learning_rate", learning_rate)
		print("CHECKPOINT: {}".format(last_checkpoint_path))
		print("Learning rate: {}".format(learning_rate))

	optimizer = optim.SGD(model_train.trainable_parameters(True), 
						lr=learning_rate, 
						momentum=0.9, 
						weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	checkpoint_saver = CheckpointSaver(checkpoint_dir, current_epoch)

	for epoch in range(num_epochs):

		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:

			if phase == 'train':
				train(model_train, 
					train_dataloader, 
					criterion, 
					optimizer,
					training_cfg)
				scheduler.step()
			elif (epoch + 1) % val_cfg['val_epochs'] == 0:
				checkpoint_saver(epoch, model_train, optimizer)


if __name__ == "__main__":
	args = parse_args()
	for part in args.parts:
		for config_path, num_epochs in zip(args.config, args.num_epochs):
			main(config_path, num_epochs, args.use_cpu, part)