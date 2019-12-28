import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np;
import torchvision
from deeplabv3.dataset import get_dataset
from deeplabv3.model import get_model
from deeplabv3.optimizer import get_optimizer
from deeplabv3.scheduler import get_scheduler
from deeplabv3.loss import get_loss
import deeplabv3.utils as utils
import pdb; pdb.set_trace()
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import CheckpointSaver
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
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--partitions', nargs='+', type=int)
	parser.add_argument('--num_epochs', type=int, default=1)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


if __name__ == "__main__":

	args = parse_args()
	exper_name = os.path.basename(args.config).split(".")[0]
	dataset_dir = os.path.join('list', args.dataset)

	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	for glob in Path(os.path.join(dataset_dir)).glob("*"):

		partition_dir = str(glob)
		partition_number = int(glob.parts[-1].split("_")[-1])

		if args.partitions is not None:
			if partition_number not in args.partitions:
				continue

		partition = os.path.join(dataset_dir, 'partition_{}').format(partition_number)
		_id_list_path = os.path.join(partition, '{}.txt')

		model_train = get_model(num_classes, training_cfg["model"], training_cfg.get('aux_loss', False)).to(device)
		train_dataloader = get_dataloader(_id_list_path.format('train'), training_cfg['dataset'], training_cfg['batch_size'], shuffle=True)

		val_expers = {}
		for _val_exper in val_cfg['val_expers']:
			model_val = get_model(num_classes, _val_exper["model"]).to(device)
			val_dataloader = get_dataloader(_id_list_path.format('val'), _val_exper['dataset'], val_cfg['batch_size'], shuffle=False)
			val_expers[_val_exper['name']] = dict(model_val=model_val, val_dataloader=val_dataloader)

		criterion = get_loss(training_cfg['loss'])
		
		optimizer = optim.SGD(model_train.trainable_parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-5)

		checkpoint_dir = os.path.join('checkpoint', args.dataset, 'partition_{}', exper_name).format(partition_number)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			print("Checkpoint")
			last_checkpoint = torch.load(last_checkpoint_path)
			model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
			current_epoch = last_checkpoint["epoch"]
		else:
			current_epoch = 0
			
		scheduler = get_scheduler(training_cfg['scheduler']['name'])(optimizer, **training_cfg['scheduler']["params"])

		# results_dir = os.path.join('results', args.dataset, 'partition_{}', exper_name).format(partition_number)
		# saver_factory = ResultsSaverFactory(num_classes, results_dir, current_epoch)
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
						scheduler, 
						device, 
						training_cfg)

				elif (epoch + 1) % val_cfg['val_epochs'] == 0:
					checkpoint_saver(epoch, model_train, optimizer)

					# for val_exper_name, val_exper in val_expers.items():
					# 	val_model, val_dataloader = val_exper['model_val'], val_exper['val_dataloader']
					# 	root_folder = val_dataloader.dataset.img_root
					# 	results_saver = saver_factory.get_saver(val_exper_name, 
					# 											root_folder, 
					# 											epoch)
					# 	val_model.load_state_dict(model_train.state_dict(), strict=False)
					# 	validate(val_model, 
					# 		val_dataloader, 
					# 		num_classes, 
					# 		device, 
					# 		saver=results_saver)