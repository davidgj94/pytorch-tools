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
from torchtools.save import CheckpointSaver, MetricSaver
from pathlib import Path
from torchtools.metrics import AverageMeter
from torchtools.metric import get_metric
from torchtools.save import makedir


def set_bn_eval(m):
	if isinstance(m, nn.modules.batchnorm._BatchNorm):
		m.eval()

def train(train_model, train_dataloader, criterion, optimizer, training_cfg):

	train_model.train()
	train_model.apply(set_bn_eval)

	running_loss = AverageMeter()
	running_loss.reset()

	iter_size = training_cfg['iter_size']
	display_iters = training_cfg['display_iters']

	optimizer.zero_grad()

	# Iterate over data.
	for _iter, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), dynamic_ncols=True):

		with torch.set_grad_enabled(True):

			outputs = train_model(data)
			loss = criterion(outputs, data) / iter_size
			loss.backward()
			running_loss.update(loss.item() * iter_size, n=train_dataloader.batch_size)

			if _iter > 0 and _iter % iter_size == 0:
				optimizer.step()
				optimizer.zero_grad()

			if _iter > 0 and _iter % (display_iters * iter_size) == 0:
				print()
				print('----------------------------------------------')
				print('>> Loss: {:.4f}'.format(running_loss.value()))
				print('----------------------------------------------')
				print()
				running_loss.reset()
				for idx, param_group in enumerate(optimizer.param_groups):
					print('Param Group {}: {}'.format(idx, param_group['lr']))

	optimizer.zero_grad()

def validate(val_model, val_loader, metric):

	val_model.eval()   # Set model to evaluate mode
	np.random.seed(0)

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, data in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):
			preds = val_model(data)
			metric(preds, data)
	
	metric_value = metric.value()
	metric.reset()

	return metric_value

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
	parser.add_argument('--datasets', type=str, required=True, nargs='+')
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()

def main(config, num_epochs, dataset_name, use_cpu=False, max_failed_attemps=2, root_checkpoint_dir=None):

	exper_name = os.path.basename(config).split(".")[0]
	if root_checkpoint_dir is None:
		root_checkpoint_dir = os.path.join('checkpoint', dataset_name)
	else:
		root_checkpoint_dir = os.path.join(root_checkpoint_dir, 'checkpoint', dataset_name)
		makedir(root_checkpoint_dir)
	checkpoint_dir = os.path.join(root_checkpoint_dir, exper_name)

	num_classes, training_cfg, val_cfg = utils.get_cfgs(config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

	model_train = get_model(num_classes, training_cfg["model"]).to(device)
	train_list_path = os.path.join('list', dataset_name, 'train.txt')
	train_dataloader = get_dataloader(train_list_path, training_cfg['dataset'], training_cfg['batch_size'], shuffle=True)
	criterion = get_loss(training_cfg['tasks'])

	current_epoch = 0
	learning_rate = training_cfg.get("learning_rate", 0.0005)

	last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
	if (last_checkpoint_path is None) and ("init" in training_cfg):
		last_checkpoint_path = os.path.join(root_checkpoint_dir, training_cfg["init"])

	if last_checkpoint_path is not None:
		last_checkpoint = torch.load(last_checkpoint_path)
		model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
		current_epoch = last_checkpoint["epoch"]
		if "init" not in training_cfg:
			learning_rate = last_checkpoint.get("learning_rate", learning_rate)
		print("CHECKPOINT: {}".format(last_checkpoint_path))
		print("Learning rate: {}".format(learning_rate))

	optimizer = optim.SGD(model_train.trainable_parameters(base_lr=learning_rate),
					lr=learning_rate,
					momentum=0.9, 
					weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	checkpoint_saver = CheckpointSaver(checkpoint_dir, current_epoch)
	metric_saver = MetricSaver(val_cfg["metric"]["name"], checkpoint_dir, current_epoch)

	if num_epochs > 0:
		for epoch in range(num_epochs):
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			print('-' * 10)
			train(model_train, 
						train_dataloader, 
						criterion, 
						optimizer,
						training_cfg)
			scheduler.step()
			if ( (epoch + 1) % val_cfg['val_epochs'] == 0 ) or ( epoch == (num_epochs - 1) ):
				checkpoint_saver(epoch, model_train, optimizer)
	else:
		epoch = 0

	val_list_path = os.path.join('list', dataset_name, 'val.txt')
	val_dataloader = get_dataloader(val_list_path, val_cfg['dataset'], val_cfg['batch_size'], shuffle=False)
	metric = get_metric(val_cfg['metric'])

	model_val = get_model(num_classes, val_cfg["model"]).to(device)
	model_val.load_state_dict(model_train.state_dict(), strict=False)
	best_metric = validate(model_val, val_dataloader, metric)
	del model_val
	metric_saver(epoch, best_metric)

	num_failed_attemps = 0
	while True:

		train(model_train, 
			train_dataloader, 
			criterion, 
			optimizer,
			training_cfg)
		scheduler.step()

		if (epoch + 1) % val_cfg['val_epochs'] == 0:

			model_val = get_model(num_classes, val_cfg["model"]).to(device)
			model_val.load_state_dict(model_train.state_dict(), strict=False)
			last_metric = validate(model_val, val_dataloader, metric)
			del model_val
			metric_saver(epoch, last_metric)

			if last_metric > best_metric:
				best_metric = last_metric
				checkpoint_saver(epoch, model_train, optimizer)
				num_failed_attemps = 0
			else:
				num_failed_attemps += 1
				if num_failed_attemps >= max_failed_attemps:
					break

		epoch += 1


if __name__ == "__main__":
	args = parse_args()
	for dataset_name in args.datasets:
		for config_path, num_epochs in zip(args.config, args.num_epochs):
			main(config_path, num_epochs, dataset_name, use_cpu=args.use_cpu)