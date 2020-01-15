import torch
from torchtools.metrics import AverageMeter
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pdb
import torch.nn as nn
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


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

			aux_loss = 0.0
			if 'aux' in outputs:
				aux_loss = criterion(outputs['aux'], data)
			loss = (criterion(outputs['out'], data) + 0.4 * aux_loss) / iter_size
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
