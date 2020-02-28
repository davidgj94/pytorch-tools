from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from torchtools.loss import utils


class MultiTaskLoss:

	def __init__(self, tasks_dict):

		self.losses = []
		self.tasks_weights = []
		self.tasks_names = []
		for task_name, task_dict in tasks_dict.items():
			self.tasks_names.append(task_name)
			self.tasks_weights.append(task_dict.get("weight", 1.0))
			self.losses.append(register.get(task_dict['loss']))

	def __call__(self, inputs, data):
		total_loss = []
		for task_loss, task_weight, task_key in zip(self.losses, self.tasks_weights, self.tasks_names):
			loss_args = {** inputs[task_key], **data[task_key]}
			total_loss.append(task_loss(**loss_args) * task_weight)
		return sum(total_loss)

@register.attach('cross_entropy')
def cross_entropy(seg, label):
	device = seg.device
	label = label.to(device)
	return utils.cross_entropy(seg, label)

@register.attach('bce')
def binary_cross_entropy(seg, label, weights):
	device = seg.device
	label = label.to(device)
	weights = weights.to(device)
	return utils.binary_loss(seg, label, weights)