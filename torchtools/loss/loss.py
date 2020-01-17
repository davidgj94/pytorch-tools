from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn


class Loss:

	def __init__(self, losses_list):

		self.losses = {}
		# hay que cambiar el cfg para que sea dict no lista
		for el in losses_list:
			loss_name = str(*el.keys())
			loss_weight = el[loss_name]
			# loss_weight = 1.0 if not bool(loss_weight)
			self.losses[loss_name] = loss_weight

	def __call__(self, inputs, data):
		total_loss = []
		for loss_name, loss_weight in self.losses.items():
			_loss = register.get(loss_name)(inputs, data) * loss_weight
			total_loss.append(_loss)
		return sum(total_loss)



@register.attach('multitask')
def multitask(inputs, data):

	targets_3classes = data['label_3c'].to(inputs.device)
	targets_2classes = data['label_2c'].to(inputs.device)

	probs_4class = F.softmax(inputs, dim=1)
	probs_4class = probs_4class.transpose(0,1)
	probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
							 probs_4class[2], 
							 probs_4class[3]]).transpose(0,1)
	probs_3class = torch.log(probs_3class)
	loss_3c = F.nll_loss(probs_3class, targets_3classes)

	probs_2class = F.log_softmax(inputs.transpose(0,1)[:2].transpose(0,1), dim=1)
	if 'weights' in data:
		batch_size, _, H, W = inputs.shape
		weights = data['weights'].to(inputs.device)
		loss_2c = torch.gather(probs_2class, 1, targets_2classes.view(batch_size, 1, H, W)).squeeze()
		loss_2c = (loss_2c * weights).view(batch_size, -1).sum(1) / (weights.view(batch_size, -1).sum(1) + 1.0)
		loss_2c = -1.0 * loss_2c.mean()
	else:
		loss_2c = F.nll_loss(probs_2class, targets_2classes)

	return loss_3c + loss_2c

@register.attach('multitask_v2')
def multitask_v2(inputs, data):

	seg = inputs["seg"]
	device = seg.device
	targets_3classes = data['label_3c'].to(device)
	targets_2classes = data['label_2c'].to(device)
	weights = data['weights'].to(device)

	seg = seg.transpose(0,1)
	seg_0, seg_1 = seg[:2]
	seg_diff = seg_1 - seg_0
	seg_max = torch.max(seg_0, seg_1).unsqueeze(0)
	seg = torch.cat([seg_max, seg[2:]], 0).transpose(0,1)

	loss_3c = F.cross_entropy(seg, targets_3classes, ignore_index=255)
	loss_2c = F.binary_cross_entropy_with_logits(seg_diff, targets_2classes, weight=weights)

	return loss_3c + loss_2c

@register.attach('multitask_v3')
def multitask_v3(inputs, data):

	x_diff = inputs["x_diff"]
	x_max = inputs["x_max"]
	device = x_diff.device

	targets_3classes = data['label_3c'].to(device)
	targets_2classes = data['label_2c'].to(device)
	weights = data['weights'].to(device)

	loss_3c = F.cross_entropy(x_max, targets_3classes, ignore_index=255)
	loss_2c = F.binary_cross_entropy_with_logits(x_diff, targets_2classes, weight=weights)

	return loss_3c + loss_2c


@register.attach('cross_entropy')
def cross_entropy(inputs, data):
	seg = inputs['seg']
	device = seg.device
	targets = data['label'].to(device)
	return F.cross_entropy(seg, targets, ignore_index=255)


@register.attach('margin_ranking')
def margin_ranking_loss(inputs, data, margin=0.1):

	hist = inputs["hist"]
	device = hist.device
	margin_label = data["margin_label"].to(device)
	indices = data['angle_range_label']

	margin_loss = []
	for _hist, _margin_label, idx in zip(hist, margin_label, indices):
		if idx.item() == 255:
			margin_loss.append(0.0)
		else:
			den = _hist[:-1] + _hist[1:]
			x1 = (_hist[:-1] / den)
			x2 = (_hist[1:] / den)
			margin_loss.append(F.margin_ranking_loss(x1, x2, _margin_label, 
				margin=margin,
				reduction="sum"))

	return sum(margin_loss) / len(margin_loss)


@register.attach('bin_loss')
def binary_loss(inputs, data):

	line_seg = inputs['line_seg']
	device = line_seg.device
	weights = data['weights'].to(device).squeeze(0)

	label_2c = data['label_2c'].to(device).squeeze(0)
	bin_loss = F.binary_cross_entropy_with_logits(line_seg, label_2c, reduction="none")
	bin_loss = (bin_loss * weights).sum() / (weights.sum() + 1.0)

	return bin_loss

@register.attach('ori_loss')
def ori_loss(inputs, data):

	# Por ahora batch size de 1

	def _bin_loss_aux(scores, label):
		bin_loss = F.binary_cross_entropy_with_logits(scores, label, reduction="none")
		return (bin_loss * weights).sum() / (weights.sum() + 1.0)


	seg = inputs['seg'].squeeze()
	seg_v = inputs['seg_v'].squeeze()
	seg_h = inputs['seg_h'].squeeze()
	device = seg.device

	weights = data['weights'].to(device).squeeze()
	label_v = data['mask_v'].to(device).squeeze()
	label_h = data['mask_h'].to(device).squeeze()
	label = data['mask'].to(device).squeeze()

	main_loss = _bin_loss_aux(seg, label)
	aux_loss = (_bin_loss_aux(seg_v, label_v) + _bin_loss_aux(seg_h, label_h)) / 2.0

	return main_loss + 0.4 * aux_loss


@register.attach('hist_loss')
def hist_loss(inputs, data):

	def _bin_loss(line_seg, bin_label, weights):
		loss = []
		for _line_seg, _bin_label in zip(line_seg, bin_label):
			_loss = F.binary_cross_entropy_with_logits(_line_seg, _bin_label, reduction="none")
			loss.append((_loss * weights).sum() / (weights.sum() + 1.0))
		return sum(loss)

	def _softmax_loss(line_seg, softmax_label):
		loss = []
		for _line_seg, _softmax_label in zip(line_seg, softmax_label):
			loss.append(F.cross_entropy(_line_seg.unsqueeze(0), _softmax_label.unsqueeze(0), ignore_index=255))
		return sum(loss)

	line_seg = inputs['line_seg']
	device = line_seg.device

	if 'softmax_label' in data:

		softmax_label = data['softmax_label'].to(device).squeeze(0)
		bin_label = data['bin_label'].to(device).squeeze(0)
		weights = data['weights'].to(device).squeeze(0)

		bin_loss = _bin_loss(line_seg, bin_label, weights)
		softmax_loss = _softmax_loss(line_seg, softmax_label)

		return (bin_loss + softmax_loss) / 2
	
	else:

		return 0.0


