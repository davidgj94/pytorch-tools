from torch.nn import functional as F
import torch
import pdb
from .register import register
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from torchtools.loss import utils


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


@register.attach('lines_binary_seg')
def lines_binary_seg(inputs, data):

	lines_seg = inputs['seg_roads']
	device = lines_seg.device

	weights = data['weights'].to(device)
	label = data['label'].to(device)

	return utils.binary_loss(lines_seg, label, weights)

@register.attach('junction_binary_seg')
def junction_binary_seg(inputs, data):

	lines_seg = inputs['seg_junction']
	device = lines_seg.device

	weights = data['junction_weights'].to(device)
	label = data['junction_gt'].to(device)

	return utils.binary_loss(lines_seg, label, weights)



@register.attach('aux_ori_loss_v1')
def aux_ori_loss_v1(inputs, data):

	seg_v = inputs['seg_v'].squeeze(1)
	seg_h = inputs['seg_h'].squeeze(1)
	device = seg_h.device

	weights = data['weights'].to(device)
	label_v = data['mask_v'].to(device)
	label_h = data['mask_h'].to(device)
	
	return (utils.binary_loss(seg_v, label_v, weights) + utils.binary_loss(seg_h, label_h, weights)) / 2


@register.attach('aux_ori_loss_v2')
def aux_ori_loss_v2(inputs, data):

	seg_v = inputs['seg_v'].squeeze(1)
	seg_h = inputs['seg_h'].squeeze(1)
	device = seg_h.device

	weights = data['weights'].to(device)
	label_v = data['mask_v'].to(device)
	label_h = data['mask_h'].to(device)

	pos_weight = torch.FloatTensor([3.0])
	
	return (utils.binary_loss(seg_v, label_v, weights, pos_weight=pos_weight) + utils.binary_loss(seg_h, label_h, weights, pos_weight=pos_weight)) / 2

@register.attach('ori_block_loss')
def ori_block_loss(inputs, data):

	def weight_losses(losses):
		K = len(losses)
		weight_factors = (np.arange(1,K+1) / np.arange(1,K+1).sum()).tolist()
		return sum([_loss * _weight for _loss, _weight in zip(losses, weight_factors)])

	lines_seg = inputs['lines_seg'].transpose(0,1)
	seg_v = inputs['seg_v'].transpose(0,1)
	seg_h = inputs['seg_h'].transpose(0,1)
	device = lines_seg.device

	weights = data['weights'].to(device)
	label = data['mask'].to(device)
	label_v = data['mask_v'].to(device)
	label_h = data['mask_h'].to(device)

	line_seg_loss = []
	for _lines_seg in lines_seg:
		_lines_seg_loss = utils.binary_loss(_lines_seg, label, weights)
		line_seg_loss.append(_lines_seg_loss)

	aux_loss = []
	for _seg_v, _seg_h in zip(seg_v, seg_h):
		_aux_loss = 0.5 * (utils.binary_loss(_seg_v, label_v, weights) + utils.binary_loss(_seg_h, label_h, weights))
		aux_loss.append(_aux_loss)
	
	return weight_losses(line_seg_loss) + 0.4 * weight_losses(aux_loss)


@register.attach('hist_loss')
def hist_loss(inputs, data):

	# def _bin_loss(seg, bin_label, weights):
	# 	loss = []
	# 	for _seg, _bin_label, _weights in zip(seg, bin_label, weights):
	# 		loss.append(utils.binary_loss(_seg.squeeze(), _bin_label.squeeze(), _weights.squeeze()))
	# 	return sum(loss) / len(loss)

	# def _softmax_loss(seg, softmax_label):
	# 	loss = []
	# 	for _seg, _softmax_label in zip(seg, softmax_label):
	# 		loss.append(utils.cross_entropy(_seg.unsqueeze(0), _softmax_label.unsqueeze(0)))
	# 	return sum(loss) / len(loss)

	seg_v = inputs['seg_v']
	seg_h = inputs['seg_h']
	device = seg_v.device

	softmax_label_v = data["softmax_label_v"].to(device)
	softmax_label_h = data["softmax_label_h"].to(device)

	bin_label_v = data["bin_label_v"].to(device)
	weights_v = data["weights_v"].to(device)
	bin_label_h = data["bin_label_h"].to(device)
	weights_h = data["weights_h"].to(device)

	# softmax_loss = (_softmax_loss(seg_v, softmax_label_v) + _softmax_loss(seg_h, softmax_label_h)) / 2.0
	# bin_loss = (_bin_loss(seg_v, bin_label_v, weights_v) + _bin_loss(seg_h, bin_label_h, weights_h)) / 2.0
	softmax_loss = (utils.cross_entropy(seg_v, softmax_label_v) + utils.cross_entropy(seg_h, softmax_label_h)) / 2.0
	bin_loss = (utils.binary_loss(seg_v, bin_label_v, weights_v) + utils.binary_loss(seg_h, bin_label_h, weights_h)) / 2.0

	return softmax_loss + bin_loss




