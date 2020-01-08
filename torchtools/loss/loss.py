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


""" @register.attach('dice_loss')
def dice_loss(inputs, data):

	pred = inputs['out']
	pred_coop = inputs['out_coop']
	device = pred.device
	label = data['label'].to(device)

	label_s = label.clone()
	label_s[label != 255] = torch.clamp(label_s[label != 255] - 1, min=0)

	label_dice = (label == 1).float()
	mask = (label == 0).float() + (label == 1).float()

	def _dice_loss(pred, target, mask):
		numerator = 2 * torch.sum(pred * target * mask)
		denominator = torch.sum((pred + target) * mask)
		return 1.0 - (numerator + 1.0) / (denominator + 1.0)

	def _cross_entropy(pred, target):

		probs_4class = F.softmax(pred, dim=1).transpose(0,1)
		probs_3class = torch.stack([(probs_4class[0] + probs_4class[1]), 
							 probs_4class[2], 
							 probs_4class[3]]).transpose(0,1)
		logprobs_3class = torch.log(probs_3class)
		return F.nll_loss(logprobs_3class, targets_3classes)

	return _dice_loss(pred_coop, label_dice, mask) + _cross_entropy(pred, label_s) """


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


@register.attach('line_detect')
def line_detect_loss(inputs, data):

	#Usar con batch-size de 1 y entrenando con imágenes de APRON

	lines_scores = inputs["lines_scores"]
	device = lines_scores.device
	lines_gt = data['lines_gt'].to(device).squeeze()
	pos_weight = 1 / lines_gt.mean()
	line_loss = F.binary_cross_entropy_with_logits(lines_scores, lines_gt, reduction="mean", pos_weight=pos_weight)
	return line_loss
