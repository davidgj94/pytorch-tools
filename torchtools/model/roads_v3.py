import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from .deeplab import Deeplabv3Plus, init_conv, Deeplabv3
import copy
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from .register import register
from .gabor import gabor
from .angle_net import predict
from torchtools.utils import save_heatmaps
import os
import os.path
import shutil
import cv2


def compute_seg(x, output_shape, classifier):
	x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
	return classifier(x)


def compute_grids(angle_range, angle_step, kernel_size, grid_size):

	center = (kernel_size - 1) // 2
	k = center - (grid_size - 1) // 2
	def _compute_grid(angle):
		theta = torch.zeros(1, 2, 3)
		theta[:, :, :2] = torch.tensor([[np.cos(angle), -1.0*np.sin(angle)],
									[np.sin(angle), np.cos(angle)]])
		grid = F.affine_grid(theta, (1,1) + _pair(kernel_size))
		grid = grid.squeeze(0)[k:-k, k:-k].unsqueeze(0)
		return grid

	min_angle, max_angle = angle_range
	angles = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
	grids = []
	for angle in (-1.0*angles).tolist():
		grid = _compute_grid(angle)
		grids.append(grid)

	return torch.cat(grids, dim=0)


def padding(dilation, kernel_size):
	return dilation * (kernel_size - 1) // 2


def create_convnet(conv_layer, planes, kernel_size, padding, dilation, groups, **kwargs):

	pdb.set_trace() # Ver el contenido de kwargs

	def _init_conv(m):
		if isinstance(m, OrientedConv2d):
			m.reset_parameters()
		else:
			init_conv(m)

	n_layers = len(planes) - 1
	layers = []
	for idx in np.arange(n_layers):
		in_planes = planes[idx]
		out_planes = planes[idx+1]
		_conv = conv_layer(in_planes, out_planes, 
								kernel_size=kernel_size, 
								padding=padding, 
								dilation=dilation,
								bias=False, **kwargs)
		_init_conv(_conv)
		norm = nn.GroupNorm(groups, out_planes)
		relu = nn.ReLU()
		layers += [_conv, norm, relu]

	return nn.Sequential(*layers)


def forward_ori(ori_net, x, idx):
	for module in ori_net._modules.values():
		if isinstance(module, OrientedConv2d):
			module.set_grid(idx)
	return ori_net(x)


class Deeplabv3_Roads(Deeplabv3):
	def __init__(self, n_classes, pretrained_model, freeze_backbone=False, freeze_aspp=False):
		super(Deeplabv3_Roads, self).__init__(n_classes, pretrained_model, aux=False)
		self.freeze_parameters(freeze_backbone, freeze_aspp)
		self.params_group = super(Deeplabv3_Roads, self).trainable_parameters(True)
	
	def freeze_parameters(self, freeze_backbone, freeze_aspp):
		for name, param in self.named_parameters():
			if ('backbone' in name) and freeze_backbone:
				param.requires_grad = False
			elif ('aspp' in name) and freeze_aspp:
				param.requires_grad = False
	
	def trainable_parameters(self):
		return self.params_group

@register.attach('roads_net_multitask')
class RoadsNetMultitask(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=15.0,
					layer1_channels=64, layer2_channels=64, 
					decoder_1_grid_size=7, decoder_1_planes=[128, 64],
					decoder_2_kernel_size=3, decoder_2_planes=[128, 128], norm_groups=8,
					stage2=False, fuse_planes=[256, 256, 256], fuse_kernel_size=3, fuse_dilation=2,
					freeze_backbone=False, freeze_aspp=False, 
					freeze_decoder_1=False, freeze_decoder_2=False,):

		super(RoadsNetMultitask, self).__init__(n_classes, pretrained_model, 
													freeze_backbone=freeze_backbone,
													freeze_aspp=freeze_aspp)

		self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
		self.n_angles = len(self.rot_angles)

		# Decoder 1

		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, layer1_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer1_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		decoder_1_planes = [256+layer1_channels] + decoder_1_planes
		kernel_size_ori = decoder_1_grid_size + 8
		self.ori_net = create_convnet(OrientedConv2d, 
										decoder_1_planes, 
										kernel_size_ori, 
										padding(1, decoder_1_grid_size), 
										dilation=1, 
										groups=norm_groups,
										min_angle=min_angle,
										max_angle=max_angle,
										angle_step=angle_step)

		self.decoder_1_clf = nn.Conv2d(decoder_1_planes[-1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.decoder_1_clf)

		# Decoder 2

		self.reduce_layer2 = nn.Sequential(
			nn.Conv2d(256, layer2_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer2_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer2.apply(init_conv)

		decoder_2_planes = [256+layer2_channels] + decoder_2_planes
		self.decoder_2_net = create_convnet(nn.Conv2d, 
										decoder_2_planes, 
										decoder_2_kernel_size, 
										padding(1, decoder_2_kernel_size), 
										dilation=1, 
										groups=norm_groups,)

		self.decoder_2_clf = nn.Conv2d(decoder_2_planes[-1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.decoder_2_clf)

		self.reduce_decoder_1 = None
		self.fuse_net = None
		self.roads_clf = None

		if stage2:

			decoder_1_out = (self.n_angles - 1) * decoder_1_planes[-1]
			self.reduce_decoder_1 = nn.Sequential(
										nn.Conv2d(decoder_1_out, 256, kernel_size=1, stride=1, bias=False),
										nn.GroupNorm(norm_groups, 256),
										nn.ReLU(),
										nn.Dropout(0.5))
			self.reduce_decoder_1.apply(init_conv)

			self.fuse_net = create_convnet(nn.Conv2d, 
										fuse_planes, 
										fuse_kernel_size, 
										padding(fuse_dilation, fuse_kernel_size), 
										dilation=fuse_dilation, 
										groups=norm_groups,)

			in_channels_roads = fuse_planes[-1] + decoder_2_planes[-1]
			self.roads_clf = nn.Sequential(
									nn.Conv2d(in_channels_roads, 256, kernel_size=1, stride=1, bias=False),
									nn.GroupNorm(norm_groups, 256),
									nn.ReLU(),
									nn.Dropout(0.1,),
									nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False))
			self.roads_clf.apply(init_conv)



	def _forward_dir(self, x, idx):
		return forward_ori(self.ori_net, x, idx) + forward_ori(self.ori_net, x, idx+1)


	def eval(self,):
		super(RoadsNetMultitask, self).eval()
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()


	def trainable_parameters(self, base_lr, alfa=10,):

		params_groups_2 = list(super(RoadsNetMultitask, self).trainable_parameters())

		decoder_1_params = []
		decoder_1_params += list(self.reduce_layer1.parameters())
		decoder_1_params += list(self.ori_net.parameters())
		decoder_1_params += list(self.decoder_1_clf.parameters())

		decoder_2_params = []
		decoder_2_params += list(self.reduce_layer2.parameters())
		decoder_2_params += list(self.decoder_2_net.parameters())
		decoder_2_params += list(self.decoder_2_clf.parameters())

		total_params_groups = [{'params': iter(decoder_1_params+decoder_2_params), 'lr': base_lr}]
		if len(params_groups_2) > 0:
			total_params_groups += [{'params': iter(params_groups_2), 'lr': base_lr / alfa}]
		return total_params_groups


	def forward_decoder1(self, x_aspp, x_layer1):

		layer1_size = x_layer1.shape[-2:]
		x_layer1 = self.reduce_layer1(x_layer1)
		x_aspp_up = F.interpolate(x_aspp, size=layer1_size, mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_layer1, x_aspp_up], dim=1)

		x_ori = []
		for idx in np.arange(self.n_angles-1):
			x_ori.append(self._forward_dir(x_concat, idx))

		return torch.cat(x_ori, dim=0)

	
	def forward_decoder2(self, x_aspp, x_layer2):
		x_layer2 = self.reduce_layer2(x_layer2)
		x_concat = torch.cat([x_layer2, x_aspp], dim=1)
		return self.decoder_2_net(x_concat)


	def forward(self, inputs, return_intermediate=False):

		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNetMultitask, self).forward(inputs, return_intermediate=True)
		pdb.set_trace()
		x_aspp = features["aspp"]
		x_out1 = self.forward_decoder1(x_aspp, features["layer1"])
		x_out2 = self.forward_decoder2(x_aspp, features["layer2"])

		seg_1 = compute_seg(x_out1, input_shape, self.decoder_1_clf)
		seg_2 = compute_seg(x_out2, input_shape, self.decoder_2_clf)

		result = OrderedDict()
		if self.training:
			result["seg_1"] = seg_1
			result["seg_2"] = seg_2

		return result


class OrientedConv2d(_ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size,
				 padding, dilation, stride=1, groups=1, bias=False, min_angle=0.0, max_angle=180.0, angle_step=30.0):

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(OrientedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias, 'zeros')
		
		kernel_size = kernel_size[0]
		grid_size = kernel_size - 8
		grids = compute_grids((min_angle, max_angle), angle_step, kernel_size, grid_size)
		self.register_buffer('grids', grids)
		self.idx = (len(self.grids) - 1) // 2 
		self.rotated_weights = []

	def set_grid(self, idx):
		self.idx = idx
	
	def eval(self,):
		super(OrientedConv2d, self).eval()
		for grid in self.grids:
			grid = grid.unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)
			self.rotated_weights.append(F.grid_sample(self.weight, grid))


	def rotate_weight(self,):
		if len(self.rotated_weights) > 0:
			return self.rotated_weights[self.idx]
		else:
			grid = self.grids[self.idx].unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)
			return F.grid_sample(self.weight, grid)

	def forward(self, input_features):
		return F.conv2d(input_features, self.rotate_weight(), self.bias, self.stride,
						self.padding, self.dilation, self.groups)

	def reset_parameters(self):
		nn.init.xavier_normal_(self.weight)
		if self.bias is not None:
			nn.init.zeros_(self.bias)

if __name__ == "__main__":
	pdb.set_trace()
	print("Done")