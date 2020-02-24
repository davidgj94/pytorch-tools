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


def create_convnet(conv_layer, planes, kernel_size, padding, dilation, groups, rm_last_relu=False, **kwargs):

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
	
	if rm_last_relu:
		layers = layers[:-1]

	return nn.Sequential(*layers)


def forward_ori(ori_net, x, idx):
	for module in ori_net._modules.values():
		if isinstance(module, OrientedConv2d):
			module.set_grid(idx)
	return ori_net(x)


class Deeplabv3_Roads(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, freeze_backbone=False):
		super(Deeplabv3_Roads, self).__init__(n_classes, pretrained_model, aux=False)
		self.freeze_parameters(freeze_backbone)
	
	def freeze_parameters(self, freeze_backbone):
		for name, param in self.named_parameters():
			if ('backbone' in name) and freeze_backbone:
				param.requires_grad = False

@register.attach('roads_net_stage1_v2_2')
class RoadsNet_Stage1(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=20.0, 
					grid_size=7, dilation=1, ori_planes=[256, 128, 64, 32],
					norm_groups=8, freeze_backbone=False):

		super(RoadsNet_Stage1, self).__init__(n_classes, pretrained_model, freeze_backbone=freeze_backbone)

		self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
		self.n_angles = len(self.rot_angles)

		padding_ori = padding(dilation, grid_size)
		kernel_size_ori = grid_size + 8
		self.ori_net = create_convnet(OrientedConv2d, 
										ori_planes, 
										kernel_size_ori, 
										padding_ori, 
										dilation, 
										norm_groups,
										min_angle=min_angle,
										max_angle=max_angle,
										angle_step=angle_step)

		self.aux_ori_clf = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.aux_ori_clf)

	def compute_ori_seg(self, x_ori, output_shape):
		x_ori = torch.cat(x_ori, dim=0)
		return compute_seg(x_ori[:-1] + x_ori[1:], output_shape, self.aux_ori_clf).squeeze(1).unsqueeze(0)

	def eval(self,):
		super(RoadsNet_Stage1, self).eval()
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()


	def trainable_parameters(self, base_lr):

		params_groups = [{'params': super(RoadsNet_Stage1, self).trainable_parameters(debug=True), 'lr': base_lr}]
		return params_groups


	def forward(self, inputs, return_intermediate=False):

		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNet_Stage1, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]
		seg_roads = compute_seg(x_decoder, input_shape, self.classifier).squeeze(1)

		# Se asume batch size de 1
		x_ori = []
		for idx in np.arange(self.n_angles):
			x_ori.append(forward_ori(self.ori_net, x_decoder, idx))
		
		seg_ori = self.compute_ori_seg(x_ori, input_shape)

		if return_intermediate:
			features['ori'] = x_ori[:-1]
			features['seg_ori'] = seg_ori
			features['seg_roads'] = seg_roads
			return features

		result = OrderedDict()
		if self.training:
			result["seg_ori"] = seg_ori
			result["seg_roads"] = seg_roads
		else:
			idx = [idx for idx, weights_mask in enumerate(inputs["ori_weights"].squeeze()) if not np.all(weights_mask.numpy() < 1)].pop()
			plt.figure()
			plt.imshow(inputs['label'].squeeze())
			plt.figure()
			plt.imshow(inputs['ori_gt'].squeeze()[idx])
			plt.figure()
			plt.imshow((seg_ori > 0).cpu().squeeze()[idx])
			plt.show()
			result["seg"] = (seg_ori > 0).cpu()

		return result


@register.attach('roads_net_stage2_v2_2')
class RoadsNet_Stage2(RoadsNet_Stage1):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=20.0, 
					grid_size=7, dilation=1, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=5, fuse_dilation=1, fuse_planes=[288, 128, 128],
					res_kernel_size=5, res_dilation=1, res_planes=[384, 256, 256],
					norm_groups=8, freeze_backbone=True, aux_junction=False):

		super(RoadsNet_Stage2, self).__init__(n_classes, pretrained_model,
													min_angle=min_angle, max_angle=max_angle, angle_step=angle_step,
													grid_size=grid_size, dilation=dilation, ori_planes=ori_planes,
													norm_groups=norm_groups, freeze_backbone=freeze_backbone)

		padding_fuse = padding(fuse_dilation, fuse_kernel_size)
		self.fuse_net = create_convnet(nn.Conv2d, 
										fuse_planes, 
										fuse_kernel_size, 
										padding_fuse, 
										fuse_dilation, 
										norm_groups,)
		self.aux_junction_clf = None
		if aux_junction:
			self.aux_junction_clf = nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_junction_clf)

		padding_res = padding(res_dilation, res_kernel_size)
		self.residual_net = create_convnet(nn.Conv2d, 
										res_planes, 
										res_kernel_size, 
										padding_res, 
										res_dilation, 
										norm_groups,
										rm_last_relu=True)


	def trainable_parameters(self, base_lr, alfa=10):

		params_1 = []
		params_2 = []
		for name, param in self.named_parameters():
			if param.requires_grad:
				if ("fuse_net" in name) or ("aux_junction_clf" in name) or ("residual_net" in name) or ("ori_net" in name) or ("aux_ori_clf" in name):
					print("{} -> params_1".format(name))
					params_1.append(param)
				else:
					print("{} -> params_2".format(name))
					params_2.append(param)
		
		params_groups = [{'params': iter(params_1), 'lr': base_lr},
						 {'params': iter(params_2), 'lr': base_lr / alfa}]
		
		return params_groups


	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]

		features = super(RoadsNet_Stage2, self).forward(inputs, return_intermediate=True)
		x_decoder = features['decoder']
		x_ori = features['ori']
		seg_ori = features['seg_ori']

		x_fuse = self.fuse_net(torch.cat(x_ori, dim=1))
		if self.aux_junction_clf is not None:
			seg_junction = compute_seg(x_fuse, input_shape, self.aux_junction_clf).squeeze(1)

		x_concat = torch.cat([x_fuse, x_decoder], dim=1)
		x_res = self.residual_net(x_concat)
		x_out = F.relu(x_decoder + x_res)
		seg_roads = compute_seg(x_out, input_shape, self.classifier).squeeze(1)

		result = OrderedDict()
		if self.training:
			result['seg_ori'] = seg_ori
			result['seg_roads'] = seg_roads
			if self.aux_junction_clf is not None:
				 result['seg_junction'] = seg_junction
		else:
			result['seg'] = (seg_roads.squeeze() > 0).cpu()
		
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