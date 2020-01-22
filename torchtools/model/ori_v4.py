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


def compute_seg(x, output_shape, classifier):
	x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
	return classifier(x)


def compute_grids(angle_range, angle_step, kernel_size, grid_size):

	min_angle, max_angle = angle_range

	center = (kernel_size - 1) // 2
	k = center - (grid_size - 1) // 2
	
	angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
	grids = []
	for angle in (np.pi/2 - angles).tolist():
		theta = torch.zeros(1, 2, 3)
		theta[:, :, :2] = torch.tensor([[np.cos(angle), -1.0*np.sin(angle)],
									[np.sin(angle), np.cos(angle)]])
		grid = F.affine_grid(theta, (1,1) + _pair(kernel_size))
		grid = grid.squeeze(0)[k:-k, k:-k].unsqueeze(0)
		grids.append(grid)

	return torch.cat(grids, dim=0)


def padding(dilation, kernel_size):
	return dilation * (kernel_size - 1) // 2


def create_convnet(conv_layer, planes, kernel_size, padding, dilation, groups):

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
								dilation=dilation,)
		_init_conv(_conv)
		norm = nn.GroupNorm(groups, out_planes)
		relu = nn.ReLU()
		layers += [_conv, norm, relu]

	return nn.Sequential(*layers)


def forward_ori(ori_net, x, grid):
	for module in ori_net._modules.values():
		if isinstance(module, OrientedConv2d):
			module.set_grid(grid)
	return ori_net(x)


class Deeplabv3_ori(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, freeze_params=False):
		super(Deeplabv3_ori, self).__init__(n_classes, pretrained_model, aux=False)
		if freeze_params:
			self.freeze_parameters()
	
	def freeze_parameters(self):
		for param in self.parameters():
			param.requires_grad = False


class OrientedNet_2dir(Deeplabv3_ori):
	def __init__(self, n_classes, pretrained_model,
					min_angle=-45.0, max_angle=45.0, angle_step=15.0, 
					grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32],
					freeze_params=True, norm_groups=8, aux=True):

		super(OrientedNet_2dir, self).__init__(n_classes, pretrained_model, freeze_params=freeze_params)
		angle_range_v = np.array([min_angle, max_angle])
		angle_range_h = angle_range_v + 90.0

		kernel_size = grid_size + (grid_size - 1) * 2
		self.grids_v = compute_grids(angle_range_v.tolist(), angle_step, kernel_size, grid_size)
		self.grids_h = compute_grids(angle_range_h.tolist(), angle_step, kernel_size, grid_size)

		padding_ori = padding(dilation, grid_size)
		self.ori_net = create_convnet(OrientedConv2d, ori_planes, kernel_size, padding_ori, dilation, norm_groups)

		self.aux_clf_ori = None
		if aux:
			self.aux_clf_ori = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)


	def to(self, device):
		device_model = super(OrientedNet_2dir, self).to(device)
		device_model.grids_v = self.grids_v.to(device)
		device_model.grids_h = self.grids_h.to(device)
		return device_model

	def _forward_dir(self, x, grids, idx):
		return forward_ori(self.ori_net, x, grids[idx]) + forward_ori(self.ori_net, x, grids[idx+1])
	
	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet_2dir, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]
		seg_multi = compute_seg(x_decoder, input_shape, self.classifier)

		x_v = []
		x_h = []
		for idx, x in zip(inputs["angle_range_label"], x_decoder):
			idx = 0 if (idx == 255) else idx
			
			x = x.unsqueeze(0)
			x_v.append(self._forward_dir(x, self.grids_v, idx))
			x_h.append(self._forward_dir(x, self.grids_h, idx))

		x_v = torch.cat(x_v, dim=0)
		features["x_v"] = x_v
		x_h = torch.cat(x_h, dim=0)
		features["x_h"] = x_h

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["seg"] = seg_multi
			if self.aux_clf_ori is not None:
				seg_v = compute_seg(x_v, input_shape, self.aux_clf_ori)
				seg_h = compute_seg(x_h, input_shape, self.aux_clf_ori)
				result["out"]["seg_v"] = seg_v
				result["out"]["seg_h"] = seg_h
		else:
			result["seg"] = seg_multi

		return result, features, input_shape



@register.attach("ori_net_2dir_max1")
class OrientedNet_2dir_max1(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, 
					aux=True, grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32]):
		super(OrientedNet_2dir_max1, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, aux=aux)
		
		self.ori_clf = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)

	
	def forward(self, inputs, return_intermediate=False):
		
		result, features, input_shape = super(OrientedNet_2dir_max1, self).forward(inputs)
		x_v, x_h = features["x_v"], features["x_h"]
		lines_seg = compute_seg(torch.max(x_v, x_h), input_shape, self.ori_clf)

		if self.training:
			result["out"]["lines_seg"] = lines_seg
		else:
			_, result["seg"] = predict(result["seg"], lines_seg.squeeze())
			result["seg"] = result["seg"].cpu()

		return result


@register.attach("ori_net_2dir_max2")
class OrientedNet_2dir_max2(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, 
					aux=True, grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32], norm_groups=8):
		super(OrientedNet_2dir_max2, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, norm_groups=norm_groups)
		
		decoder_channels = ori_planes[-1] // 4
		self.reduce_decoder = nn.Sequential(
			nn.Conv2d(256, decoder_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, decoder_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_decoder.apply(init_conv)

		self.ori_clf = nn.Conv2d(ori_planes[-1] + decoder_channels, 1, kernel_size=1, stride=1, bias=False)

	def forward(self, inputs):
		
		result, features, input_shape = super(OrientedNet_2dir_max2, self).forward(inputs)
		x_v, x_h = features["x_v"], features["x_h"]
		x_decoder = self.reduce_decoder(features["decoder"])
		x_cat = torch.cat([torch.max(x_v, x_h), x_decoder], dim=1)
		lines_seg = compute_seg(x_cat, input_shape, self.ori_clf)

		if self.training:
			result["out"]["lines_seg"] = lines_seg
		else:
			_, result["seg"] = predict(result["seg"], lines_seg.squeeze())
			result["seg"] = result["seg"].cpu()

		return result

@register.attach("ori_net_2dir_concat1")
class OrientedNet_2dir_concat1(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, 
					aux=True, grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=7, fuse_dilation=2, fuse_planes=[64, 64, 64, 64], norm_groups=8):
		super(OrientedNet_2dir_concat1, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, norm_groups=norm_groups)
		fuse_padding = padding(fuse_dilation, fuse_kernel_size)
		self.fuse_net = create_convnet(nn.Conv2d, fuse_planes, fuse_kernel_size, fuse_padding, fuse_dilation, norm_groups)

		self.ori_clf = nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False)
		
	def forward(self, inputs):
		
		result, features, input_shape = super(OrientedNet_2dir_concat1, self).forward(inputs)
		x_v, x_h = features["x_v"], features["x_h"]
		x_fuse = self.fuse_net(torch.cat([x_v, x_h], dim=1))
		lines_seg = compute_seg(x_fuse, input_shape, self.ori_clf)

		if self.training:
			result["out"]["lines_seg"] = lines_seg
		else:
			_, result["seg"] = predict(result["seg"], lines_seg.squeeze())
			result["seg"] = result["seg"].cpu()

		return result

@register.attach("ori_net_2dir_concat2")
class OrientedNet_2dir_concat2(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, 
					aux=True, grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=7, fuse_dilation=2, fuse_planes=[64, 64, 64, 64], norm_groups=8):
		super(OrientedNet_2dir_concat2, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, norm_groups=norm_groups)
		decoder_channels = fuse_planes[-1] // 4
		self.reduce_decoder = nn.Sequential(
			nn.Conv2d(256, decoder_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, decoder_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_decoder.apply(init_conv)
		
		fuse_padding = padding(fuse_dilation, fuse_kernel_size)
		self.fuse_net = create_convnet(nn.Conv2d, fuse_planes, fuse_kernel_size, fuse_padding, fuse_dilation, norm_groups)

		self.ori_clf = nn.Conv2d(fuse_planes[-1] + decoder_channels, 1, kernel_size=1, stride=1, bias=False)
		
	def forward(self, inputs):
		
		result, features, input_shape = super(OrientedNet_2dir_concat2, self).forward(inputs)
		x_v, x_h = features["x_v"], features["x_h"]
		x_fuse = self.fuse_net(torch.cat([x_v, x_h], dim=1))
		x_decoder = self.reduce_decoder(features["decoder"])
		x_cat = torch.cat([x_fuse, x_decoder], dim=1)
		lines_seg = compute_seg(x_cat, input_shape, self.ori_clf)

		if self.training:
			result["out"]["lines_seg"] = lines_seg
		else:
			_, result["seg"] = predict(result["seg"], lines_seg.squeeze())
			result["seg"] = result["seg"].cpu()

		return result


class OrientedConv2d(_ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size,
				 padding, dilation, stride=1, groups=1, bias=False):

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(OrientedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias, 'zeros')

		self.grid = None

	def set_grid(self, grid):
		self.grid = grid.unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)

	def rotate_weight(self,):
		if self.grid is not None:
			return F.grid_sample(self.weight, self.grid)
		else:
			return self.weight

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