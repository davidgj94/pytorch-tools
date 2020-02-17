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
	for angle in (np.pi/2 - angles).tolist():
		grid = _compute_grid(angle)
		grids.append(grid)

	return torch.cat(grids, dim=0)


def padding(dilation, kernel_size):
	return dilation * (kernel_size - 1) // 2


def create_convnet(conv_layer, planes, kernel_size, padding, dilation, groups, **kwargs):

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


class Deeplabv3_Roads(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, freeze_all=False):
		super(Deeplabv3_Roads, self).__init__(n_classes, pretrained_model, aux=False)
		self.freeze_parameters(freeze_all)
		self.params_group = super(Deeplabv3_Roads, self).trainable_parameters(True)
	
	def freeze_parameters(self, freeze_all):
		for name, param in self.named_parameters():
			if ('backbone' in name) or ('classifier' in name) or freeze_all:
				param.requires_grad = False
	
	def trainable_parameters(self):
		return self.params_group


class RoadsNet(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=30.0, 
					grid_size=7, dilation=1, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=5, fuse_dilation=1, fuse_planes=[192, 128, 128],
					freeze_all=False, norm_groups=8, aux_ori=True, aux_junction=True):

		super(RoadsNet, self).__init__(n_classes, pretrained_model, freeze_all=freeze_all)

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
		
		padding_fuse = padding(fuse_dilation, fuse_kernel_size)
		self.fuse_net = create_convnet(nn.Conv2d, 
										fuse_planes, 
										fuse_kernel_size, 
										padding_fuse, 
										fuse_dilation, 
										norm_groups,)
		
		decoder_channels = fuse_planes[-1] // 4
		self.reduce_decoder = nn.Sequential(
			nn.Conv2d(256, decoder_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, decoder_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_decoder.apply(init_conv)

		self.roads_clf = nn.Sequential(
			nn.Conv2d(fuse_planes[-1] + decoder_channels, fuse_planes[-1], kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, fuse_planes[-1]),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False))
		self.roads_clf.apply(init_conv)

		self.aux_ori_clf = None
		if aux_ori:
			self.aux_ori_clf = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_ori_clf)
		
		self.aux_junction_clf = None
		if aux_junction:
			self.aux_junction_clf = nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_junction_clf)


	def _forward_dir(self, x, idx):
		return forward_ori(self.ori_net, x, idx) + forward_ori(self.ori_net, x, idx+1)


	def eval(self,):
		super(RoadsNet, self).eval()
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()


	def forward(self, inputs, return_intermediate=False):

		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNet, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]

		# Se asume batch size de 1
		x_ori = []
		for idx in np.arange(self.n_angles):
			x_ori.append(self._forward_dir(x_decoder, idx))
		
		if (not self.training) and (self.aux_ori_clf is not None):
			seg_ori = compute_seg(torch.cat(x_ori, dim=0), input_shape, self.aux_ori_clf).squeeze(1).unsqueeze(0)
		
		x_fuse = self.fuse_net(torch.cat(x_ori, dim=1))

		if (not self.training) and (self.aux_junction_clf is not None):
			seg_junction = compute_seg(x_fuse, input_shape, self.aux_junction_clf).squeeze(1)
		
		x_decoder = self.reduce_decoder(x_decoder)
		x_concat = torch.cat([x_fuse, x_decoder], dim=1)
		seg_roads = compute_seg(x_concat, input_shape, self.roads_clf).squeeze(1)

		result = OrderedDict()
		if self.training:
			result["seg_roads"] = seg_roads
			if self.aux_ori_clf is not None:
				result["seg_ori"] = seg_ori
			if self.aux_junction_clf is not None:
				result["seg_junction"] = seg_junction
		else:
			result["seg"] = seg_roads

		return result


@register.attach('test_roads')
class OrientedNetTest(RoadsNet):
	def __init__(self, n_classes, pretrained_model, aux=False):

		grid_size = 25
		kernel_size = grid_size + (grid_size - 1) * 2
		ori_planes=[256, 1]

		super(OrientedNetTest, self).__init__(n_classes, 
											  pretrained_model, 
											  grid_size=grid_size,
											  ori_planes=ori_planes)

		test_weight = torch.FloatTensor(gabor(np.pi/2, kernel_size))
		test_weight = test_weight.unsqueeze(0).repeat(ori_planes[0],1,1)
		test_weight = test_weight.unsqueeze(0).repeat(ori_planes[1],1,1,1)
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.weight = nn.Parameter(test_weight, requires_grad=False)

	def get_weight(self, idx):
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.set_grid(idx)
				return module.rotate_weight()
		return None


	def forward(self, inputs):

		ori_weights = inputs['ori_weights']
		for idx, weights in enumerate(ori_weights):
			weights = weights.numpy()
			if not np.all(weights < 1.0):
				rotated_filter = self.get_weight(idx)[0,0].numpy()
				plt.figure()
				plt.imshow(rotated_filter)
				plt.title("{}".format(self.rot_angles[idx]))
				rotated_filter = self.get_weight(idx+1)[0,0].numpy()
				plt.figure()
				plt.imshow(rotated_filter)
				plt.title("{}".format(self.rot_angles[idx+1]))
		plt.show()


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
		
		grid_size = kernel_size - 8
		grids = compute_grids((min_angle, max_angle), angle_step, kernel_size, grid_size)
		self.register_buffer('grids', grids)
		self.idx = (len(self.grids) - 1) // 2 
		self.rotated_weights = []

	def set_grid(self, idx):
		self.idx = idx
	
	def eval(self,):
		super(OrientedConv2d, self).eval()
		pdb.set_trace() #Comprobar tamaño de los grids
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