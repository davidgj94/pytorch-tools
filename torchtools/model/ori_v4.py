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
	
	angles = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
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
		self.grids_h = compute_grids(angle_range_v.tolist(), angle_step, kernel_size, grid_size)
		self.grids_v = compute_grids(angle_range_h.tolist(), angle_step, kernel_size, grid_size)
		# Estan cambiados para que coincida con el criterio del dataset

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
	
	def forward(self, inputs, return_intermediate=False):

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet_2dir, self).forward(inputs, return_intermediate=True)
		if return_intermediate:
			return features
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


@register.attach('test_ori_v4')
class OrientedNetTest(OrientedNet_2dir):
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

	def get_weight(self, grid):
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.set_grid(grid)
				return module.rotate_weight()
		return None

	
	def forward(self,idx):

		rotated_weight_v = self.get_weight(self.grids_v[idx])[0,0].numpy()
		rotated_weight_h = self.get_weight(self.grids_h[idx])[0,0].numpy()
		plt.figure()
		plt.imshow(rotated_weight_v + rotated_weight_h)

		rotated_weight_v = self.get_weight(self.grids_v[idx+1])[0,0].numpy()
		rotated_weight_h = self.get_weight(self.grids_h[idx+1])[0,0].numpy()
		plt.figure()
		plt.imshow(rotated_weight_v + rotated_weight_h)

		# for idx in np.arange(self.grids_v.shape[0]):
		# 	rotated_weight_v = self.get_weight(self.grids_v[idx])[0,0].numpy()
		# 	rotated_weight_h = self.get_weight(self.grids_h[idx])[0,0].numpy()
		# 	plt.figure()
		# 	plt.imshow(rotated_weight_v)
		# 	plt.title("Rotated V")
		# 	plt.figure()
		# 	plt.imshow(rotated_weight_h)
		# 	plt.title("Rotated H")
		# 	plt.show()


@register.attach("ori_net_2dir_concat2")
class OrientedNet_2dir_concat2(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, 
					aux=True, grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=7, fuse_dilation=2, fuse_planes=[64, 64, 64, 64], norm_groups=8, freeze_params=True):
		super(OrientedNet_2dir_concat2, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, norm_groups=norm_groups,
												freeze_params=freeze_params)
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

@register.attach("ori_net_2dir_hist")
class OrientedNet_2dir_hist(OrientedNet_2dir):
	def __init__(self, n_classes, pretrained_model, aux=True,
					grid_size=5, dilation=2, ori_planes=[256, 128, 64, 32],
					fuse_kernel_size=7, fuse_dilation=2, fuse_planes=[64, 64, 64, 64], 
					norm_groups=8, freeze_params=True):
		super(OrientedNet_2dir_hist, self).__init__(n_classes, pretrained_model, 
												grid_size=grid_size, dilation=dilation, 
												ori_planes=ori_planes, norm_groups=norm_groups,
												freeze_params=freeze_params, aux=True)
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
	
	def plot_hist(self, seg, true_idx):
		seg = torch.sigmoid(seg)
		for idx, _seg in enumerate(seg.squeeze()):
			_seg = _seg.cpu().detach().numpy()
			plt.figure()
			plt.imshow(_seg)
			if idx == true_idx:
				title = "True"
			else:
				title = "False"
			plt.title(title)
		plt.show()
	
	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet_2dir_hist, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]
		seg_multi = compute_seg(x_decoder, input_shape, self.classifier)

		x_v_true = []
		x_h_true = []
		x_v = []
		x_h = []
		for true_idx, x in zip(inputs["angle_range_label"], x_decoder.unsqueeze(1)):
			x_v_aux = []
			x_h_aux = []
			for idx in np.arange(len(self.grids_v) - 1):
				x_v_aux.append(self._forward_dir(x, self.grids_v, idx))
				x_h_aux.append(self._forward_dir(x, self.grids_h, idx))
			x_v_aux = torch.cat(x_v_aux, dim=0)
			x_h_aux = torch.cat(x_h_aux, dim=0)

			x_v.append(compute_seg(x_v_aux, input_shape, self.aux_clf_ori).squeeze().unsqueeze(0))
			x_h.append(compute_seg(x_h_aux, input_shape, self.aux_clf_ori).squeeze().unsqueeze(0))
			# # Pa debuggear
			# self.plot_hist(x_v[-1], true_idx)
			# x_v_true_aux = compute_seg(x_v_aux[true_idx].unsqueeze(0), input_shape, self.aux_clf_ori).squeeze().unsqueeze(0)
			# self.plot_hist(x_v_true_aux, 0)
			# ##############

			x_v_true.append(x_v_aux[true_idx].unsqueeze(0))
			x_h_true.append(x_h_aux[true_idx].unsqueeze(0))
		
		seg_v = torch.cat(x_v, dim=0)
		seg_h = torch.cat(x_h, dim=0)

		x_v_true = torch.cat(x_v_true, dim=0)
		x_h_true = torch.cat(x_h_true, dim=0)
		x_fuse = self.fuse_net(torch.cat([x_v_true, x_h_true], dim=1))
		x_decoder = self.reduce_decoder(features["decoder"])
		x_cat = torch.cat([x_fuse, x_decoder], dim=1)
		lines_seg = compute_seg(x_cat, input_shape, self.ori_clf)

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["seg"] = seg_multi
			result["out"]["lines_seg"] = lines_seg
			result["out"]["seg_v"] = seg_v
			result["out"]["seg_h"] = seg_h
		else:
			result["seg"] = seg_multi

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