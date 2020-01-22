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

def compute_grids(min_angle, max_angle, angle_step, kernel_size, grid_size):

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
			layers += [conv, norm, relu]

		return nn.Sequential(*layers)
	
def forward_ori(self, ori_net, x, grid):
	for module in ori_net._modules.values():
		if isinstance(module, OrientedConv2d):
			module.set_grid(grid)
	return ori_net(x)


class Deeplabv3_ori(Deeplabv3):
	def __init__(self, n_classes, pretrained_model, freeze_params=False):
		super(Deeplabv3_ori, self).__init__(n_classes, pretrained_model, aux=False)
		self.aspp = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])
		if freeze_params:
			self.freeze_parameters()
	
	def freeze_parameters(self):
		for param in self.parameters():
			param.requires_grad = False

class ResidualBlock(nn.Module):
	def __init__(self, in_planes, planes, groups=8, dilation=2):
		super(ResidualBlock, self).__init__()
		self.conv_net = nn.Sequential(
			nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
			nn.GroupNorm(groups, planes),
			nn.ReLU(),
			nn.Conv2d(planes, in_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
			nn.GroupNorm(groups, in_planes)
			)
	def forward(self, x):
		return F.relu(x + self.conv_net(x))

@register.attach("ori_net_v3")
class OrientedNet(Deeplabv3_ori):
	def __init__(self, n_classes, pretrained_model, aux=True,
					min_angle=-45.0, max_angle=45.0, angle_step=15.0, 
					grid_size=5, dilation=2, ori_planes=[64, 64, 64, 32],
					fuse_kernel_size=5, fuse_dilation=6, fuse_planes = [64, 64, 64, 64], 
					freeze_params=False, norm_groups=8, grouped_conv=False):

		super(OrientedNet, self).__init__(n_classes, pretrained_model, freeze_params=freeze_params)

		conv_groups = norm_groups if grouped_conv else 1
		kernel_size = grid_size + (grid_size - 1) * 2
		channels_layer1 = ori_planes[0]

		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, channels_layer1, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, channels_layer1),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		self.grids_v, self.grids_h = self.compute_grids(min_angle, max_angle, angle_step, kernel_size, grid_size)

		self.ori_convs = self.create_ori_convs(ori_planes, kernel_size, grid_size, dilation, norm_groups, conv_groups)

		self.fuse_net = self.create_fuse_net(fuse_planes, fuse_kernel_size, fuse_dilation, norm_groups, conv_groups)
		self.fuse_net.apply(init_conv)

		def _group_concat(x, y):
			x = torch.chunk(x, norm_groups, dim=1)
			y = torch.chunk(y, norm_groups, dim=1)
			out = []
			for _x, _y in zip(x, y):
				out += [_x, _y]
			return torch.cat(out, dim=1)
			
		def _concat(x, y):
			return torch.cat([x, y], dim=1)

		self.concat = _group_concat if grouped_conv else _concat

		self.aux_clf_ori = None
		if grouped_conv:
			layers = [nn.Conv2d(fuse_planes[-1], norm_groups, kernel_size=1, stride=1, bias=False, groups=norm_groups)]
			layers += [nn.GroupNorm(1, norm_groups), nn.ReLU()]
			layers += [nn.Conv2d(norm_groups, 1, kernel_size=1, stride=1, bias=False)]
			self.line_clf_ori = nn.Sequential(*layers)
			self.line_clf_ori.apply(init_conv)
			if aux:
				layers = [nn.Conv2d(ori_planes[-1], norm_groups, kernel_size=1, stride=1, bias=False, groups=norm_groups)]
				layers += [nn.GroupNorm(1, norm_groups), nn.ReLU()]
				layers += [nn.Conv2d(norm_groups, 1, kernel_size=1, stride=1, bias=False)]
				self.aux_clf_ori = nn.Sequential(*layers)
				self.aux_clf_ori.apply(init_conv)
		else:
			self.line_clf_ori = nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.line_clf_ori)
			if aux:
				self.aux_clf_ori = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)
				init_conv(self.aux_clf_ori)

	
	def padding(self, dilation, kernel_size):
		return dilation * (kernel_size - 1) // 2


	def create_ori_convs(self, ori_planes, kernel_size, grid_size, dilation, norm_groups, conv_groups):

		n_layers = len(ori_planes) - 1
		layers = []
		for idx in np.arange(n_layers):
			in_planes = ori_planes[idx]
			out_planes = ori_planes[idx+1]
			ori_conv = OrientedConv2d(in_planes, out_planes, 
									kernel_size=kernel_size, 
									padding=self.padding(dilation, grid_size), 
									dilation=dilation,
									groups=conv_groups)
			ori_conv.reset_parameters()
			norm = nn.GroupNorm(norm_groups, out_planes)
			relu = nn.ReLU()
			layers += [ori_conv, norm, relu]

		return nn.Sequential(*layers)
	
	
	def create_fuse_net(self, fuse_planes, grid_size, dilation, norm_groups, conv_groups):
		n_layers = len(fuse_planes) - 1
		layers = []
		for idx in np.arange(n_layers):
			in_planes = fuse_planes[idx]
			out_planes = fuse_planes[idx+1]
			conv = nn.Conv2d(in_planes, out_planes, 
						kernel_size=grid_size, 
						stride=1, 
						padding=self.padding(dilation, grid_size), 
						dilation=dilation, 
						bias=False,
						groups=conv_groups)
			norm = nn.GroupNorm(norm_groups, out_planes)
			relu = nn.ReLU()
			layers += [conv, norm, relu]

		return nn.Sequential(*layers)


	def to(self, device):
		device_model = super(OrientedNet, self).to(device)
		device_model.grids_v = self.grids_v.to(device)
		device_model.grids_h = self.grids_h.to(device)
		return device_model
	
	
	def compute_grids(self, min_angle, max_angle, angle_step, kernel_size, grid_size):

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
		grids = torch.cat(grids, dim=0)

		n_angles = (len(angles) - 1) // 2 + 1
		return grids[n_angles-1:], grids[:n_angles]


	def forward_ori(self, x, grids, idx):

		def set_up(grid):
			for module in self.ori_convs._modules.values():
				if isinstance(module, OrientedConv2d):
					module.set_grid(grid)

		set_up(grids[idx])
		x_1 = self.ori_convs(x)
		set_up(grids[idx+1])
		x_2 = self.ori_convs(x)
		return x_1 + x_2


	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet, self).forward(inputs, return_intermediate=True)
		x_aspp = features["aspp"]
		seg_multi = compute_seg(x_aspp, input_shape, self.classifier)

		x_layer1 = self.reduce_layer1(features['layer1'])

		x_v = []
		x_h = []
		for idx, x in zip(inputs["angle_range_label"], x_layer1):

			idx = 0 if (idx == 255) else idx

			x = x.unsqueeze(0)
			x_v.append(self.forward_ori(x, self.grids_v, idx))
			x_h.append(self.forward_ori(x, self.grids_h, idx))

		x_v = torch.cat(x_v, dim=0)
		x_h = torch.cat(x_h, dim=0)

		x_cat = self.concat(x_v, x_h)
		x_fuse = self.fuse_net(x_cat)

		lines_seg = compute_seg(x_fuse, input_shape, self.line_clf_ori)

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["seg"] = seg_multi
			result["out"]["lines_seg"] = lines_seg
			if self.aux_clf_ori is not None:
				seg_v = compute_seg(x_v, input_shape, self.aux_clf_ori)
				seg_h = compute_seg(x_h, input_shape, self.aux_clf_ori)
				result["out"]["seg_v"] = seg_v
				result["out"]["seg_h"] = seg_h
		else:
			_, result["seg_mask_v2"] = predict(seg_multi, lines_seg.squeeze())
			result["seg_mask_v2"] = result["seg_mask_v2"].cpu()

			seg_v = compute_seg(x_v, input_shape, self.aux_clf_ori).cpu().squeeze().numpy()
			seg_h = compute_seg(x_h, input_shape, self.aux_clf_ori).cpu().squeeze().numpy()
			# plt.figure()
			# plt.imshow(seg)
			# plt.title("Seg")
			# plt.figure()
			plt.imshow(seg_v)
			plt.title("Seg V")
			plt.figure()
			plt.imshow(seg_h)
			plt.title("Seg H")
			plt.show()

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