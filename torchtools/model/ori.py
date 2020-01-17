import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from .deeplab import Deeplabv3Plus, init_conv
import copy
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from .register import register
from .gabor import gabor

def compute_seg(x, output_shape, classifier):
	x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
	return classifier(x)

class AttentionModule(nn.Module):
	def __init__(self, channels_input, channels_gating):
		super(AttentionModule, self).__init__()

		self._gate_conv = nn.Sequential(
			nn.Conv2d(channels_input + channels_gating, channels_input, 1, bias=False),
			nn.GroupNorm(channels_input // 8, channels_input),
			nn.ReLU(), 
			nn.Conv2d(channels_input, 1, 1, bias=False),
			nn.GroupNorm(1, 1),
			nn.Sigmoid()
		)
	
	def forward(self, input_features, gating_features):
		alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
		return input_features * alphas

@register.attach("ori_net")
class OrientedNet(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,
					min_angle=-45.0, max_angle=45.0, angle_step=15.0, 
					kernel_size=9, grid_size=5, dilation=2):

		super(OrientedNet, self).__init__(n_classes, pretrained_model, 
									aux=False, 
									out_planes_skip=out_planes_skip)
		
		self.freeze_parameters()
		
		channels_layer1 = 256
		channels_decoder = channels_layer1 // 4

		self.reduce_decoder = nn.Sequential(
			nn.Conv2d(256, channels_decoder, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(channels_decoder // 8, channels_decoder),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_decoder.apply(init_conv)

		self.at_mod = AttentionModule(channels_layer1, channels_decoder)
		self.at_mod.apply(init_conv)

		self.grids_v, self.grids_h = self.compute_grids(min_angle, max_angle, angle_step, kernel_size, grid_size)

		n_out_1 = 64
		n_out_2 = 32
		self.ori_conv1 = OrientedConv2d(256, n_out_1, kernel_size=kernel_size, padding=2*dilation, dilation=dilation)
		self.ori_conv1.reset_parameters()
		self.ori_conv2 = OrientedConv2d(n_out_1, n_out_2, kernel_size=kernel_size, padding=2*dilation, dilation=dilation)
		self.ori_conv2.reset_parameters()

		in_fuse = 2 * n_out_2
		inter_fuse = in_fuse // 2
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(in_fuse, inter_fuse, kernel_size=grid_size, stride=1, padding=2*dilation, dilation=dilation, bias=False),
			nn.GroupNorm(inter_fuse // 8, inter_fuse),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(inter_fuse, inter_fuse, kernel_size=grid_size, stride=1, padding=2*dilation, dilation=dilation, bias=False),
			nn.GroupNorm(inter_fuse // 8, inter_fuse),
			nn.ReLU(),)
		self.fuse_conv.apply(init_conv)

		self.line_clf_ori = nn.Conv2d(inter_fuse, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.line_clf_ori)

		self.aux_clf_ori = nn.Conv2d(n_out_2, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.aux_clf_ori)

		self.relu = nn.ReLU()
	
	def freeze_parameters(self):
		for param in self.parameters():
			param.requires_grad = False
	
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
	
	def forward(self, inputs):

		def _ori_conv(x, conv, grids, idx):
			x_1 = self.relu(conv(x, grids[idx]))
			x_2 = self.relu(conv(x, grids[idx+1]))
			return x_1 + x_2

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet, self).forward(inputs, return_intermediate=True)

		x_decoder = self.reduce_decoder(features["decoder"])
		x_layer1 = self.at_mod(features['layer1'], x_decoder)

		x_v = []
		x_h = []
		x_cat = []
		for idx, x in zip(inputs["angle_range_label"], x_layer1):

			x = x.unsqueeze(0)

			x_1_v = _ori_conv(x, self.ori_conv1, self.grids_v, idx)
			x_1_h = _ori_conv(x, self.ori_conv1, self.grids_h, idx)

			x_2_v = _ori_conv(x_1_v, self.ori_conv2, self.grids_v, idx)
			x_2_h = _ori_conv(x_1_h, self.ori_conv2, self.grids_h, idx)

			x_v.append(x_2_v)
			x_h.append(x_2_h)
			x_cat.append(torch.cat([x_2_v, x_2_h], dim=1))
		
		x_v = torch.cat(x_v, dim=0)
		x_h = torch.cat(x_h, dim=0)
		x_cat = torch.cat(x_cat, dim=0)
		x_fuse = self.fuse_conv(x_cat)

		seg_v = compute_seg(x_v, input_shape, self.aux_clf_ori)
		seg_h = compute_seg(x_h, input_shape, self.aux_clf_ori)
		seg = compute_seg(x_fuse, input_shape, self.line_clf_ori)

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["seg_v"] = seg_v
			result["out"]["seg_h"] = seg_h
			result["out"]["seg"] = seg
		else:
			seg_v = torch.sigmoid(seg_v).squeeze().cpu().numpy()
			seg_h = torch.sigmoid(seg_h).squeeze().cpu().numpy()
			seg = torch.sigmoid(seg).squeeze().cpu().numpy()

			plt.figure()
			plt.imshow(seg)
			plt.title("Seg")
			plt.figure()
			plt.imshow(seg_v)
			plt.title("Seg V")
			plt.figure()
			plt.imshow(seg_h)
			plt.title("Seg H")
			plt.show()
		return result


@register.attach('test_ori')
class OrientedNetTest(OrientedNet):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		kernel_size = 35
		grid_size = 25

		super(OrientedNetTest, self).__init__(n_classes, 
											  pretrained_model, 
											  kernel_size=kernel_size, 
											  grid_size=grid_size)
		test_weight = torch.FloatTensor(gabor(np.pi/2, kernel_size))
		test_weight = test_weight.unsqueeze(0).repeat(256,1,1)
		test_weight = test_weight.unsqueeze(0).repeat(64,1,1,1)
		self.ori_conv1.weight = nn.Parameter(test_weight, requires_grad=False)
	
	def forward(self, idx):

		rotated_weight_v_1 = self.ori_conv1.rotate_weight(self.grids_v[idx])[0,0].numpy()
		rotated_weight_h_1 = self.ori_conv1.rotate_weight(self.grids_h[idx])[0,0].numpy()

		rotated_weight_v_2 = self.ori_conv1.rotate_weight(self.grids_v[idx+1])[0,0].numpy()
		rotated_weight_h_2 = self.ori_conv1.rotate_weight(self.grids_h[idx+1])[0,0].numpy()

		plt.figure()
		plt.imshow(rotated_weight_v_1 + rotated_weight_h_1)
		plt.title("Rotated 1")
		plt.figure()
		plt.imshow(rotated_weight_h_1)
		plt.title("Rotated H 1")

		plt.figure()
		plt.imshow(rotated_weight_v_2 + rotated_weight_h_2)
		plt.title("Rotated 2")
		plt.figure()
		plt.imshow(rotated_weight_h_2)
		plt.title("Rotated H 2")
	
		# for idx in np.arange(self.grids_v.shape[0]):
		# 	rotated_weight_v = self.ori_conv1.rotate_weight(self.grids_v[idx])[0,0].numpy()
		# 	rotated_weight_h = self.ori_conv1.rotate_weight(self.grids_h[idx])[0,0].numpy()
		# 	plt.figure()
		# 	plt.imshow(rotated_weight_v)
		# 	plt.title("Rotated V")
		# 	plt.figure()
		# 	plt.imshow(rotated_weight_h)
		# 	plt.title("Rotated H")
		# 	plt.show()

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
	
	def rotate_weight(self, grid,):
		grid = grid.unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)
		rotated_weight = F.grid_sample(self.weight, grid)
		return rotated_weight

	def forward(self, input_features, grid):
		return F.conv2d(input_features, self.rotate_weight(grid), self.bias, self.stride,
						self.padding, self.dilation, self.groups)
  
	def reset_parameters(self):
		nn.init.xavier_normal_(self.weight)
		if self.bias is not None:
			nn.init.zeros_(self.bias)

if __name__ == "__main__":
	pdb.set_trace()
	print("Done")