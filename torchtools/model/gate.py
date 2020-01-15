import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from .deeplab import Deeplabv3Plus, init_conv
import copy
from .register import register
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from .gabor import gabor

class GatedGaborConv2d(_ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
				 padding=12, dilation=6, groups=1, bias=False):
		"""
		:param in_channels:
		:param out_channels:
		:param kernel_size:
		:param stride:
		:param padding:
		:param dilation:
		:param groups:
		:param bias:
		"""

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(GatedGaborConv2d, self).__init__(
			2 * in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias, 'zeros')

		min_angle = -45.0
		max_angle = 45.0
		angle_step = 15.0
		angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		gabor_bank = self.set_up_gabor(in_channels, angles, kernel_size=35)
		n_angles = (len(angles) - 1) // 2 + 1
		self.gabor_bank_v = gabor_bank[:n_angles]
		self.gabor_bank_h = gabor_bank[n_angles-1:]

		self._gate_conv = nn.Sequential(
			nn.Conv2d(in_channels + in_channels // 4, in_channels, 1),
			nn.GroupNorm(in_channels // 8, in_channels),
			nn.ReLU(), 
			nn.Conv2d(in_channels, 1, 1),
			nn.GroupNorm(1, 1),
			nn.Sigmoid()
		)
		self._gate_conv.apply(init_conv)

		self.relu = nn.ReLU()
	
	def parameters(self,):
		pdb.set_trace()
		params = list(self._gate_conv.parameters())

	def forward(self, input_features, gating_features, idx):
		"""
		:param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
		:param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
		:return:
		"""

		if idx == 255:
			idx = 0

		def gabor_depthwise(gabor_bank):
			x_0 = gabor_bank[idx](input_features)
			x_1 = gabor_bank[idx+1](input_features)
			return self.relu(x_0) + self.relu(x_1)

		x_v = gabor_depthwise(self.gabor_bank_v)
		x_h = gabor_depthwise(self.gabor_bank_h)
		gabor_features = torch.cat([x_v, x_h], dim=1)

		alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

		return F.conv2d(gabor_features * alphas, self.weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)


	def set_up_gabor(self, in_planes, angles, kernel_size):

		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		_gabor_bank = []
		for _filter_weights in filter_weights:
			conv_layer = nn.Conv2d(in_planes, in_planes, 
									kernel_size=kernel_size, 
									stride=1, 
									bias=False, 
									groups=in_planes,
									padding=int((kernel_size  - 1) // 2))
			_filter_weights = np.repeat(_filter_weights[np.newaxis, ...], in_planes, 0)
			_filter_weights = torch.FloatTensor(_filter_weights).view_as(conv_layer.weight.data)
			conv_layer.weight =  nn.Parameter(_filter_weights, requires_grad=True)
			_gabor_bank.append(conv_layer)
		return nn.ModuleList(_gabor_bank)