import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchtools.lines as lines
from .deeplab import Deeplabv3Plus, init_conv, DeepLabDecoder
import copy
from torchtools.utils import check_gradient
from sklearn.cluster import MeanShift
from .register import register

def gabor(theta, sigma_x=0.075, sigma_y=0.75, Lambda=0.2, psi=0.0, kernel_size=51):

	y, x = np.meshgrid(np.linspace(-0.5, 0.5, kernel_size), np.linspace(-0.5, 0.5, kernel_size))

	# Rotation
	x_theta = x * np.cos(theta) - y * np.sin(theta)
	y_theta = x * np.sin(theta) + y * np.cos(theta)

	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
	
	return gb.astype(np.float32)

def split(x, combine):
	if combine:
		x = x[:-1] + x[1:]
		n_intervals = x.shape[0] // 2
		x_v = x[:n_intervals].unsqueeze(0)
		x_h = x[n_intervals:].unsqueeze(0)
	else:
		n_angles = (x.shape[0] - 1) // 2 + 1
		x_v = x[:n_angles].unsqueeze(0)
		x_h = x[n_angles-1:].unsqueeze(0)
	return torch.cat([x_v, x_h], 0)


class GaborNet_v1(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0, 
					   kernel_size=35, planes=[32, 16, 8], combine=False):

		super(GaborNet_v1, self).__init__()

		angles =  np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		""" angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
		num_angles = len(angles_v)
		self.plot_gabor(filter_weights[:num_angles], angles_v)
		plt.show() """

		self.depthwise_conv = []
		self.pointwise_conv = []
		self.n_layers = len(planes) - 1
		for idx in np.arange(self.n_layers):
			in_planes = planes[idx]
			out_planes = planes[idx + 1]
			self.depthwise_conv.append(self.gabor_bank(in_planes, filter_weights, kernel_size))
			self.pointwise_conv.append(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
		self.depthwise_conv = nn.ModuleList(self.depthwise_conv)
		self.pointwise_conv = nn.ModuleList(self.pointwise_conv)
		self.pointwise_conv.apply(init_conv)
		
		self.relu = nn.ReLU()

		self.clf = nn.Conv2d(planes[-1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.clf)
		
		self.combine = combine

	def plot_gabor(self, filter_weights, angles_v):

		for gf, theta in zip(filter_weights, angles_v):
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))

	
	def gabor_bank(self, in_planes, filter_weights, kernel_size):

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

	def parameters(self):
		params = []
		for _pointwise_conv in self.pointwise_conv:
			params += list(_pointwise_conv.parameters())
		params += list(self.clf.parameters())
		return iter(params)

	def forward(self, x, output_shape):

		# Por ahora solo batch size de uno

		def separable_conv_0(x, depthwise_conv, pointwise_conv):
			res = []
			for _depthwise_conv in depthwise_conv:
				x_depth = _depthwise_conv(x)
				x_point = pointwise_conv(x_depth)
				res.append(x_point)
			return self.relu(torch.cat(res, 0))

		def separable_conv_1(x, depthwise_conv, pointwise_conv):
			res = []
			for _depthwise_conv, _x in zip(depthwise_conv, x):
				_x = _x.unsqueeze(0)
				x_depth = _depthwise_conv(_x)
				x_point = pointwise_conv(x_depth)
				res.append(x_point)
			return self.relu(torch.cat(res, 0))

		x_0 = separable_conv_0(x, self.depthwise_conv[0], self.pointwise_conv[0])
		x_1 = separable_conv_1(x_0, self.depthwise_conv[1], self.pointwise_conv[1])
		x_1_up = F.interpolate(x_1, size=output_shape, mode='bilinear', align_corners=False)
		x_out = self.clf(x_1_up)
		pdb.set_trace()
		return split(x_out, self.combine)


class GaborClassfier_v2(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0,
					   kernel_size=35, in_planes=256, combine=False):

		super(GaborClassfier_v2, self).__init__()

		angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		self.depthwise_conv = self.gabor_bank(in_planes, filter_weights, kernel_size)
		self.pointwise_conv = nn.Conv2d(
		    in_planes, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.pointwise_conv)

		self.combine = combine

	def plot_gabor(self, filter_weights, angles_v):

		for gf, theta in zip(filter_weights, angles_v):
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))

	def gabor_bank(self, in_planes, filter_weights, kernel_size):
		_gabor_bank = []
		for _filter_weights in filter_weights:
			conv_layer = nn.Conv2d(in_planes, in_planes,
									kernel_size=kernel_size,
									stride=1,
									bias=False,
									groups=in_planes,
									padding=int((kernel_size - 1) // 2))
			_filter_weights = np.repeat(_filter_weights[np.newaxis, ...], in_planes, 0)
			_filter_weights = torch.FloatTensor(
			    _filter_weights).view_as(conv_layer.weight.data)
			conv_layer.weight = nn.Parameter(_filter_weights, requires_grad=True)
			_gabor_bank.append(conv_layer)
		return nn.ModuleList(_gabor_bank)

	def parameters(self):
		return self.pointwise_conv.parameters()

	def forward(self, x, output_shape):
		res = []
		for _depthwise_conv in self.depthwise_conv:
			x_depth = _depthwise_conv(x)
			x_depth = F.interpolate(x_depth, size=output_shape, mode='bilinear', align_corners=False)
			x_point = self.pointwise_conv(x_depth)
			res.append(x_point)
		res = torch.cat(res, 0).squeeze(1)

		return split(res, self.combine)


class GaborClassfier_v3(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0,
				 kernel_size=35, in_planes=256, combine=False):

		super(GaborClassfier_v3, self).__init__()

		angles = angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		self.gabor_bank = nn.Conv2d(
			1, len(angles), kernel_size=kernel_size, stride=1, bias=False)
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(
				gabor(angle, kernel_size=kernel_size) / kernel_size)
		filter_weights = np.stack(filter_weights, 0)
		filter_weights = torch.Tensor(filter_weights).view_as(
			self.gabor_bank.weight.data)
		self.gabor_bank.weight = nn.Parameter(
			filter_weights, requires_grad=True)

		self.clf = nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.clf)

		self.combine = combine

	def plot_gabor(self, filter_weights, angles_v):

		for gf, theta in zip(filter_weights, angles_v):
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))

	def parameters(self):
		return self.clf.parameters()

	def forward(self, x, output_shape):
		x = self.gabor_bank(self.clf(x)).squeeze(0)
		x_up = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
		return split(x_up, self.combine)


class GaborNet_v4(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0, 
					   kernel_size=35, in_planes=64, out_planes=128):

		super(GaborNet_v4, self).__init__()

		angles =  np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		""" angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
		num_angles = len(angles_v)
		self.plot_gabor(filter_weights[:num_angles], angles_v)
		plt.show() """

		depthwise_conv = self.gabor_bank(in_planes, filter_weights, kernel_size)
		n_angles = (len(angles) - 1) // 2 + 1
		self.depthwise_conv_v = depthwise_conv[:n_angles]
		self.depthwise_conv_h = depthwise_conv[n_angles-1:]
		self.pointwise_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
		init_conv(self.pointwise_conv)
		
		self.relu = nn.ReLU()

	def plot_gabor(self, filter_weights, angles_v):

		for gf, theta in zip(filter_weights, angles_v):
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))

	
	def gabor_bank(self, in_planes, filter_weights, kernel_size):

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

	def parameters(self):
		return self.pointwise_conv.parameters()

	def forward(self, x, angle_idx):

		# Por ahora solo batch size de uno

		def separable_conv(x, idx):
			x_v = self.pointwise_conv(self.depthwise_conv_v[idx](x))
			x_h = self.pointwise_conv(self.depthwise_conv_h[idx](x))
			return self.relu(torch.max(x_v, x_h))

		pdb.set_trace()
		return separable_conv(x, angle_idx) + separable_conv(x, angle_idx+1)