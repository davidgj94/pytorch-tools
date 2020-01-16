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
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

def gabor(theta, kernel_size, sigma_x=0.075, sigma_y=0.75, Lambda=0.2, psi=0.0):

	y, x = np.meshgrid(np.linspace(-0.5, 0.5, kernel_size), np.linspace(-0.5, 0.5, kernel_size))

	# Rotation
	x_theta = x * np.cos(theta) - y * np.sin(theta)
	y_theta = x * np.sin(theta) + y * np.cos(theta)

	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
	
	return gb.astype(np.float32)


def gabor_bank(kernel_size, in_planes, min_angle=-45.0, max_angle=45.0, angle_step=15.0):

	angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
	filter_bank = []
	for angle in (angles + np.pi/2).tolist():
			filter_weights = gabor(angle, kernel_size=kernel_size) / kernel_size
			conv_layer = nn.Conv2d(in_planes, in_planes, 
									kernel_size=kernel_size, 
									stride=1, 
									bias=False, 
									groups=in_planes,
									padding=int((kernel_size  - 1) // 2))
			filter_weights = np.repeat(filter_weights[np.newaxis, ...], in_planes, 0)
			filter_weights = torch.FloatTensor(filter_weights).view_as(conv_layer.weight.data)
			conv_layer.weight =  nn.Parameter(filter_weights, requires_grad=False)
			filter_bank.append(conv_layer)
	filter_bank = nn.ModuleList(filter_bank)

	n_angles = (len(angles) - 1) // 2 + 1
	return filter_bank[:n_angles], filter_bank[n_angles-1:]


def plot_gabor_bank(filter_weights, angles):
	for gf, theta in zip(filter_weights, angles):
			plt.figure()
			plt.imshow(gf.weight.squeeze().cpu().numpy())
			plt.title("Theta={}".format(theta))


class GaborNet_v4(nn.Module):

	def __init__(self, in_planes=64, out_planes=128, kernel_size=35):
		super(GaborNet_v4, self).__init__()
		self.depthwise_conv_v, self.depthwise_conv_h = gabor_bank(kernel_size, in_planes)
		self.pointwise_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
		init_conv(self.pointwise_conv)
		self.relu = nn.ReLU()
	

	def forward(self, x, angle_idx):

		# Por ahora solo batch size de uno

		def separable_conv(x, idx):
			x_v = self.pointwise_conv(self.depthwise_conv_v[idx](x))
			x_h = self.pointwise_conv(self.depthwise_conv_h[idx](x))
			return self.relu(torch.max(x_v, x_h))

		if angle_idx == 255:
			angle_idx = 0 # pongo ese mismo, porque luego no se tiene en cuenta en la loss
		""" else:
			plt.figure()
			plt.imshow(self.depthwise_conv_v[angle_idx].weight[0].squeeze().cpu().numpy())
			plt.figure()
			plt.imshow(self.depthwise_conv_v[angle_idx+1].weight[0].squeeze().cpu().numpy())
			plt.show()
 """
		return separable_conv(x, angle_idx) + separable_conv(x, angle_idx+1)


class GaborDecoder(nn.Module):
	def __init__(self, in_planes, out_planes_low=128, out_planes_decoder=128, kernel_size=35,):
		super(GaborDecoder, self).__init__()

		self.depthwise_conv_v, self.depthwise_conv_h = gabor_bank(kernel_size, in_planes)
		self.pointwise_conv = nn.Conv2d(in_planes, out_planes_low, kernel_size=1, stride=1, bias=False)
		init_conv(self.pointwise_conv)
		
		self.relu = nn.ReLU()

		self.fuse_conv = nn.Sequential(
			nn.Conv2d(256 + 2 * out_planes_low, out_planes_decoder, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(int(out_planes_decoder/8), out_planes_decoder),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(out_planes_decoder, out_planes_decoder, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(int(out_planes_decoder/8), out_planes_decoder),
			nn.ReLU())
		self.fuse_conv.apply(init_conv)


	def reduce_conv(self, x, idx):

		def gabor_branch(depthwise_conv):
			x_0 = self.pointwise_conv(depthwise_conv[idx](x))
			x_1 = self.pointwise_conv(depthwise_conv[idx+1](x))
			return self.relu(x_0) + self.relu(x_1)

		x_v = gabor_branch(self.depthwise_conv_v)
		x_h = gabor_branch(self.depthwise_conv_h)
		return torch.cat([x_v, x_h], dim=1)


	def forward(self, x, x_low, angle_idx):

		if angle_idx == 255:
			angle_idx = 0
		""" else:
			plt.figure()
			plt.imshow(self.depthwise_conv_v[angle_idx].weight[0].squeeze().cpu().numpy())
			plt.figure()
			plt.imshow(self.depthwise_conv_v[angle_idx+1].weight[0].squeeze().cpu().numpy())
			plt.show() """

		low_size = x_low.shape[-2:]
		high_size = x.shape[-2:]
		x_low = self.reduce_conv(x_low, angle_idx)
		if low_size > high_size:
			x = F.interpolate(x, size=low_size, mode='bilinear', align_corners=False)
		x = torch.cat([x_low, x], dim=1)
		x = self.fuse_conv(x)
		return x

class GatedGabor(nn.Module):
	def __init__(self, in_channels, kernel_size=35,):
		super(GatedGabor, self).__init__()

		self.gabor_bank_v, self.gabor_bank_h = gabor_bank(kernel_size=kernel_size,
																  in_planes=in_channels)

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
		
		
	def forward(self, input_features, gating_features, idx):

		alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
		input_features = input_features * alphas
		
		if idx == 255:
			idx = 0

		def gabor_depthwise(gabor_bank):
			x_0 = gabor_bank[idx](input_features)
			x_1 = gabor_bank[idx+1](input_features)
			return self.relu(x_0) + self.relu(x_1)

		x_v = gabor_depthwise(self.gabor_bank_v)
		x_h = gabor_depthwise(self.gabor_bank_h)
		gabor_features = torch.cat([x_v, x_h], dim=1)

		return gabor_features


if __name__ == "__main__":

	min_angle = -45.0
	max_angle = 45.0
	angle_step = 15.0

	angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
	angles_h = angles_v + 90.0
	gabor_bank_v, gabor_bank_h = gabor_bank(35,1)

	plot_gabor_bank(gabor_bank_v, angles_v)
	plot_gabor_bank(gabor_bank_h, angles_h)
	plt.show()
	
	print('Done')