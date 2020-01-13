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

class GaborNet(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0, 
					   kernel_size=35, in_planes=64, out_planes=128):

		super(GaborNet, self).__init__()

		angles =  np.deg2rad(np.arange(min_angle, max_angle + 90.0, angle_step))
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		""" angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
		num_angles = len(angles_v)
		self.plot_gabor(filter_weights[:num_angles], angles_v)
		plt.show() """

		depthwise_conv = self.gabor_bank(in_planes, filter_weights, kernel_size)
		n_angles = len(angles) // 2
		self.depthwise_conv_v = depthwise_conv[:n_angles]
		self.depthwise_conv_h = depthwise_conv[n_angles:]
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

	def forward(self, x, angle_indices):

		# Por ahora solo batch size de uno

		def separable_conv(x, idx):
			x_v = self.pointwise_conv(self.depthwise_conv_v[idx](x))
			x_h = self.pointwise_conv(self.depthwise_conv_h[idx](x))
			return self.relu(torch.max(x_v, x_h))

		pdb.set_trace()
		idx0, idx1 = angle_indices
		return separable_conv(x, idx0) + separable_conv(x, idx1)

	

@register.attach('angle_net_v2_2')
class AngleNet(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		super(AngleNet, self).__init__(n_classes, 
										  pretrained_model, 
										  aux=False, 
										  out_planes_skip=out_planes_skip)

		self.lines_decoder = DeepLabDecoder(256, out_planes=out_planes_skip, out_planes_decoder=64)
		self.lines_decoder.apply(init_conv)
		self.gabor_net = GaborNet(in_planes=64, out_planes=128) # no inicializar, ya se inicializa solo
		self.lines_clf = nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.lines_clf)
		self.test_tresh = 0.5
	
	def plot_out(self, x_out):
		for _x_out in x_out:
			plt.figure()
			plt.imshow(_x_out.cpu().numpy())

	def forward(self, inputs):

		def compute_seg(x, output_shape, classifier):
			x = F.interpolate(x, size=output_shape,
						mode='bilinear', align_corners=False)
			return classifier(x)

		pdb.set_trace() # Falta comprobar lo de los angulos n_angles

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet, self).forward(inputs, return_intermediate=True)

		seg = compute_seg(features["decoder"], input_shape, self.classifier)

		x_aspp = features['aspp']
		x_low = features['layer1']
		x_decoder = self.lines_decoder(x_aspp, x_low)
		x_gabor = self.gabor_net(x_decoder, inputs['angle_indices'])
		line_seg = compute_seg(x_gabor, input_shape, self.lines_clf)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['line_seg'] = line_seg
			result['out']['seg'] = seg

		return result






