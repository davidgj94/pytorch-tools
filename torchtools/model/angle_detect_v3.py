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

	y, x = np.meshgrid(np.linspace(-0.5, 0.5, kernel_size),
	                   np.linspace(-0.5, 0.5, kernel_size))

	# Rotation
	x_theta = x * np.cos(theta) - y * np.sin(theta)
	y_theta = x * np.sin(theta) + y * np.cos(theta)

	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 /
	            sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)

	return gb.astype(np.float32)


class GaborClassfier(nn.Module):
	def __init__(self, angle_step=15.0, min_angle=-45.0, max_angle=45.0,
					   kernel_size=35, in_planes=256,):

		super(GaborClassfier, self).__init__()

		angles = np.deg2rad(np.arange(min_angle, max_angle + 90.0, angle_step))
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size) / kernel_size)

		self.depthwise_conv = self.gabor_bank(in_planes, filter_weights, kernel_size)
		self.pointwise_conv = nn.Conv2d(
		    in_planes, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.pointwise_conv)

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
		n_angles = len(self.depthwise_conv) // 2
		res_v = res[:n_angles].unsqueeze(0)
		res_h = res[n_angles:].unsqueeze(0)
		res = torch.cat([res_v, res_h], 0)
		return res


@register.attach('angle_net_v3')
class AngleNet(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		super(AngleNet, self).__init__(n_classes - 1, 
										pretrained_model, 
										aux=False, 
										out_planes_skip=out_planes_skip)

		self.gabor_clf = GaborClassfier() # no inicializar, ya se inicializa solo
		self.test_tresh = 0.5

	def plot_out(self, x_out):
		for _x_out in x_out:
			plt.figure()
			plt.imshow(_x_out.cpu().numpy())

	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]

		x = F.interpolate(x_decoder, size=input_shape, mode='bilinear', align_corners=False)
		seg_scores = self.classifier(x)
		gabor_scores = self.gabor_clf(x_decoder, input_shape)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['seg'] = seg_scores
			result['out']['line_seg'] = gabor_scores
		else:
			gabor_probs = torch.sigmoid(gabor_scores)
			gabor_probs = torch.clamp(gabor_probs[0] + gabor_probs[1], min=0.0, max=1.0)
			_, idx = gabor_probs.view(gabor_probs.shape[0], -1).sum(1).unsqueeze(0).max(1)
			lines_seg = gabor_probs[idx] > self.test_tresh
			# _, seg = torch.argmax(seg_scores, 1)
			plt.imshow(lines_seg.cpu().numpy().squeeze())
			plt.show()

		return result






