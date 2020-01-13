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
		self.n_angles = len(angles) // 2
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

	def plot_gabor(self, filter_weights, angles_v):

		for gf, theta in zip(filter_weights, angles_v):
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))

	def parameters(self):
		return self.clf.parameters()

	def forward(self, x, output_shape):

		x = self.gabor_bank(self.clf(x))
		x = x.transpose(0, 1)
		x_0 = F.interpolate(x[:self.n_angles].transpose(
			0, 1), size=output_shape, mode='bilinear', align_corners=False)
		x_1 = F.interpolate(x[self.n_angles:].transpose(
			0, 1), size=output_shape, mode='bilinear', align_corners=False)
		return torch.cat([x_0, x_1], 0)


@register.attach('angle_net_v4')
class AngleNet(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		super(AngleNet, self).__init__(n_classes - 1,
									   pretrained_model,
									   aux=aux,
									   out_planes_skip=out_planes_skip)

		self.gabor_clf = GaborClassfier()  # no inicializar, ya se inicializa solo
		if aux:
			self.aux_gabor_clf = GaborClassfier()
		self.test_tresh = 0.5

	def plot_out(self, x_out):
		for _x_out in x_out:
			plt.figure()
			plt.imshow(_x_out.cpu().numpy())

	def forward(self, inputs):

		def compute_seg(x, output_shape):
			x = F.interpolate(x, size=output_shape,
						mode='bilinear', align_corners=False)
			return self.classifier(x)

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet, self).forward(
			inputs, return_intermediate=True)
		x_decoder = features["decoder"]

		seg_scores = compute_seg(x_decoder, input_shape)
		gabor_scores = self.gabor_clf(x_decoder, input_shape)

		if "aux" in features:
			x_aux = features["aux"]
			seg_scores_aux = compute_seg(x_aux, input_shape)
			x_aux_up = F.interpolate(x_aux, size=x_decoder.shape[-2:], mode='bilinear', align_corners=False)
			gabor_scores_aux = self.aux_gabor_clf(x_aux_up, input_shape)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['seg'] = seg_scores
			result['out']['line_seg'] = gabor_scores
			if "aux" in features:
				result['aux'] = OrderedDict()
				result['aux']['seg'] = seg_scores_aux
				result['aux']['line_seg'] = gabor_scores_aux
		else:
			gabor_probs = torch.sigmoid(gabor_scores)
			gabor_probs = torch.clamp(
				gabor_probs[0] + gabor_probs[1], min=0.0, max=1.0)
			_, idx = gabor_probs.view(
				gabor_probs.shape[0], -1).sum(1).unsqueeze(0).max(1)
			lines_seg = gabor_probs[idx] > self.test_tresh
			# _, seg = torch.argmax(seg_scores, 1)
			plt.imshow(lines_seg.cpu().numpy().squeeze())
			plt.show()

		return result
