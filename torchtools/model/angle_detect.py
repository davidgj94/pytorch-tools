import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchtools.lines as lines
from .deeplab import Deeplabv3Plus, init_conv
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

class GaborBank(nn.Module):
	def __init__(self, thetas, kernel_size=51, Lambda=0.2):
		super(GaborBank, self).__init__()
		self.thetas = torch.Tensor(thetas).float()
		self.kernel_size = kernel_size
		self.Lambda = Lambda
		self.alfa = nn.Parameter(0.1 * torch.ones(1))
		self.sigma_y = 0.75

	def get_device(self,):
		return self.alfa.device

	def plot_filters(self,):
		print(">>>>>> sigma_x: {}".format(self.alfa * self.sigma_y))
		print(">>>>>> sigma_y: {}".format(self.sigma_y))
		# gabor_filters = self.compute_weigths()
		# for _filter in gabor_filters[:int(len(self.thetas)/2)]:
		# 	plt.figure()
		# 	plt.imshow(_filter.cpu().detach().numpy().squeeze())
		# plt.show()


	def compute_weigths(self,):

		y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, self.kernel_size), torch.linspace(-0.5, 0.5, self.kernel_size)])
		x = x.to(self.get_device())
		y = y.to(self.get_device())
		gabor_filters = []
		for _theta in self.thetas.to(self.get_device()):
			rotx = x * torch.cos(_theta) - y * torch.sin(_theta)
			roty = x * torch.sin(_theta) + y * torch.cos(_theta)
			gf = torch.exp(-0.5 * (rotx ** 2 / (self.sigma_y * self.alfa + 1e-3) ** 2 + roty ** 2 / (self.sigma_y + 1e-3) ** 2)) * torch.cos(2 * 3.14 * rotx / self.Lambda) 
			gabor_filters.append(gf.unsqueeze(0))
		gabor_filters = torch.cat(gabor_filters, 0).unsqueeze(1)

		return gabor_filters


	def forward(self, _input):
		# print(">>>>>> sigma_x: {}".format(self.sigma_x))
		gabor_filters = self.compute_weigths()
		return F.conv2d(_input, gabor_filters)


@register.attach('angle_net')
class AngleNet(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,
		 angle_step=15.0, min_angle=-45.0, max_angle=45.0, 
		 kernel_size=51, pos=0, detect_lines=False):

		super(AngleNet, self).__init__(n_classes, 
									   pretrained_model,
									   aux=False,
									   out_planes_skip=out_planes_skip)

		angles =  np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		self.angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
		self.num_angles = len(self.angles_v)
		self.gabor_bank = nn.Conv2d(1, len(angles), kernel_size=kernel_size, stride=1, bias=False)
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size))
		filter_weights = np.dstack(filter_weights)
		filter_weights = np.transpose(filter_weights, (2,0,1))
		filter_weights = torch.Tensor(filter_weights).view_as(self.gabor_bank.weight.data)
		self.gabor_bank.weight = nn.Parameter(filter_weights, requires_grad=True)
		self.relu = nn.ReLU()

		if detect_lines:

			self.resol = 10.0
			self.test_tresh = 0.75

			width = 7
			input_size = (width, 2 * width)
			nfilters = 100

			self.prelu = nn.PReLU(init=0.75)
			self.avg_pooling = nn.AdaptiveAvgPool2d(input_size)
			self.line_clf = nn.Sequential(
				nn.Conv2d(1, nfilters, kernel_size=input_size, stride=1, bias=False),
				nn.ReLU(),
				nn.Conv2d(nfilters, nfilters, kernel_size=1, stride=1, bias=False),
				nn.ReLU(),
				nn.Conv2d(nfilters, 1, kernel_size=1, stride=1, bias=False))
			self.line_clf.apply(init_conv)


	def plot_gabor(self, indices=None):

		gabor_filters = self.gabor_bank.weight.data[:self.num_angles]
		gabor_angles = self.angles_v.copy()
		if indices is not None:
			gabor_filters = gabor_filters[indices]
			gabor_angles = gabor_angles[indices]

		for gf, theta in zip(gabor_filters, gabor_angles.tolist()):
			gf = gf.squeeze().cpu().numpy()
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))


	def plot_gabor_response(self, res):

		for _res, _angle in zip(res.squeeze(0), self.angles_v):
			_res = _res.cpu().detach().numpy().squeeze()
			plt.figure()
			plt.imshow(_res)
			plt.title("angle={}".format(_angle))


	def trainable_parameters(self,):

		if self.line_clf is not None:
			params = list(self.prelu.parameters())
			params += list(self.line_clf.parameters())
			return params
		else:
			return self.parameters()
		
	def vis_sampling(self, sampled_scores):
		plt.figure()
		plt.imshow(sampled_scores.squeeze().cpu().detach().numpy())

	def lines_detect(self, x_gabor, lines_endpoints, plot=False):

		sz = x_gabor.shape[-2:]
		lines_features = []
		lines_features_flip = []
		for endpoints in lines_endpoints:

			endpoints = endpoints.numpy()
			sampled_points = lines.sample_grid(endpoints, sz, self.resol, w=self.width)
			sampled_points = torch.Tensor(sampled_points).to(self.get_device()).unsqueeze(0)

			sampled_scores = F.grid_sample(x_gabor, sampled_points)
			sampled_scores = self.prelu(sampled_scores)
			sampled_scores = self.avg_pooling(sampled_scores)
			sampled_scores = sampled_scores / torch.abs(sampled_scores).squeeze().max()
			sampled_scores_flip = torch.flip(sampled_scores, (3,))
			if plot:
				self.vis_sampling(sampled_scores)
				self.vis_sampling(sampled_scores_flip)
				plt.show()
			lines_features.append(sampled_scores)
			lines_features_flip.append(sampled_scores_flip)
		
		lines_features = torch.cat(lines_features, 0)
		lines_features_flip = torch.cat(lines_features_flip, 0)

		lines_scores = self.line_clf(lines_features).squeeze()
		lines_scores_flip = self.line_clf(lines_features_flip).squeeze()

		return torch.max(lines_scores, lines_scores_flip)

	def lines_detect_test(self, x_gabor, proposed_lines, lines_endpoints):

		lines_scores = self.lines_detect(x_gabor, lines_endpoints, plot=False)
		lines_preds = torch.sigmoid(lines_scores).squeeze().cpu().numpy() > self.test_tresh
		_lines = proposed_lines.squeeze(0).cpu().numpy()[lines_preds]

		ms = MeanShift(bandwidth=50.0)
		pred = ms.fit_predict(_lines[:,0][...,np.newaxis])
		cluster_labels = np.unique(pred)

		cluster_lines = []
		for _label in cluster_labels.tolist():
			close_lines = _lines[pred == _label]
			cluster_lines.append(close_lines.mean(0).tolist())

		return cluster_lines


	def forward(self, inputs):

		def _gabor_bank(x):
			x = self.gabor_bank(x)
			x = x.transpose(0,1)
			x = x[:-1] + x[1:]
			n_intervals = x.shape[0] // 2
			x_v = x[:n_intervals]
			x_h = x[n_intervals:]
			x_vh = torch.max(x_v, x_h).transpose(0,1)
			hist = self.relu(x_vh).view(1, n_intervals,-1).mean(2)
			return (x_v, x_h), hist

		input_shape = inputs["image"].shape[-2:]
		result, features = super(AngleNet).forward(inputs, return_intermediate=True)
		x_features = features["decoder"].transpose(0,1)

		x_0, x_1 = x_features[self.pos:self.pos+2]
		(x_gabor_v, x_gabor_h), hist = _gabor_bank(x_1 - x_0)

		x_0_ = torch.max(x_0, x_1)
		x_features = torch.cat([x_0_.unsqueeze(1), x_features[self.pos+2:]], dim=0).transpose(0,1)

		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features_up)

		if self.line_clf is not None:

			idx = inputs["angle_range_label"]

			x_gabor_v = x_gabor_v[idx].unsqueeze(0)
			x_gabor_v = F.interpolate(x_gabor_v, size=input_shape, mode='bilinear', align_corners=False)

			x_gabor_h = x_gabor_h[idx].unsqueeze(0)
			x_gabor_h = F.interpolate(x_gabor_h, size=input_shape, mode='bilinear', align_corners=False)
			
			lines_scores_v = self.lines_detect(x_gabor_v, inputs['lines_endpoints_v'].squeeze(0))
			lines_scores_h = self.lines_detect(x_gabor_h, inputs['lines_endpoints_h'].squeeze(0))
			lines_scores = torch.cat([lines_scores_v, lines_scores_h])

		if self.training:

			if self.line_clf is not None:
				result["lines_scores"] = lines_scores
			else:
				result["seg"] = x_seg
				result["hist"] = hist
			
			return result

		""" else:

			(x_gabor_v, x_gabor_h) , hist = _gabor_bank(x_features_up)
			idx = inputs['angle_range_label'].item()
			x_gabor_v = x_gabor_v[idx].unsqueeze(0)
			x_gabor_h = x_gabor_h[idx].unsqueeze(0)

			lines_v = self.lines_detect_test(x_gabor_v, inputs['proposed_lines_v'], inputs['lines_endpoints_v'].squeeze(0))
			lines_h = self.lines_detect_test(x_gabor_h, inputs['proposed_lines_h'], inputs['lines_endpoints_h'].squeeze(0))
			lines_mask = lines.create_grid(tuple(x_gabor_v.shape[-2:]), lines_v + lines_h)

			result["lines_mask"] = lines_mask

		return result """






