import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import deeplabv3.lines as lines
from deeplabv3.model.deeplab import _Deeplabv3Plus, DeepLabDecoder1, init_conv
from deeplabv3.utils import check_gradient
from sklearn.cluster import MeanShift

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




class AngleDetect(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict,
		aux=False, out_planes_skip=48, angle_step=15.0, min_angle=-30.0, max_angle=30.0, kernel_size=51, train_gabor=False):
		
		super(AngleDetect, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)
		self.reduce = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False))

		self.relu = nn.ReLU()


		angles1 = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)

		self.train_gabor = train_gabor
		if train_gabor:
			self.gabor_bank = GaborBank(angles)
		else:
			self.gabor_bank = nn.Conv2d(1, 2 * self.num_angles, kernel_size=kernel_size, stride=1, bias=False)
			filter_weights = []
			for angle in (np.pi/2 - angles).tolist():
				filter_weights.append(gabor(angle, kernel_size=kernel_size))
			filter_weights = np.dstack(filter_weights)
			filter_weights = np.transpose(filter_weights, (2,0,1))
			filter_weights = torch.Tensor(filter_weights).view_as(self.gabor_bank.weight.data)
			self.gabor_bank.weight = nn.Parameter(filter_weights, requires_grad=True)

		self.line_sampler = lines.LineSampler(angle_step=2.5, rho_step=25)


	def plot_gabor(self, indices=None):

		gabor_filters = self.gabor_bank.weight.data[:self.num_angles]
		gabor_angles = self.angles_v.copy()
		if indices is not None:
			gabor_filters = gabor_filters[indices]
			gabor_angles = gabor_angles[indices]

		for gf, theta in zip(gabor_filters, gabor_angles):
			gf = gf.squeeze().cpu().numpy()
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))
		plt.show()


	def plot_gabor_response(self, res):

		for _res in res.squeeze(0)[:self.num_angles]:
			_res = _res.cpu().detach().numpy().squeeze()
			plt.figure()
			plt.imshow(_res)
		plt.show()

		
	def load_state_dict(self, state_dict, strict=True):
		super(AngleDetect, self).load_state_dict(state_dict, strict)
		if 'reduce.weight' not in state_dict:
			w0, w1 = self.classifier.weight.data[:2]
			self.reduce[0].weight = nn.Parameter((w1 - w0).unsqueeze(0), requires_grad=True)


	def compute_angle_range(self, x_features):

		@check_gradient('GABOR_OUPUT')
		def apply_gabor_bank(x):
			x = self.gabor_bank(x)
			x = self.relu(x)
			x = x.transpose(0,1)
			x1 = x[:self.num_angles].transpose(0,1)
			x2 = x[self.num_angles:].transpose(0,1)
			return x1 + x2

		@check_gradient('HIST_OUPUT')
		def compute_hist(x):
			bs = x.shape[0]
			x = x.view(bs, self.num_angles, -1).sum(2)
			x = x.transpose(0,1)
			return (x[:-1] + x[1:]).transpose(0,1)

		x = apply_gabor_bank(x_features)
		return compute_hist(x)

	def trainable_parameters(self,):
		params = list(self.reduce.parameters())
		if self.train_gabor:
			params += list(self.gabor_bank.parameters())
		return params

	def lines_detect(self, scores, angle_ranges_probs):

		sz = scores.shape[-2:]
		idx = torch.argmax(angle_ranges_probs.squeeze()).item()

		angle_range_v = np.deg2rad((self.angles_v[idx], self.angles_v[idx+1]))
		angle_range_h = angle_range_v + np.pi/2

		lines_coeffs_v, line_endpoints_v = self.line_sampler(angle_range_v, sz)
		sampled_points_v = lines.sample_line(line_endpoints_v, sz)

		lines_coeffs_h, line_endpoints_h = self.line_sampler(angle_range_h, sz)
		sampled_points_h = lines.sample_line(line_endpoints_h, sz)

		proposed_lines = np.array(lines_coeffs_v + lines_coeffs_h, dtype=np.float32)
		sampled_points = np.vstack((sampled_points_v, sampled_points_h))[np.newaxis,...]
		grid = torch.Tensor(sampled_points).to(self.get_device())
		sampled_scores = F.grid_sample(scores, grid)
		line_probs = torch.sigmoid(sampled_scores.transpose(1,2)).mean(3).squeeze()
		pre, post = self.predict(line_probs, proposed_lines, sz)

		return post


	def forward(self, inputs):

		@check_gradient('REDUCE_OUPUT', tensor_stats=True)
		def _reduce(x):
			return self.reduce(x)

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x_score = _reduce(x)
		angle_ranges = self.compute_angle_range(x_score)

		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["angle_ranges"] = angle_ranges
			return result
		else:
			# x_score = F.interpolate(x_score, size=input_shape, mode='bilinear', align_corners=False)
			# return angle_ranges, self.lines_detect(x_score, angle_ranges)
			return angle_ranges, 0


class AngleDetect_v2(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict,
		aux=False, out_planes_skip=48, angle_step=15.0, min_angle=-30.0, max_angle=30.0, kernel_size=51, group_size=5, n_groups=5, resol=10):
		
		super(AngleDetect_v2, self).__init__(n_classes, 
											pretrained_model, 
											DeepLabDecoder1(256, out_planes=out_planes_skip), 
											predict,
											aux=aux)

		self.reduce = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.reduce)

		angles1 = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
		self.angles_v = np.rad2deg(angles1)
		angles2 = angles1 + np.pi/2
		angles = np.array(angles1.tolist() + angles2.tolist())
		self.num_angles = len(angles1)
		gabor_bank = nn.Conv2d(1, 2 * self.num_angles, kernel_size=kernel_size, stride=1, bias=False)
		filter_weights = []
		for angle in (np.pi/2 - angles).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size))
		filter_weights = np.dstack(filter_weights)
		filter_weights = np.transpose(filter_weights, (2,0,1))
		filter_weights = torch.Tensor(filter_weights).view_as(gabor_bank.weight.data)
		gabor_bank.weight = nn.Parameter(filter_weights, requires_grad=True)
		self.gabor_bank = nn.Sequential(gabor_bank, nn.ReLU())

		# self.group_size = group_size
		self.n_groups = n_groups
		self.avg_pooling = nn.AdaptiveAvgPool1d(self.n_groups)
		self.line_clf = nn.Conv2d(256, 1, kernel_size=(self.n_groups, 1), stride=1, bias=False)
		init_conv(self.line_clf)
		self.resol = resol


	def plot_gabor(self, indices=None):

		gabor_filters = self.gabor_bank.weight.data[:self.num_angles]
		gabor_angles = self.angles_v.copy()
		if indices is not None:
			gabor_filters = gabor_filters[indices]
			gabor_angles = gabor_angles[indices]

		for gf, theta in zip(gabor_filters, gabor_angles):
			gf = gf.squeeze().cpu().numpy()
			plt.figure()
			plt.imshow(gf)
			plt.title("Theta={}".format(theta))
		plt.show()


	def plot_gabor_response(self, res):

		for _res in res.squeeze(0)[:self.num_angles]:
			_res = _res.cpu().detach().numpy().squeeze()
			plt.figure()
			plt.imshow(_res)
		plt.show()


	def angle_detection(self, x_features):

		# @check_gradient('GABOR_OUPUT')
		def apply_gabor_bank(x):
			x = self.reduce(x)
			x = self.gabor_bank(x)
			x = x.transpose(0,1)
			x1 = x[:self.num_angles].transpose(0,1)
			x2 = x[self.num_angles:].transpose(0,1)
			return x1 + x2

		# @check_gradient('HIST_OUPUT')
		def compute_hist(x):
			bs = x.shape[0]
			x = x.view(bs, self.num_angles, -1).sum(2)
			x = x.transpose(0,1)
			return (x[:-1] + x[1:]).transpose(0,1)

		x_gabor = apply_gabor_bank(x_features)
		x_hist = compute_hist(x_gabor)

		return x_gabor, x_hist

	def trainable_parameters(self,):
		params = list(super(AngleDetect_v2, self).parameters())
		params += list(self.reduce.parameters())
		params += list(self.line_clf.parameters())
		return params

	def lines_detect(self, x_features, lines_endpoints):

		# def avg_pooling(sampled_points):
		# 	n_points = sampled_points.shape[0]
		# 	group_size = n_points // self.n_groups
		# 	points_groups = np.split(sampled_points, np.arange(group_size, n_points, group_size))
		# 	points_groups = np.dstack(points_groups).transpose((2,0,1))
		# 	points_groups = torch.Tensor(points_groups).to(self.get_device()).unsqueeze(0)
		# 	sampled_features = F.grid_sample(x_features, points_groups)


		# def _group(sampled_points):
		# 	n_points = sampled_points.shape[0]
		# 	sampled_points_groups = np.split(sampled_points, np.arange(self.group_size, n_points, self.group_size))
		# 	sampled_points_groups = sampled_points_groups[:-1]
		# 	sampled_points_groups += [sampled_points[-self.group_size:]]
		# 	groups = np.dstack(sampled_points_groups)
		# 	centers = groups[self.group_size // 2, ...]
		# 	groups = groups.transpose((2,0,1))
		# 	centers = centers.transpose((1,0))
		# 	return groups, centers

		sz = x_features.shape[-2:]
		line_features = []
		for endpoints in lines_endpoints:
			endpoints = endpoints.numpy()
			sampled_points = lines.sample_line(endpoints, sz, self.resol, m=self.n_groups)
			sampled_points = torch.Tensor(sampled_points).to(self.get_device()).unsqueeze(0).unsqueeze(2)
			sampled_features = F.grid_sample(x_features, sampled_points)
			line_features.append(self.avg_pooling(sampled_features.squeeze(3)))
		line_features = torch.cat(line_features, 0).unsqueeze(3)
		lines_scores = self.line_clf(line_features).squeeze()

		return lines_scores


	def forward(self, inputs):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_features = self.decoder(x, x_low)
		x_gabor, x_angles = self.angle_detection(x_features)
		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features_up)

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["angle"] = x_angles
			result["out"]["seg"] = x_seg
			# if 'proposed_lines_endpoints' in inputs:
			# 	lines_endpoints = inputs['proposed_lines_endpoints'].squeeze(0)
			# 	lines_scores = self.lines_detect(x_features_up, lines_endpoints)
			# 	pdb.set_trace()
			# 	result["out"]["lines_scores"] = lines_scores
		else:
			_, x_seg = torch.max(x_seg, 1)
			result["seg"] = x_seg
			result["gabor"] = x_gabor
			result["hist"] = x_angles

		return result


class AngleNet(_Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, predict,
		aux=False, out_planes_skip=48, angle_step=15.0, min_angle=-45.0, max_angle=45.0, kernel_size=101):
		# Para debuggear
		n_classes = 4
		super(AngleNet, self).__init__(n_classes, 
									   pretrained_model, 
									   DeepLabDecoder1(256, out_planes=out_planes_skip), 
									   predict,
									   aux=aux)

		self.reduce = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.reduce)

		angles =  np.deg2rad(np.arange(min_angle, max_angle + 90.0 + angle_step, angle_step))
		self.angles_v = np.arange(min_angle, max_angle + angle_step, angle_step)
		self.num_angles = len(self.angles_v)
		# angles1 = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
		# self.angles_v = np.rad2deg(angles1)
		# angles2 = angles1 + np.pi/2
		# angles = np.array(angles1.tolist() + angles2.tolist())
		# self.num_angles = len(angles1)
		# self.gabor_bank = nn.Conv2d(1, 2 * self.num_angles, kernel_size=kernel_size, stride=1, bias=False)
		self.gabor_bank = nn.Conv2d(1, len(angles), kernel_size=kernel_size, stride=1, bias=False)
		filter_weights = []
		for angle in (angles + np.pi/2).tolist():
			filter_weights.append(gabor(angle, kernel_size=kernel_size))
		filter_weights = np.dstack(filter_weights)
		filter_weights = np.transpose(filter_weights, (2,0,1))
		filter_weights = torch.Tensor(filter_weights).view_as(self.gabor_bank.weight.data)
		self.gabor_bank.weight = nn.Parameter(filter_weights, requires_grad=True)

		self.resol = 10.0
		self.width = 7
		self.group_size = 5
		self.nfilters = 100
		self.test_tresh = 0.75

		self.relu = nn.ReLU()
		self.prelu = nn.PReLU(init=0.75)
		self.avg_pooling = nn.AdaptiveAvgPool2d(self.width)
		self.line_clf = nn.Sequential(
			nn.Conv2d(1, self.nfilters, kernel_size=7, stride=1, bias=False), 
			nn.ReLU(),
			nn.Conv2d(self.nfilters, 1, kernel_size=1, stride=1, bias=False))
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
		# params = list(super(AngleNet, self).parameters())
		# pdb.set_trace()
		# params += list(self.reduce.parameters())
		# params += list(self.line_clf.parameters())
		# params += list(self.prelu.parameters())
		return self.parameters()

	#Para debuggear
	def load_state_dict(self, state_dict, strict=True):
		super(AngleNet, self).load_state_dict(state_dict, strict)
		if 'reduce.weight' not in state_dict:
			print('REDUCE')
			w0, w1 = self.classifier.weight.data[:2]
			self.reduce.weight = nn.Parameter((w1 - w0).unsqueeze(0), requires_grad=True)

	def vis_sampling(self, sampled_scores):
		plt.figure()
		plt.imshow(sampled_scores.squeeze().cpu().detach().numpy())

	def lines_detect(self, x_gabor, lines_endpoints, plot=False):

		# def group_points(sampled_points):
		# 	n_points = sampled_points.shape[0]
		# 	points_groups = np.split(sampled_points, np.arange(self.group_size, n_points, self.group_size))
		# 	return np.dstack(points_groups).transpose((2,0,1))

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


		# def _interval(x):
		# 	return (x[:-1] + x[1:])

		
		# def _gabor_bank(x_features):
		# 	x = self.reduce(x_features)
		# 	x = self.gabor_bank(x)
		# 	x = x.transpose(0,1)
		# 	xv = _interval(x[:self.num_angles])
		# 	xh = _interval(x[self.num_angles:])
		# 	x = torch.max(xv, xh).transpose(0,1)
		# 	hist = self.relu(x).view(1, self.num_angles -1,-1).mean(2)
		# 	return (xv.squeeze(1), xh.squeeze(1)), hist

		def _gabor_bank(x_features):
			x = self.reduce(x_features)
			x = self.gabor_bank(x)
			x = x.transpose(0,1)
			x = x[:-1] + x[1:]
			n_intervals = x.shape[0] // 2
			x_v = x[:n_intervals]
			x_h = x[n_intervals:]
			x_vh = torch.max(x_v, x_h).transpose(0,1)
			hist = self.relu(x_vh).view(1, n_intervals,-1).mean(2)
			return (x_v, x_h), hist


		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["out"]
		x_low = features["skip1"]
		x = self.aspp(x)
		x_features = self.decoder(x, x_low)
		x_features_up = F.interpolate(x_features, size=input_shape, mode='bilinear', align_corners=False)
		x_seg = self.classifier(x_features_up)

		result = OrderedDict()

		if self.training:

			result["out"] = OrderedDict()
			result["out"]["seg"] = x_seg

			idx = inputs['angle_range_label'].item()
			if idx != 255:

				(x_gabor_v, x_gabor_h), hist = _gabor_bank(x_features_up)

				x_gabor_v = x_gabor_v[idx].unsqueeze(0)
				x_gabor_h = x_gabor_h[idx].unsqueeze(0)

				lines_scores_v = self.lines_detect(x_gabor_v, inputs['lines_endpoints_v'].squeeze(0))
				lines_scores_h = self.lines_detect(x_gabor_h, inputs['lines_endpoints_h'].squeeze(0))
				lines_scores = torch.cat([lines_scores_v, lines_scores_h])

				result["out"]["hist"] = hist
				result["out"]["lines_scores"] = lines_scores
		else:

			(x_gabor_v, x_gabor_h) , hist = _gabor_bank(x_features_up)
			idx = inputs['angle_range_label'].item()
			x_gabor_v = x_gabor_v[idx].unsqueeze(0)
			x_gabor_h = x_gabor_h[idx].unsqueeze(0)

			lines_v = self.lines_detect_test(x_gabor_v, inputs['proposed_lines_v'], inputs['lines_endpoints_v'].squeeze(0))
			lines_h = self.lines_detect_test(x_gabor_h, inputs['proposed_lines_h'], inputs['lines_endpoints_h'].squeeze(0))
			lines_mask = lines.create_grid(tuple(x_gabor_v.shape[-2:]), lines_v + lines_h)

			result["lines_mask"] = lines_mask

		return result






