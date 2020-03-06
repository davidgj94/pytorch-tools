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
from torchtools.utils import save_heatmaps
import os
import os.path
import shutil
import cv2


def compute_seg(x, output_shape, classifier):
	x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
	return classifier(x)


def compute_grids(angle_range, angle_step, kernel_size, grid_size):

	center = (kernel_size - 1) // 2
	k = center - (grid_size - 1) // 2
	def _compute_grid(angle):
		theta = torch.zeros(1, 2, 3)
		theta[:, :, :2] = torch.tensor([[np.cos(angle), -1.0*np.sin(angle)],
									[np.sin(angle), np.cos(angle)]])
		grid = F.affine_grid(theta, (1,1) + _pair(kernel_size))
		grid = grid.squeeze(0)[k:-k, k:-k].unsqueeze(0)
		return grid

	min_angle, max_angle = angle_range
	angles = np.deg2rad(np.arange(min_angle, max_angle + angle_step, angle_step))
	grids = []
	for angle in (-1.0*angles).tolist():
		grid = _compute_grid(angle)
		grids.append(grid)

	return torch.cat(grids, dim=0)


def padding(dilation, kernel_size):
	return dilation * (kernel_size - 1) // 2


def create_convnet(conv_layer, planes, kernel_size, padding, dilation, groups, remove_last_relu=False, **kwargs):

	# def _init_conv(m):
	# 	if isinstance(m, OrientedConv2d):
	# 		m.reset_parameters()
	# 	else:
	# 		init_conv(m)

	n_layers = len(planes) - 1
	layers = []
	for idx in np.arange(n_layers):
		in_planes = planes[idx]
		out_planes = planes[idx+1]
		_conv = conv_layer(in_planes, out_planes, 
								kernel_size=kernel_size, 
								padding=padding, 
								dilation=dilation,
								bias=False, **kwargs)
		init_conv(_conv)
		norm = nn.GroupNorm(groups, out_planes)
		relu = nn.ReLU()
		layers += [_conv, norm, relu]
	
	if remove_last_relu:
		layers = layers[:-1]

	return nn.Sequential(*layers)


def forward_ori(ori_net, x, idx):
	for module in ori_net._modules.values():
		if isinstance(module, OrientedConv2d):
			module.set_grid(idx)
	return ori_net(x)


class Deeplabv3_Roads(Deeplabv3):
	def __init__(self, n_classes, pretrained_model, freeze_backbone=False, freeze_aspp=False):
		super(Deeplabv3_Roads, self).__init__(n_classes, pretrained_model, aux=False)
		self.freeze_parameters(freeze_backbone, freeze_aspp)
	
	def freeze_parameters(self, freeze_backbone, freeze_aspp):
		for name, param in self.named_parameters():
			if ('backbone' in name) and freeze_backbone:
				param.requires_grad = False
			elif ('aspp' in name) and freeze_aspp:
				param.requires_grad = False

@register.attach('roads_net')
class RoadsNet(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=15.0,
					layer1_channels=48, decoder_grid_size=7, decoder_planes=[128, 64, 32],
					fuse_kernel_size=5, norm_groups=8, freeze_backbone=False, freeze_aspp=False, train_ori=False):

		super(RoadsNet, self).__init__(n_classes, pretrained_model, 
													freeze_backbone=freeze_backbone,
													freeze_aspp=freeze_aspp)

		self.rot_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
		self.n_angles = len(self.rot_angles)

		# Decoder

		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, layer1_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer1_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		decoder_planes = [256+layer1_channels] + decoder_planes
		decoder_kernel_size = decoder_grid_size + 8
		self.ori_net = create_convnet(OrientedConv2d, 
										decoder_planes, 
										decoder_kernel_size, 
										padding(1, decoder_grid_size), 
										dilation=1, 
										groups=norm_groups,
										min_angle=min_angle,
										max_angle=max_angle,
										angle_step=angle_step)

		self.aux_clf_ori = None
		if train_ori:
			self.aux_clf_ori = nn.Conv2d(decoder_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_clf_ori)

		out_planes_decoder = decoder_planes[-1] * (self.n_angles - 1)
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(out_planes_decoder, 256, kernel_size=fuse_kernel_size, stride=1, bias=False),
			nn.GroupNorm(norm_groups, 256),
			nn.ReLU(),
			nn.Dropout(0.1))
		self.fuse_conv.apply(init_conv)


	def eval(self,):
		super(RoadsNet, self).eval()
		for module in self.ori_net._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()


	# def trainable_parameters(self, base_lr, alfa=10,):

	# 	params_groups_2 = list(super(RoadsNetMultitask, self).trainable_parameters())

	# 	decoder_1_params = []
	# 	decoder_1_params += list(self.reduce_layer1.parameters())
	# 	decoder_1_params += list(self.ori_net.parameters())
	# 	decoder_1_params += list(self.decoder_1_clf.parameters())

	# 	decoder_2_params = []
	# 	decoder_2_params += list(self.reduce_layer2.parameters())
	# 	decoder_2_params += list(self.decoder_2_net.parameters())
	# 	decoder_2_params += list(self.decoder_2_clf.parameters())

	# 	total_params_groups = [{'params': iter(decoder_1_params+decoder_2_params), 'lr': base_lr}]
	# 	if len(params_groups_2) > 0:
	# 		total_params_groups += [{'params': iter(params_groups_2), 'lr': base_lr / alfa}]
	# 	return total_params_groups


	def forward_decoder(self, x_aspp, x_layer1):

		layer1_size = x_layer1.shape[-2:]
		x_layer1 = self.reduce_layer1(x_layer1)
		x_aspp_up = F.interpolate(x_aspp, size=layer1_size, mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_layer1, x_aspp_up], dim=1)

		x_ori = []
		for idx in np.arange(self.n_angles-1):
			x_ori.append(forward_ori(self.ori_net, x_concat, idx))
		x_ori.append(forward_ori(self.ori_net, x_concat, 0))

		return x_ori


	def forward(self, inputs, return_intermediate=False):

		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNet, self).forward(inputs, return_intermediate=True)
		if return_intermediate:
			return features

		x_aspp = features["aspp"]
		x_decoder = self.forward_decoder(x_aspp, features["layer1"])

		if self.aux_clf_ori is not None:
			x_decoder_aux = torch.cat(x_decoder, dim=0)
			x_decoder_sum = x_decoder_aux[:-1] + x_decoder_aux[1:]
			seg_aux = compute_seg(x_decoder_sum, input_shape, self.aux_clf_ori).squeeze(1).unsqueeze(0)

		x_decoder_concat = torch.cat(x_decoder[:-1], dim=1)
		x_fuse = self.fuse_conv(x_decoder_concat)
		seg = compute_seg(x_fuse, input_shape, self.classifier).squeeze(1)

		result = OrderedDict()
		result["binary_seg"] = OrderedDict()
		if self.training:
			result["binary_seg"]["seg"] = seg
			if self.aux_clf_ori is not None:
				result["ori_seg"] = OrderedDict()
				result["ori_seg"]["seg"] = seg_aux
		else:
			result["binary_seg"]["seg"] = (seg.squeeze(1) > 0).cpu()

		return result

@register.attach('roads_net_v2')
class RoadsNet_v2(RoadsNet):
	def __init__(self, n_classes, pretrained_model,
					angle_step=15.0, decoder_planes=[128, 64, 32], fuse_kernel_size=5, out_fuse_channels=128, 
					freeze_backbone=False, freeze_aspp=False, train_ori=False, norm_groups=8, layer1_channels=48):

		super(RoadsNet_v2, self).__init__(n_classes, pretrained_model, angle_step=angle_step, 
											decoder_planes=decoder_planes, fuse_kernel_size=fuse_kernel_size, 
											freeze_backbone=freeze_backbone, freeze_aspp=freeze_aspp, train_ori=train_ori, 
											norm_groups=norm_groups, layer1_channels=layer1_channels)

		fine_net_planes = [layer1_channels + 256, 256, 256]
		fine_net_kernel_size = 3
		self.fine_net = create_convnet(nn.Conv2d, 
										fine_net_planes, 
										fine_net_kernel_size, 
										padding(1, fine_net_kernel_size), 
										dilation=1, 
										groups=norm_groups)
		
		out_planes_decoder = decoder_planes[-1] * (self.n_angles - 1)
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(out_planes_decoder, out_fuse_channels, 
						kernel_size=fuse_kernel_size, 
						stride=1, 
						padding=padding(1, fuse_kernel_size), 
						bias=False),
			nn.GroupNorm(norm_groups, out_fuse_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fuse_conv.apply(init_conv)

		self.fuse_branches =  nn.Sequential(
			nn.Conv2d(out_fuse_channels + 256, 256, 
						kernel_size=fuse_kernel_size, 
						stride=1, 
						padding=padding(1, fuse_kernel_size), 
						bias=False),
			nn.GroupNorm(norm_groups, 256),
			nn.ReLU(),
			nn.Dropout(0.1))
		self.fuse_branches.apply(init_conv)


	def forward_decoders(self, x_aspp, x_layer1):

		layer1_size = x_layer1.shape[-2:]
		x_layer1 = self.reduce_layer1(x_layer1)
		x_aspp_up = F.interpolate(x_aspp, size=layer1_size, mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_layer1, x_aspp_up], dim=1)

		x_fine = self.fine_net(x_concat)

		x_ori = []
		for idx in np.arange(self.n_angles-1):
			x_ori.append(forward_ori(self.ori_net, x_concat, idx))
		x_ori.append(forward_ori(self.ori_net, x_concat, 0))

		return x_ori, x_fine
	
	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNet_v2, self).forward(inputs, return_intermediate=True)
		x_aspp = features["aspp"]
		x_ori, x_fine = self.forward_decoders(x_aspp, features["layer1"])

		if self.aux_clf_ori is not None:
			x_ori_aux = torch.cat(x_ori, dim=0)
			x_ori_sum = x_ori_aux[:-1] + x_ori_aux[1:]
			aux_label_size = inputs["ori_seg"]["label"].shape[-2:]
			seg_aux = compute_seg(x_ori_sum, aux_label_size, self.aux_clf_ori).squeeze(1).unsqueeze(0)

		x_ori = self.fuse_conv(torch.cat(x_ori[:-1], dim=1))
		x_concat_branches = torch.cat([x_ori, x_fine], dim=1)
		x_fuse_branches = self.fuse_branches(x_concat_branches)
		seg = compute_seg(x_fuse_branches, input_shape, self.classifier).squeeze(1)

		result = OrderedDict()
		result["binary_seg"] = OrderedDict()
		if self.training:
			result["binary_seg"]["seg"] = seg
			if self.aux_clf_ori is not None:
				result["ori_seg"] = OrderedDict()
				result["ori_seg"]["seg"] = seg_aux
		else:
			result["binary_seg"]["seg"] = (seg.squeeze(1) > 0).cpu()

		return result

@register.attach('roads_net_v3')
class RoadsNet_v3(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=20.0,
					layer1_channels=48, layer2_channels=64, ori_decoder_planes=[128, 64, 64], ori_grid_size=7, ori_fuse_ks=3, 
					norm_groups=8, freeze_backbone=False, freeze_aspp=False):

		super(RoadsNet_v3, self).__init__(n_classes, pretrained_model, 
													freeze_backbone=freeze_backbone,
													freeze_aspp=freeze_aspp)
		
		self.n_angles = len(np.arange(0.0, 180.0, angle_step))
		
		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, layer1_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer1_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		self.reduce_layer2 = nn.Sequential(
			nn.Conv2d(512, layer2_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer2_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer2.apply(init_conv)

		self.fuse_layer1 = create_convnet(nn.Conv2d, [256 + layer1_channels, 256, 256], 3, 
																	padding=padding(1, 3), 
																	dilation=1, 
																	groups=norm_groups)
		
		self.fuse_layer2 =  nn.Sequential(
			nn.Conv2d(256 + layer2_channels, ori_decoder_planes[0], kernel_size=3, stride=1, bias=False, padding=padding(1,3)),
			nn.GroupNorm(norm_groups, ori_decoder_planes[0]),
			nn.ReLU(),)
		self.fuse_layer2.apply(init_conv)
		
		ori_kernel_size = ori_grid_size + 8
		self.ori_conv = nn.Sequential(
			OrientedConv2d(ori_decoder_planes[0], ori_decoder_planes[1], ori_kernel_size, padding(1, ori_grid_size), dilation=1, angle_step=angle_step),
			nn.GroupNorm(norm_groups, ori_decoder_planes[1]),
			nn.ReLU(),)
		self.ori_conv.apply(init_conv)
		
		self.ori_fuse = nn.Sequential(
			nn.Conv2d(ori_decoder_planes[1] * self.n_angles, ori_decoder_planes[2], 
												kernel_size=ori_fuse_ks, 
												stride=1, 
												bias=False,
												padding=padding(1, ori_fuse_ks)),
			nn.GroupNorm(norm_groups, ori_decoder_planes[2]),
			nn.ReLU(),)
		self.ori_fuse.apply(init_conv)
		
		self.fuse_decoders =  nn.Sequential(
			nn.Conv2d(256 + ori_decoder_planes[2], 256, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, 256),
			nn.ReLU(),)
		self.fuse_decoders.apply(init_conv)

		self.aux_clf_ori = nn.Conv2d(ori_decoder_planes[1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.aux_clf_ori)

	def forward_decoder1(self, x_aspp, x_layer1):
		x_layer1 = self.reduce_layer1(x_layer1)
		x_aspp_up = F.interpolate(x_aspp, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_layer1, x_aspp_up], dim=1)
		x_decoder1 = self.fuse_layer1(x_concat)
		return x_decoder1


	def forward_decoder2(self, x_aspp, x_layer2):

		x_layer2 = self.reduce_layer2(x_layer2)
		x_concat = torch.cat([x_layer2, x_aspp], dim=1)
		x_fuse = self.fuse_layer2(x_concat)

		x_ori = []
		for idx in np.arange(self.n_angles):
			x_ori.append(forward_ori(self.ori_conv, x_fuse, idx))

		x_decoder2 = self.ori_fuse(torch.cat(x_ori, dim=1))
		return torch.cat(x_ori, dim=0), x_decoder2


	def trainable_parameters(self, base_lr, alfa, debug=True):

		pretrained_params = []
		pretrained_params_name = []

		new_params = []
		new_params_name = []

		clf_params = []
		clf_params_name = []

		for name, params in self.named_parameters():
			if ("aspp" in name) or ("backbone" in name):
				pretrained_params.append(params)
				pretrained_params_name.append(name)
			elif ("classifier" in name) or ("aux_clf_ori" in name):
				clf_params.append(params)
				clf_params_name.append(name)
			else:
				new_params.append(params)
				new_params_name.append(name)

		params_groups = [{'params': iter(clf_params), 'lr': base_lr}, 
						 {'params': iter(new_params), 'lr': base_lr / alfa[0]},
						 {'params': iter(pretrained_params), 'lr': base_lr / alfa[1]}]
		
		if debug:
			for params_names, group_name, _param_group in zip([clf_params_name, new_params_name, pretrained_params_name], 
												['cls_params',    'new_params',    'pretrained_params'],
												params_groups):
				lr = _param_group['lr']
				print(">>>>>>>>> {}:".format(group_name))
				print(">>>>>>>>> Learning rate: {}".format(lr))
				for name in params_names:
					print("-- {}".format(name))
				print()

		return params_groups

	def eval(self,):
		super(RoadsNet_v3, self).eval()
		for module in self.ori_conv._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()


	def forward(self, inputs):

		binary_label_shape = inputs["binary_seg"]["label"].shape[-2:]

		features = super(RoadsNet_v3, self).forward(inputs, return_intermediate=True)
		x_aspp = features["aspp"]

		x_decoder1 = self.forward_decoder1(x_aspp, features["layer1"])
		x_ori, x_decoder2 = self.forward_decoder2(x_aspp, features["layer2"])
		if self.training:
			ori_label_shape = inputs["ori_seg"]["label"].shape[-2:]
			seg_ori = compute_seg(x_ori, ori_label_shape, self.aux_clf_ori).squeeze(1).unsqueeze(0)

		x_decoder2_up = F.interpolate(x_decoder2, size=x_decoder1.shape[-2:], mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_decoder1, x_decoder2_up], dim=1)
		x_out = self.fuse_decoders(x_concat)
		seg = compute_seg(x_out, binary_label_shape, self.classifier).squeeze(1)

		result = OrderedDict()
		if self.training:
			result['binary_seg'] = OrderedDict()
			result['ori_seg'] = OrderedDict()
			result['binary_seg']['seg'] = seg
			result['ori_seg']['seg'] = seg_ori
		else:
			result['binary_seg'] = OrderedDict()
			result['binary_seg']['seg'] = (seg.squeeze(1) > 0).cpu()
		
		return result

class RoadsDecoder(nn.Module):

	def __init__(self, layer1_channels, layer2_channels, ori_planes, ori_grid_size, angle_step, norm_groups, res_kernel_size):

		super(RoadsDecoder, self).__init__()

		self.n_angles = len(np.arange(0.0, 180.0 + angle_step, angle_step))

		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, layer1_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer1_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		self.reduce_layer2 = nn.Sequential(
			nn.Conv2d(512, layer2_channels, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(norm_groups, layer2_channels),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer2.apply(init_conv)

		# self.fuse_layer2 =  nn.Sequential(
		# 	nn.Conv2d(256 + layer2_channels, ori_planes[0], kernel_size=3, 
		# 													stride=1,
		# 													bias=False,
		# 													padding=padding(1, 3)),
		# 	nn.GroupNorm(norm_groups, ori_planes[0]),
		# 	nn.ReLU(),)
		# self.fuse_layer2.apply(init_conv)

		ori_kernel_size = ori_grid_size + 8
		ori_planes = [256+layer2_channels] + ori_planes
		self.ori_net = create_convnet(OrientedConv2d, ori_planes[:-1], ori_kernel_size, padding=padding(1, ori_grid_size), 
																						dilation=1,
																						groups=norm_groups, 
																						angle_step=angle_step)
		# self.ori_conv = nn.Sequential(
		# 	OrientedConv2d(ori_planes[0], ori_planes[1], kernel_size=ori_kernel_size, 
		# 												 padding=padding(1, ori_grid_size),
		# 												 dilation=1,
		# 												 angle_step=angle_step),
		# 	nn.GroupNorm(norm_groups, ori_planes[1]),
		# 	nn.ReLU(),)
		# self.ori_conv.apply(init_conv)

		self.ori_fuse = nn.Sequential(
			nn.Conv2d(ori_planes[-2] * (self.n_angles - 1), ori_planes[-1], kernel_size=3, 
																	  stride=1, 
																	  bias=False,
																	  padding=padding(1, 3)),
			nn.GroupNorm(norm_groups, ori_planes[-1]),
			nn.ReLU(),)
		self.ori_fuse.apply(init_conv)

		self.res_conv = nn.Conv2d(ori_planes[-1] + 256, 256, kernel_size=res_kernel_size, 
															 stride=1, 
															 bias=False,
															 padding=padding(1, res_kernel_size))
		init_conv(self.res_conv)
		self.norm_relu = nn.Sequential(nn.GroupNorm(norm_groups, 256), nn.ReLU())
		
		fuse_layer1_channels = [256 + layer1_channels, 256, 256]
		self.fuse_layer1 = create_convnet(nn.Conv2d, fuse_layer1_channels, kernel_size=3, 
																		   padding=padding(1, 3), 
																		   dilation=1, 
																		   groups=norm_groups,)


	def forward_layer2(self, x_aspp, x_layer2):
		x_layer2 = self.reduce_layer2(x_layer2)
		x_concat = torch.cat([x_layer2, x_aspp], dim=1)
		x_ori = []
		for idx in np.arange(self.n_angles - 1):
			x_ori.append(forward_ori(self.ori_net, x_concat, idx))
		x_ori.append(forward_ori(self.ori_net, x_concat, 0))
		return x_ori

	def res_block(self, x_aspp, x_ori_fused):
		x_concat = torch.cat([x_ori_fused, x_aspp], dim=1)
		res = self.res_conv(x_concat)
		return self.norm_relu(x_aspp + res)

	def forward_layer1(self, x_aspp, x_layer1):
		x_layer1 = self.reduce_layer1(x_layer1)
		x_aspp_up = F.interpolate(x_aspp, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
		x_concat = torch.cat([x_layer1, x_aspp_up], dim=1)
		return self.fuse_layer1(x_concat)

	def forward(self, x_aspp, x_layer1, x_layer2):
		x_ori = self.forward_layer2(x_aspp, x_layer2)
		x_ori_fused = self.ori_fuse(torch.cat(x_ori[:-1], dim=1))
		x_aspp = self.res_block(x_aspp, x_ori_fused)
		x_out = self.forward_layer1(x_aspp, x_layer1)
		return x_out, torch.cat(x_ori, dim=0)


@register.attach('roads_net_decoder')
class RoadsNet_Decoder(Deeplabv3_Roads):
	def __init__(self, n_classes, pretrained_model,
					min_angle=0.0, max_angle=180.0, angle_step=15.0,
					layer1_channels=48, layer2_channels=64, ori_planes=[128, 64, 128], ori_grid_size=7,
					norm_groups=8, res_kernel_size=3, freeze_backbone=False, freeze_aspp=False, sum_aux_ori=False, aux_hidden_kz=0):

		super(RoadsNet_Decoder, self).__init__(n_classes, pretrained_model, 
													freeze_backbone=freeze_backbone,
													freeze_aspp=freeze_aspp)

		self.decoder = RoadsDecoder(layer1_channels, layer2_channels, ori_planes, ori_grid_size, angle_step, norm_groups, res_kernel_size)
		self.sum_aux_ori = sum_aux_ori
		if aux_hidden_kz > 0:
			self.aux_clf_ori = nn.Sequential(
				nn.Conv2d(ori_planes[-2], ori_planes[-2], kernel_size=aux_hidden_kz, stride=1, bias=False, padding=padding(1, aux_hidden_kz)),
				nn.GroupNorm(norm_groups, ori_planes[-2]),
				nn.ReLU(),
				nn.Conv2d(ori_planes[-2], 1, kernel_size=1, stride=1, bias=False))
			self.aux_clf_ori.apply(init_conv)
		else:
			self.aux_clf_ori = nn.Conv2d(ori_planes[-2], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_clf_ori)
	
	def eval(self,):
		super(RoadsNet_Decoder, self).eval()
		for module in self.decoder.ori_conv._modules.values():
			if isinstance(module, OrientedConv2d):
				module.eval()
	
	def trainable_parameters(self, base_lr, alfa, debug=True):

		pretrained_params = []
		pretrained_params_name = []

		new_params = []
		new_params_name = []

		clf_params = []
		clf_params_name = []

		for name, params in self.named_parameters():
			if ("aspp" in name) or ("backbone" in name):
				pretrained_params.append(params)
				pretrained_params_name.append(name)
			elif ("classifier" in name) or ("aux_clf_ori" in name):
				clf_params.append(params)
				clf_params_name.append(name)
			else:
				new_params.append(params)
				new_params_name.append(name)

		params_groups = [{'params': iter(clf_params), 'lr': base_lr}, 
						 {'params': iter(new_params), 'lr': base_lr / alfa[0]},
						 {'params': iter(pretrained_params), 'lr': base_lr / alfa[1]}]
		
		if debug:
			for params_names, group_name, _param_group in zip([clf_params_name, new_params_name, pretrained_params_name], 
												['cls_params',    'new_params',    'pretrained_params'],
												params_groups):
				lr = _param_group['lr']
				print(">>>>>>>>> {}:".format(group_name))
				print(">>>>>>>>> Learning rate: {}".format(lr))
				for name in params_names:
					print("-- {}".format(name))
				print()

		return params_groups
	
	def aux_block(self, x_ori, output_shape):
		if self.sum_aux_ori:
			x_ori = x_ori[:-1] + x_ori[1:]
		else:
			x_ori = x_ori[:-1]
		return compute_seg(x_ori, output_shape, self.aux_clf_ori).squeeze(1).unsqueeze(0)

	def forward(self, inputs):
		input_shape = inputs["image"].shape[-2:]
		features = super(RoadsNet_Decoder, self).forward(inputs, return_intermediate=True)
		x_aspp = features["aspp"]
		x_layer1 = features["layer1"]
		x_layer2 = features["layer2"]
		x_out, x_ori = self.decoder(x_aspp, x_layer1, x_layer2)

		seg = compute_seg(x_out, input_shape, self.classifier).squeeze(1)
		if self.training:
			ori_label_shape = inputs["ori_seg"]["label"].shape[-2:]
			seg_ori = self.aux_block(x_ori, ori_label_shape)
		
		result = OrderedDict()
		if self.training:
			result['binary_seg'] = OrderedDict()
			result['ori_seg'] = OrderedDict()
			result['binary_seg']['seg'] = seg
			result['ori_seg']['seg'] = seg_ori
		else:
			result['binary_seg'] = OrderedDict()
			result['binary_seg']['seg'] = (seg.squeeze(1) > 0).cpu()
			# seg_aux = compute_seg(x_ori, input_shape, self.aux_clf_ori).squeeze(1)
			# plt.figure()
			# plt.imshow(inputs["binary_seg"]["label"].squeeze())
			# for _seg in seg_aux:
			# 	plt.figure()
			# 	plt.imshow(_seg.cpu())
			# plt.show()
			# result['seg'] = (seg.squeeze(1) > 0).cpu()
		
		return result



class OrientedConv2d(_ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size,
				 padding, dilation, stride=1, groups=1, bias=False, min_angle=0.0, max_angle=180.0, angle_step=30.0):

		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(OrientedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), groups, bias, 'zeros')
		
		kernel_size = kernel_size[0]
		grid_size = kernel_size - 8
		grids = compute_grids((min_angle, max_angle), angle_step, kernel_size, grid_size)
		self.register_buffer('grids', grids)
		self.idx = (len(self.grids) - 1) // 2 
		self.rotated_weights = []

	def set_grid(self, idx):
		self.idx = idx
	
	def eval(self,):
		super(OrientedConv2d, self).eval()
		for grid in self.grids:
			grid = grid.unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)
			self.rotated_weights.append(F.grid_sample(self.weight, grid))


	def rotate_weight(self,):
		if len(self.rotated_weights) > 0:
			return self.rotated_weights[self.idx]
		else:
			grid = self.grids[self.idx].unsqueeze(0).repeat(self.weight.shape[0], 1, 1, 1)
			return F.grid_sample(self.weight, grid)

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