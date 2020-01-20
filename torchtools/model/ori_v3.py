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
from .angle_net import predict

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
			nn.Dropout(0.1),
			nn.Conv2d(channels_input, 1, 1, bias=False),
			nn.GroupNorm(1, 1),
			nn.Sigmoid()
		)
	
	def forward(self, input_features, gating_features):
		alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
		return input_features * alphas

class Deeplabv3_ori(Deeplabv3):
	def __init__(self, n_classes, pretrained_model, freeze_params=False):
		super(Deeplabv3_ori, self).__init__(n_classes, pretrained_model, aux=False)
		self.aspp = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])
		if freeze_params:
			self.freeze_parameters()
	
	def freeze_parameters(self):
		for param in self.parameters():
			param.requires_grad = False

@register.attach("ori_net_v2")
class OrientedNet(Deeplabv3_ori):
	def __init__(self, n_classes, pretrained_model, aux=False,
					min_angle=-45.0, max_angle=45.0, angle_step=15.0, 
					grid_size=5, dilation=2, ori_planes=[64, 32, 16, 8],
					fuse_kernel_size=7, fuse_dilation=3, fuse_planes = [16, 16, 16], 
					freeze_params=False, attention_mod=False):

		super(OrientedNet, self).__init__(n_classes, pretrained_model, freeze_params=freeze_params)

		kernel_size = grid_size + (grid_size - 1) * 2
		
		channels_layer1 = 64
		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, channels_layer1, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(channels_layer1 // 8, channels_layer1),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		self.att_mod = None
		if attention_mod:

			channels_aspp = channels_layer1 // 4
			self.reduce_aspp = nn.Sequential(
			nn.Conv2d(256, channels_aspp, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(channels_aspp // 8, channels_aspp),
			nn.ReLU(),
			nn.Dropout(0.5))
			self.reduce_aspp.apply(init_conv)

			self.att_mod = AttentionModule(channels_layer1, channels_aspp)
			self.att_mod.apply(init_conv)

		self.grids_v, self.grids_h = self.compute_grids(min_angle, max_angle, angle_step, kernel_size, grid_size)

		self.ori_convs = self.create_ori_convs(ori_planes, kernel_size, grid_size, dilation)

		self.fuse_net = self.create_fuse_net(fuse_planes, fuse_kernel_size, fuse_dilation)
		self.fuse_net.apply(init_conv)

		self.line_clf_ori = nn.Conv2d(fuse_planes[-1], 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.line_clf_ori)

		self.aux_clf_ori = None
		if aux:
			self.aux_clf_ori = nn.Conv2d(ori_planes[-1], 1, kernel_size=1, stride=1, bias=False)
			init_conv(self.aux_clf_ori)

	
	def padding(self, dilation, kernel_size):
		return dilation * (kernel_size - 1) // 2
	
	
	def create_ori_convs(self, ori_planes, kernel_size, grid_size, dilation):

		n_layers = len(ori_planes) - 1
		layers = []
		for idx in np.arange(n_layers):
			in_planes = ori_planes[idx]
			out_planes = ori_planes[idx+1]
			ori_conv = OrientedConv2d(in_planes, out_planes, 
									kernel_size=kernel_size, 
									padding=self.padding(dilation, grid_size), 
									dilation=dilation)
			ori_conv.reset_parameters()
			layers.append(ori_conv)

		return nn.ModuleList(layers)
	
	
	def create_fuse_net(self, fuse_planes, grid_size, dilation):
		n_layers = len(fuse_planes) - 1
		layers = []
		for idx in np.arange(n_layers):
			in_planes = fuse_planes[idx]
			out_planes = fuse_planes[idx+1]
			conv = nn.Conv2d(in_planes, out_planes, 
						kernel_size=grid_size, 
						stride=1, 
						padding=self.padding(dilation, grid_size), 
						dilation=dilation, 
						bias=False)
			norm = nn.GroupNorm(out_planes // 8, out_planes)
			relu = nn.ReLU()
			layers += [conv, norm, relu]

		return nn.Sequential(*layers)


	def to(self, device):
		device_model = super(OrientedNet, self).to(device)
		device_model.grids_v = self.grids_v.to(device)
		device_model.grids_h = self.grids_h.to(device)
		return device_model
	
	
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

		def apply_ori_conv(x, conv, grids, idx):
			x_1 = F.relu(conv(x, grids[idx]))
			x_2 = F.relu(conv(x, grids[idx+1]))
			return x_1 + x_2

		input_shape = inputs["image"].shape[-2:]
		features = super(OrientedNet, self).forward(inputs, return_intermediate=True)
		x_aspp = features["aspp"]
		seg_multi = compute_seg(x_aspp, input_shape, self.classifier)

		x_layer1 = self.reduce_layer1(features['layer1'])
		if self.att_mod is not None:
			x_aspp = F.interpolate(x_aspp, size=x_layer1.shape[-2:], mode='bilinear', align_corners=False)
			x_aspp = self.reduce_aspp(x_aspp)
			x_layer1 = self.att_mod(x_layer1, x_aspp)

		x_v = []
		x_h = []
		x_cat = []
		for idx, x in zip(inputs["angle_range_label"], x_layer1):

			idx = 0 if (idx == 255) else idx

			x_v_aux = x.unsqueeze(0)
			for _ori_conv in self.ori_convs:
				x_v_aux = apply_ori_conv(x_v_aux, _ori_conv, self.grids_v, idx)
			x_v.append(x_v_aux)
			
			x_h_aux = x.unsqueeze(0)
			for _ori_conv in self.ori_convs:
				x_h_aux = apply_ori_conv(x_h_aux, _ori_conv, self.grids_h, idx)
			x_h.append(x_h_aux)

			x_cat.append(torch.cat([x_v_aux, x_h_aux], dim=1))
		
		x_v = torch.cat(x_v, dim=0)
		x_h = torch.cat(x_h, dim=0)
		x_cat = torch.cat(x_cat, dim=0)
		x_fuse = self.fuse_net(x_cat)

		lines_seg = compute_seg(x_fuse, input_shape, self.line_clf_ori)

		result = OrderedDict()
		if self.training:
			result["out"] = OrderedDict()
			result["out"]["seg"] = seg_multi
			result["out"]["lines_seg"] = lines_seg
			if self.aux_clf_ori is not None:
				seg_v = compute_seg(x_v, input_shape, self.aux_clf_ori)
				seg_h = compute_seg(x_h, input_shape, self.aux_clf_ori)
				result["out"]["seg_v"] = seg_v
				result["out"]["seg_h"] = seg_h
		else:

			_, result["seg_mask_v2"] = predict(seg_multi, lines_seg.squeeze())
			result["seg_mask_v2"] = result["seg_mask_v2"].cpu()

			# plt.figure()
			# plt.imshow(seg)
			# plt.title("Seg")
			# plt.figure()
			# plt.imshow(seg_v)
			# plt.title("Seg V")
			# plt.figure()
			# plt.imshow(seg_h)
			# plt.title("Seg H")
			# plt.show()

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