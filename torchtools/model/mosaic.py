import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt


class WarpNet(nn.Module):
	def __init__(self):
		super(WarpNet, self).__init__()
		self.offset_net = nn.Sequential(nn.Conv2d(2, 16, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(16, 32, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(32, 16, 1, 1, 0, 1, bias=True),
								   nn.Conv2d(16, 2, 1, 1, 0, 1, bias=True))
		self.offset_net.apply(init_conv)
		self.weights = nn.Parameter(torch.zeros((1, 2048), requires_grad=True))

	def forward(self, x_frame, x_mosaic, coords):
		batch_size = x_frame.shape[0]
		coords = F.interpolate(coords.transpose(3,1).transpose(3,2), size=x_frame.shape[-2:], mode='nearest')
		offset = self.offset_net(coords)
		#offset = 0.0
		coords = (coords + offset).transpose(3,1).transpose(1,2)
		

		x_frame_sampled = F.grid_sample(x_mosaic, coords)
		weights_sigmoid = torch.sigmoid(self.weights).repeat(batch_size, 1).view(batch_size, 2048, 1, 1)
		x_fuse = (1 - weights_sigmoid) * x_frame + weights_sigmoid * x_frame_sampled

		return x_fuse, offset


class MosaicNet(nn.Module):

	def __init__(self, frame_net, mosaic_backbone):
		super(MosaicNet, self).__init__()

		self.backbone = frame_net.backbone
		self.aspp = frame_net.aspp
		self.decoder = frame_net.decoder
		self.classifier = frame_net.classifier
		self.predict = frame_net.predict

		self.mosaic_backbone = mosaic_backbone
		self.warp_net = WarpNet()

	def get_device(self,):
		return self.classifier.weight.device

	def forward(self, inputs):

		device = self.get_device()
		frame = inputs['frame_img'].to(device)
		mosaic = inputs['mosaic_img'].to(device)
		grid_coords = inputs['grid_coords'].to(device)

		input_shape = frame.shape[-2:]
		frame_features = self.backbone(frame)
		x_frame = frame_features['out']
		mosaic_features = self.mosaic_backbone(mosaic)
		x_mosaic = mosaic_features['out']
		# print("mosaic_size:{}".format(x_mosaic.shape[-2:]))
		x_mosaic = F.interpolate(x_mosaic, scale_factor=16.0, mode='bilinear', align_corners=False)
		# print("mosaic_size:{}".format(x_mosaic.shape[-2:]))
		# print("frame_size:{}".format(x_frame.shape[-2:]))
		# pdb.set_trace()
		x, offset = self.warp_net(x_frame, x_mosaic, grid_coords)
		x_low = frame_features['skip1']
		x = self.aspp(x)
		x = self.decoder(x, x_low)
		x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
		x = self.classifier(x)
		
		if self.training:
			result = OrderedDict()
			result["out"] = OrderedDict()
			result["out"]["seg"] = x
			result["out"]["offset"] = offset
			return result
		else:
			return self.predict(x)

