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
from .gabor import GaborNet_v4, GaborDecoder, GatedGabor
from .gabor_old import GaborNet_v1, GaborClassfier_v2, GaborClassfier_v3

def compute_seg(x, output_shape, classifier):
	x = F.interpolate(x, size=output_shape, mode='bilinear', align_corners=False)
	return classifier(x)

def predict(seg, line_seg):
	_, seg_mask = torch.max(seg, 1)
	seg_mask = seg_mask.squeeze()
	lines_mask = torch.sigmoid(line_seg) > 0.5
	seg_mask_3c = torch.clamp(seg_mask-1, min=0, max=2)
	seg_mask_v1 = copy.deepcopy(seg_mask)
	seg_mask_v1[seg_mask_3c == 0] = lines_mask[seg_mask_3c == 0].type(seg_mask_v1.type())
	return seg_mask, seg_mask_v1


@register.attach('angle_net_v1')
class AngleNet_v1(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48, combine=False):

		super(AngleNet_v1, self).__init__(n_classes - 1, 
										  pretrained_model, 
										  aux=False, 
										  out_planes_skip=out_planes_skip)

		self.lines_decoder = DeepLabDecoder(256, out_planes=out_planes_skip, out_planes_decoder=64)
		self.lines_decoder.apply(init_conv)
		self.gabor_net = GaborNet_v1(planes=[64, 32, 64], combine=combine) # no inicializar, ya se inicializa solo
		self.test_tresh = 0.5

	def forwametricrd(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v1, self).forward(inputs, return_intermediate=True)

		seg = compute_seg(features["decoder"], input_shape, self.classifier)

		x_aspp = features['aspp']
		x_low = features['layer1']
		x_decoder = self.lines_decoder(x_aspp, x_low)
		line_seg = self.gabor_net(x_decoder, input_shape)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['line_seg'] = line_seg
			result['out']['seg'] = seg

		return result

@register.attach('angle_net_v2')
class AngleNet_v2(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48, combine=False):

		super(AngleNet_v2, self).__init__(n_classes - 1, 
										pretrained_model, 
										aux=False, 
										out_planes_skip=out_planes_skip)

		self.gabor_clf = GaborClassfier_v2(combine=combine) # no inicializar, ya se inicializa solo
		self.test_tresh = 0.5

	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v2, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]

		seg = compute_seg(x_decoder, input_shape, self.classifier)
		lines_seg = self.gabor_clf(x_decoder, input_shape)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['seg'] = seg
			result['out']['line_seg'] = lines_seg
		
		return result

@register.attach('angle_net_v3')
class AngleNet_v3(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48, combine=False):

		super(AngleNet_v3, self).__init__(n_classes - 1,
									   pretrained_model,
									   aux=aux,
									   out_planes_skip=out_planes_skip)

		self.gabor_clf = GaborClassfier_v3(combine=combine)  # no inicializar, ya se inicializa solo
		if aux:
			self.aux_gabor_clf = GaborClassfier_v3()
		self.test_tresh = 0.5

	def plot_out(self, x_out):
		for _x_out in x_out:
			plt.figure()
			plt.imshow(_x_out.cpu().numpy())

	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v3, self).forward(inputs, return_intermediate=True)
		x_decoder = features["decoder"]

		seg = compute_seg(x_decoder, input_shape, self.classifier)
		lines_seg = self.gabor_clf(x_decoder, input_shape)

		if "aux" in features:
			x_aux = features["aux"]
			seg_aux = compute_seg(x_aux, input_shape, self.aux_clf)
			x_aux_up = F.interpolate(x_aux, size=x_decoder.shape[-2:], mode='bilinear', align_corners=False)
			lines_seg_aux = self.aux_gabor_clf(x_aux_up, input_shape)

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['seg'] = seg
			result['out']['line_seg'] = lines_seg
			if "aux" in features:
				result['aux'] = OrderedDict()
				result['aux']['seg'] = seg_aux
				result['aux']['line_seg'] = lines_seg_aux

@register.attach('angle_net_v5')
class AngleNet_v5(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		super(AngleNet_v5, self).__init__(n_classes,
									pretrained_model, 
									aux=False, 
									out_planes_skip=out_planes_skip)

		self.lines_decoder = GaborDecoder(256, out_planes_low=128, out_planes_decoder=128)
		self.lines_clf = nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.lines_clf)
		self.test_tresh = 0.5

	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v5, self).forward(inputs, return_intermediate=True)

		seg = compute_seg(features["decoder"], input_shape, self.classifier)

		x_aspp = features['aspp']
		x_low = features['layer1']
		x_lines_decoder = self.lines_decoder(x_aspp, x_low, inputs["angle_range_label"].item())
		lines_seg = compute_seg(x_lines_decoder, input_shape, self.lines_clf)
		lines_seg = lines_seg.squeeze()
		
		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['line_seg'] = lines_seg
			result['out']['seg'] = seg
		else:
			seg_mask, seg_mask_v2 = predict(seg, lines_seg)
			result['seg_mask'] = seg_mask.cpu()
			result['seg_mask_v2'] = seg_mask_v2.cpu()

			""" plt.figure()
			plt.imshow(torch.sigmoid(lines_seg).squeeze().cpu().numpy())
			plt.figure()
			plt.imshow(seg.squeeze()[1].cpu().numpy())
			plt.show() """

		return result

@register.attach('angle_net_v4')
class AngleNet_v4(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48,):

		super(AngleNet_v4, self).__init__(n_classes,
									pretrained_model, 
									aux=False, 
									out_planes_skip=out_planes_skip)

		self.lines_decoder = DeepLabDecoder(256, out_planes=out_planes_skip, out_planes_decoder=128)
		self.lines_decoder.apply(init_conv)
		self.gabor_net = GaborNet_v4(in_planes=128, out_planes=256) # no inicializar, ya se inicializa solo
		self.lines_clf = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.lines_clf)
		self.test_tresh = 0.5

	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v4, self).forward(inputs, return_intermediate=True)

		seg = compute_seg(features["decoder"], input_shape, self.classifier)

		x_aspp = features['aspp']
		x_low = features['layer1']
		x_decoder = self.lines_decoder(x_aspp, x_low)
		x_gabor = self.gabor_net(x_decoder, inputs['angle_range_label'].item())
		line_seg = compute_seg(x_gabor, input_shape, self.lines_clf).squeeze()

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['line_seg'] = line_seg
			result['out']['seg'] = seg
		else:
			plt.imshow(torch.sigmoid(line_seg).squeeze().cpu().numpy())
			plt.show()

		return result

@register.attach('angle_net_v6')
class AngleNet_v6(Deeplabv3Plus):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48, rate=4):

		super(AngleNet_v6, self).__init__(n_classes, pretrained_model, 
									aux=False, 
									out_planes_skip=out_planes_skip)

		n_channels_layer1 = 64
		self.reduce_layer1 = nn.Sequential(
			nn.Conv2d(256, n_channels_layer1, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(n_channels_layer1 // 8, n_channels_layer1),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_layer1.apply(init_conv)

		n_channels_decoder = n_channels_layer1 // 4
		self.reduce_decoder = nn.Sequential(
			nn.Conv2d(256, n_channels_decoder, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(n_channels_decoder // 8, n_channels_decoder),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.reduce_decoder.apply(init_conv)

		self.gabor_gated_conv = GatedGabor(n_channels_layer1)

		self.fuse_net = nn.Sequential(
			nn.Conv2d(2 * n_channels_layer1, n_channels_layer1, kernel_size=5, stride=1, padding=2*rate, dilation=rate, bias=False),
			nn.GroupNorm(int(n_channels_layer1/8), n_channels_layer1),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(n_channels_layer1, n_channels_layer1, kernel_size=5, stride=1, padding=2*rate, dilation=rate, bias=False),
			nn.GroupNorm(int(n_channels_layer1/8), n_channels_layer1),
			nn.ReLU(),)
		self.fuse_net.apply(init_conv)

		self.line_clf = nn.Conv2d(n_channels_layer1, 1, kernel_size=1, stride=1, bias=False)
		init_conv(self.line_clf)
	
	def forward(self, inputs):

		input_shape = inputs["image"].shape[-2:]
		features = super(AngleNet_v6, self).forward(inputs, return_intermediate=True)

		x_decoder = features["decoder"]
		x_layer1 = features['layer1']

		seg = compute_seg(x_decoder, input_shape, self.classifier)

		x_layer1 = self.reduce_layer1(x_layer1)
		x_decoder = self.reduce_decoder(x_decoder)
		x_gabor = self.gabor_gated_conv(x_layer1, x_decoder, inputs["angle_range_label"].item())

		x_fuse = self.fuse_net(x_gabor)
		line_seg = compute_seg(x_fuse, input_shape, self.line_clf).squeeze()

		result = OrderedDict()

		if self.training:
			result['out'] = OrderedDict()
			result['out']['line_seg'] = line_seg
			result['out']['seg'] = seg
		else:
			seg_mask, seg_mask_v2 = predict(seg, lines_seg)
			result['seg_mask'] = seg_mask.cpu()
			result['seg_mask_v2'] = seg_mask_v2.cpu()
			# plt.imshow(torch.sigmoid(line_seg).squeeze().cpu().numpy())
			# plt.show()

		return result

	

	









