import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision

import torchvision.models
import torch.utils.data
import pdb
from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models.resnet as resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, ASPP
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchtools.lines as lines

class Deeplabv3(nn.Module):

	def __init__(self, n_classes, pretrained_model, aux=False):
		super(_Deeplabv3, self).__init__()
		self.backbone = pretrained_model.backbone
		self.aspp = list(pretrained_model.classifier.children())[0])
		self.classifier = nn.Conv2d(256, n_classes, 1, 1, 0, 1, bias=False)
		init_conv(self.classifier)

		self.aux_clf = None
		if aux:
			aux_clf = pretrained_model.aux_classifier
			aux_clf_out = nn.Conv2d(256, n_classes, 1, 1, 0, 1, bias=False)
			init_conv(aux_clf_out)
			aux_clf_layers = list(aux_clf.children())
			aux_clf_layers[-1] = aux_clf_out
			self.aux_clf = nn.Sequential(*aux_clf_layers)

	def get_device(self,):
		return self.classifier.weight.device

	def trainable_parameters(self,):
		return self.parameters()
	
	def forward(self, inputs, return_intermediate=False):

		x = inputs['image'].to(self.get_device())
		input_shape = x.shape[-2:]
		features = self.backbone(x)
		x = features["layer4"]
		x_features = self.aspp(x)
		features['aspp'] = x_features

		result = OrderedDict()
		
		if self.training and (self.aux_clf is not None):
			x = features["layer3"]
			x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
			seg_aux = self.aux_clf(x)
			result["aux"] = OrderedDict()
			result["aux"]["seg"] = seg_aux
			
		if return_intermediate:
			return result, features
		else:
			x = F.interpolate(features['aspp'], size=input_shape, mode='bilinear', align_corners=False)
			seg = self.classifier(x)
			result["seg"] = seg
			return result



class Deeplabv3Plus(Deeplabv3):
	def __init__(self, n_classes, pretrained_model, aux=False, out_planes_skip=48):
		super(Deeplabv3Plus, self).__init__(n_classes, pretrained_model, aux=aux)
		self.decoder = DeepLabDecoder(256, out_planes=out_planes_skip)
		self.decoder.apply(init_conv)

	def forward(self, inputs, return_intermediate=False):
		result, features = super(Deeplabv3Plus).forward(inputs, return_intermediate=True)
		x_aspp = features['aspp']
		x_low = features['layer1']
		x_features = self.decoder(x_aspp, x_low)
		features["decoder"] = x_features

		if return_intermediate:
			return result, features
		else:
			input_shape = inputs["image"].shape[-2:]
			x = F.interpolate(features["decoder"], size=input_shape, mode='bilinear', align_corners=False)
			seg = self.classifier(x)
			result["seg"] = seg
			return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabDecoder(nn.Module):
	def __init__(self, in_planes, out_planes=48):
		super(DeepLabDecoder, self).__init__()
		self.reduce_conv = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.GroupNorm(int(out_planes/8), out_planes),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(256 + out_planes, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.GroupNorm(32, 256),
			nn.ReLU())

	def forward(self, x, x_low):
		low_size = x_low.shape[-2:]
		high_size = x.shape[-2:]
		x_low = self.reduce_conv(x_low)
		if low_size > high_size:
			x = F.interpolate(x, size=low_size, mode='bilinear', align_corners=False)
		x = torch.cat([x_low, x], dim=1)
		x = self.fuse_conv(x)
		return x
		

def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True, replace_stride_with_dilation=[False, True, True], rates=[12, 24, 36], return_layers = {'layer4': 'out'}):
	backbone = resnet.__dict__[backbone_name](
		pretrained=pretrained_backbone,
		replace_stride_with_dilation=replace_stride_with_dilation)

	backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

	aux_classifier = None
	if aux:
		inplanes = 1024
		aux_classifier = FCNHead(inplanes, num_classes)

	model_map = {
		'deeplabv3': (DeepLabHead, DeepLabV3),
		'fcn': (FCNHead, FCN),
	}
	inplanes = 2048
	classifier = model_map[name][0](inplanes, num_classes, rates)
	base_model = model_map[name][1]

	model = base_model(backbone, classifier, aux_classifier)

	return model

def load_pretrained_model(kw_backbone_args):
	pretrained_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21)
	model = _segm_resnet('deeplabv3', 'resnet101', 21, True, **kw_backbone_args)
	model.load_state_dict(pretrained_model.state_dict(), strict=True)
	return model

def init_conv(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)

