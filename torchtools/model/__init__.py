from .register import register; register.load_modules()
from torchtools.model import deeplab
import pdb

output_stride_params = { 16: dict(replace_stride_with_dilation=[False, False, True], rates=[6, 12, 18]),
						 8:  dict(replace_stride_with_dilation=[False, True, True],  rates=[12, 24, 36]),
						 4 : dict(replace_stride_with_dilation=[True, True, True],   rates=[24, 48, 72]),
						 32: dict(replace_stride_with_dilation=[False, False, False], rates=[3, 6, 9])
						}


def get_model(n_classes, cfg):

	return_layers = dict(layer4='layer4', layer3='layer3', layer2='layer2', layer1='layer1')

	kw_backbone_args = dict(output_stride_params[cfg['stride']])
	kw_backbone_args.update(return_layers=return_layers)
	pretrained_model = deeplab.load_pretrained_model(kw_backbone_args)
	model_params = cfg.get('params', {})
	model = register.get(cfg['name'])(n_classes, 
									  pretrained_model,
									  **model_params)
	return model

