from deeplabv3.model import deeplab
from deeplabv3.model import angle_detect
from deeplabv3.model import predict
from deeplabv3.model import mosaic
from .multi_frame import MultiFrameMerge

models_map = {'deeplabv3' : deeplab.Deeplabv3,
			  'deeplabv3+1' : deeplab.Deeplabv3Plus1,
			  'angle_detect': angle_detect.AngleDetect_v2,
			  'angle_net': angle_detect.AngleNet,
			  }

predict_map = {'argmax': predict.argmax_predict,
			   'line_dect': predict.line_detection,
			   'line_dect_multi': MultiFrameMerge(nframes=9),
			   }

output_stride_params = { 16: dict(replace_stride_with_dilation=[False, False, True], rates=[6, 12, 18]),
						 8:  dict(replace_stride_with_dilation=[False, True, True],  rates=[12, 24, 36]),
						 4 : dict(replace_stride_with_dilation=[True, True, True],   rates=[24, 48, 72]),
						 32: dict(replace_stride_with_dilation=[False, False, False], rates=[3, 6, 9])
						}

# def get_model(n_classes, cfg):
# 	return_layers = dict(layer4='out', layer1='skip1', layer3='aux')
# 	kw_backbone_args = dict(output_stride_params[cfg['stride']])
# 	kw_backbone_args.update(return_layers=return_layers)
# 	return models_map[cfg['name']](n_classes, cfg, kw_backbone_args)

def get_model(n_classes, cfg, aux=False):

	# if cfg['name'] == 'mosaic':
	# 	return _get_model_mosaic(n_classes, cfg, aux=False)

	return_layers = dict(layer4='out', layer1='skip1', layer3='aux')

	kw_backbone_args = dict(output_stride_params[cfg['stride']])
	kw_backbone_args.update(return_layers=return_layers)
	pretrained_model = deeplab.load_pretrained_model(kw_backbone_args)
	
	predict_key = cfg.get('predict', 'argmax')
	out_planes_skip = cfg.get('out_planes_skip', 48)
	model = models_map[cfg['name']](n_classes, 
									pretrained_model, 
									predict_map[predict_key], 
									aux=aux, 
									out_planes_skip=out_planes_skip)
	return model


# def _get_model_mosaic(n_classes, cfg, aux=False):

# 	def _load_pretrained_model(stride):
# 		return_layers = dict(layer4='out', layer1='skip1', layer3='aux')
# 		kw_backbone_args = dict(output_stride_params[stride])
# 		kw_backbone_args.update(return_layers=return_layers)
# 		pretrained_model = load_pretrained_model(kw_backbone_args)
# 		return pretrained_model

# 	pretrained_model = _load_pretrained_model(cfg['stride'])
# 	predict_key = cfg.get('predict', 'argmax')
# 	out_planes_skip = cfg.get('out_planes_skip', 48)
# 	base_model = models_map[cfg['base_model']](n_classes, 
# 										   pretrained_model, 
# 										   predict_map[predict_key], 
# 										   aux=aux, 
# 										   out_planes_skip=out_planes_skip)
# 	#base_model.load_state_dict(torch.load(cfg['init'])["model_state_dict"], strict=False)

# 	mosaic_backbone = _load_pretrained_model(cfg['mosaic-stride']).backbone

# 	return models_map['mosaic'](base_model, mosaic_backbone)

