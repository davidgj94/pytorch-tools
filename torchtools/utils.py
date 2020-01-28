import yaml
import pdb
from collections import OrderedDict
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from functools import partial
from .save import makedir


# def list_to_od(_list):
# 	od = OrderedDict()
# 	for el in _list:
# 		key = str(*el.keys())
# 		value = el[key]
# 		if isinstance(value, list):
# 			od[key] = list_to_od(value)
# 		else:
# 			od[key] = value
# 	return od

def get_cfgs_v2(config_path):

	with open(config_path, 'r') as stream:
	    try:
	    	config = yaml.safe_load(stream)
	    	return config
	    except yaml.YAMLError as exc:
	        print(exc)
	return None


def get_cfgs(config_path):

	with open(config_path, 'r') as stream:
	    try:
	        config = yaml.safe_load(stream)
	        num_classes = config["num_classes"]
	        training_cfg = config["training"]
	        val_cfg = config["validation"]
	        return num_classes, training_cfg, val_cfg
	    except yaml.YAMLError as exc:
	        print(exc)
	return None

def get_cfgs_video(config_path):

	exper_name = os.path.basename(config_path).split('.')[0]

	with open(config_path, 'r') as stream:
	    try:
	        config = yaml.safe_load(stream)
	        num_classes = config["num_classes"]
	        video_cfg = config["video"]
	        return num_classes, video_cfg
	    except yaml.YAMLError as exc:
	        print(exc)
	return None


def check_gradient(name, tensor_stats=False):

	def _check_gradient(func):

		def inner_func(*args, **kwargs):

			x = func(*args, **kwargs)

			def hook_func(grad):

				print('-- GRADIENT CHECKING OF {} --'.format(name))
				print('>> Shape: {}'.format(grad.shape))
				grad_mean = torch.abs(grad).reshape(-1).max()
				print('>> Max Grad value: {}'.format(grad_mean))
				if tensor_stats:
					x_mean = torch.abs(x).reshape(-1).max()
					print('>> Max X value: {}'.format(x_mean))
				print()

			x.register_hook(hook_func)

			return x

		return inner_func

	return _check_gradient

def save_heatmaps(segs, save_path, v_min=None, v_max=None):

	if v_min is not None:
		segs = torch.clamp(segs, min=v_min)
	
	if v_max is not None:
		segs = torch.clamp(segs, max=v_max)

	cbar_save_path = os.path.join(save_path, 'cbar.png')
	heatmaps_save_dir = os.path.join(save_path, 'heats')
	makedir(heatmaps_save_dir)

	min_value = segs.min()
	max_value = segs.max()
	dummy_seg = np.random.uniform(low=min_value, high=max_value, size=(100, 100))
	X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
	mpb = plt.pcolormesh(X,Y,dummy_seg,cmap='viridis')
	_,ax = plt.subplots()
	plt.colorbar(mpb,ax=ax)
	ax.remove()
	plt.savefig(cbar_save_path,bbox_inches='tight')

	cmap = plt.get_cmap('viridis')
	segs -= segs.min()
	segs /= segs.max()
	for idx, seg in enumerate(segs):
		heat_save_path = os.path.join(heatmaps_save_dir, 'heat_{}.png').format(idx)
		colored_seg = (cmap(seg.numpy())[:, :, :3] * 255).astype(np.uint8)
		Image.fromarray(colored_seg).save(heat_save_path)
	








# if __name__ == "__main__":
# 	with open("config/resnet101.yml", 'r') as stream:
# 	    try:
# 	        config = yaml.safe_load(stream)
# 	        if 'augmentations' in config['dataset']['train']:
# 	        	config['dataset']['train']['augmentations'] = list_to_od(config['dataset']['train']['augmentations'])
# 	        pdb.set_trace()
# 	    except yaml.YAMLError as exc:
# 	        print(exc)