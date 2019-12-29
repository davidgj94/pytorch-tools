from .register import register; register.load_modules()
from .loss import Loss

def get_loss(losses_list):
	return Loss(losses_list)

""" def get_loss(cfg):
	loss = register.get(cfg['name'])
	if 'params' in cfg:
		return loss(**cfg['params'])
	return loss """

