from .register import register
import deeplabv3.loss.loss


def get_loss(cfg):
	loss = register.get(cfg['name'])
	if 'params' in cfg:
		return loss(**cfg['params'])
	return loss

