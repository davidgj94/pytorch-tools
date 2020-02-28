from .register import register; register.load_modules()
from .loss import MultiTaskLoss

def get_loss(tasks_cfg):
	return MultiTaskLoss(tasks_cfg)

