from .register import register; register.load_modules()

def get_metric(metric_cfg):
	return register.get(metric_cfg['name'])(**metric_cfg['params'])