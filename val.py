import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchtools.dataset import get_dataset
from torchtools.model import get_model
from torchtools.loss import get_loss
import torchtools.utils as utils
from torchtools.metrics import RunningScore, AccuracyAngleRange
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
from torchsummary import summary
from torchtools.save import ResultsSaver, SegVisSaver
from pathlib import Path
import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('diff_time = {}'.format(te-ts))
        return result

    return timed

def get_last_checkpoint(checkpoint_dir):

	if not os.path.exists(checkpoint_dir):
		return None
	else:
		checkpoints_globs = list(Path(checkpoint_dir).glob('*.pth'))
		if len(checkpoints_globs) == 0:
			return None
		key = lambda x: int(os.path.basename(str(x)).split('.')[0].split('_')[-1])
		last_checkpoint_path = str(sorted(checkpoints_globs, key=key, )[-1])
		return last_checkpoint_path

def get_dataloader(id_list_path, dataset_cfg, batch_size, shuffle=True):

	dataset_params = dict(dataset_cfg['params'])
	dataset_params.update(id_list_path=id_list_path)
	dataset = get_dataset(dataset_cfg['name'])(**dataset_params)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('-train', dest='train', action='store_true')
	parser.add_argument('--epoch', type=int)
	parser.set_defaults(train=False)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


@timeit
def validate(val_model, val_loader, result_saver):

	val_model.eval()   # Set model to evaluate mode
	np.random.seed(0)

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, data in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):
			preds = val_model(data)
			result_saver.update_metrics(preds, data)
			result_saver.save_vis(preds, data)
		
	result_saver.save_metrics()


if __name__ == "__main__":

	args = parse_args()

	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	model_train = get_model(num_classes, training_cfg["model"]).to(device)

	id_list_path = os.path.join('test', 'list', '{}.txt').format('train' if args.train else 'val')
	val_expers = {}
	for _val_exper in val_cfg['val_expers']:
		model_val = get_model(num_classes, _val_exper["model"]).to(device)
		val_dataloader = get_dataloader(id_list_path, _val_exper['dataset'], val_cfg['batch_size'], shuffle=False)
		val_expers[_val_exper['name']] = dict(model_val=model_val, val_dataloader=val_dataloader)

	exper_name = os.path.basename(args.config).split(".")[0]
	checkpoint_dir = os.path.join('test', 'checkpoint', exper_name)
	if args.epoch is not None:
		last_checkpoint_path = os.path.join(checkpoint_dir, "epoch_{}.pth".format(args.epoch))
	else:
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)

	if last_checkpoint_path is not None:
		last_checkpoint = torch.load(last_checkpoint_path)
		model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
		current_epoch = last_checkpoint["epoch"]
		print("CHECKPOINT: {}".format(last_checkpoint_path))
	else:
		current_epoch = 0

	result_dir = os.path.join('test', 'results', exper_name, 'epoch_{}'.format(current_epoch))
	for val_exper_name, val_exper in val_expers.items():
		print('>> {}'.format(val_exper_name))
		val_model, val_dataloader = val_exper['model_val'], val_exper['val_dataloader']
		val_model.load_state_dict(model_train.state_dict(), strict=False)
		metric = RunningScore(4, pred_name='seg', label_name='mask_test')
		angle_metric = AccuracyAngleRange(pred_name="hist", label_name="angle_range_label")
		vis_saver = SegVisSaver(4, pred_name='seg', label_name='vis_image')
		result_saver = ResultsSaver(result_dir, metrics=dict(angle=angle_metric))
		validate(val_model, val_dataloader, result_saver)