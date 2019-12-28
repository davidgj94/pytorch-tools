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
from deeplabv3.dataset import get_dataset
from deeplabv3.model import get_model
from deeplabv3.optimizer import get_optimizer
from deeplabv3.scheduler import get_scheduler
from deeplabv3.loss import get_loss
import deeplabv3.utils as utils
from deeplabv3.metrics import RunningScore, AccuracyAngleRange
import pdb
import time
from tqdm import tqdm
from skimage.io import imsave, imread
import os.path
from argparse import ArgumentParser
from torchsummary import summary
from deeplabv3.save import ResultsSaver, CheckpointSaver
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
	parser.add_argument('--dataset', type=str, default='APR_TAX_RWY')
	parser.add_argument('--partitions', nargs='+', type=int)
	parser.add_argument('-train', dest='train', action='store_true')
	parser.set_defaults(train=False)
	parser.add_argument('-cpu', dest='use_cpu', action='store_true')
	parser.set_defaults(use_cpu=False)
	return parser.parse_args()


@timeit
def validate(val_model, val_loader, metric, vis_saver=None):

	val_model.eval()   # Set model to evaluate mode
	np.random.seed(0)

	with torch.set_grad_enabled(False):

		# Iterate over data.
		for _iter, data in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):
			preds = val_model(data)
			metric(preds, data)
			if vis_saver is not None:
				vis_saver(preds, data)


if __name__ == "__main__":
	args = parse_args()

	exper_name = os.path.basename(args.config).split(".")[0]
	dataset_dir = os.path.join('list', args.dataset)

	num_classes, training_cfg, val_cfg = utils.get_cfgs(args.config)
	device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

	for glob in Path(os.path.join(dataset_dir)).glob("*"):

		partition_dir = str(glob)
		partition_number = int(glob.parts[-1].split("_")[-1])

		if args.partitions is not None:
			if partition_number not in args.partitions:
				continue
		print('>> Partition {}'.format(partition_number))
		print()

		partition = os.path.join(dataset_dir, 'partition_{}').format(partition_number)
		_id_list_path = os.path.join(partition, '{}.txt')

		model_train = get_model(num_classes, training_cfg["model"], False).to(device)

		val_expers = {}
		for _val_exper in val_cfg['val_expers']:
			model_val = get_model(num_classes, _val_exper["model"]).to(device)
			id_list_path = _id_list_path.format('train' if args.train else 'val')
			val_dataloader = get_dataloader(id_list_path, _val_exper['dataset'], val_cfg['batch_size'], shuffle=False)
			val_expers[_val_exper['name']] = dict(model_val=model_val, val_dataloader=val_dataloader)

		checkpoint_dir = os.path.join('checkpoint', args.dataset, 'partition_{}', exper_name).format(partition_number)
		last_checkpoint_path = get_last_checkpoint(checkpoint_dir)
		if last_checkpoint_path is not None:
			last_checkpoint = torch.load(last_checkpoint_path)
			model_train.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
			current_epoch = last_checkpoint["epoch"]
			print("CHECKPOINT")
		else:
			current_epoch = 0

		results_dir = os.path.join('results', args.dataset, 'partition_{}', exper_name).format(partition_number)

		for val_exper_name, val_exper in val_expers.items():
			print('>> {}'.format(val_exper_name))
			val_model, val_dataloader = val_exper['model_val'], val_exper['val_dataloader']
			val_model.load_state_dict(model_train.state_dict(), strict=False)
			########### Aquí definimos la métrica que vamos a utilizar (en un futuro mediante fichero conf) ###########################
			metric = AccuracyAngleRange()
			# metric = RunningScore(num_classes)
			# save_vis_dir = os.path.join(results_dir, val_exper_name, 'epoch_{}'.format(current_epoch), 'train' if args.train else 'val')
			# save_vis = ResultsSaver(num_classes, val_dataloader.dataset.root, save_vis_dir, current_epoch)
			save_vis = None
			###########################################################################################################################
			validate(val_model, val_dataloader, metric, save_vis)
			print(">>>> score: {}".format(metric.value()))


