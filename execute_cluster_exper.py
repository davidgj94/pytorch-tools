from train_val import main as train_val
import argparse
import os.path
import pdb
from pathlib import Path
import yaml
from torchtools.save import makedir
import shutil
import sys
from random import randint
from time import sleep

RESULTS_DIR = "/export/data_gpm/canard/cluster"

class StoreDict(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		kv={}
		if not isinstance(values, (list,)):
			values=(values,)
		for value in values:
			n, v = value.split('=')
			kv[n]=v
		setattr(namespace, self.dest, kv)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, required=True)
	parser.add_argument('--exper_params', action=StoreDict, required=False, nargs='+')
	parser.add_argument('--num_epochs', type=int, default=0)
	parser.add_argument('--dataset', type=str, required=True)
	return parser.parse_args()


def get_params_str(exper_params):
	params_str = ""
	for key, value in exper_params.items():
		params_str += "{}={}_".format(key, value)
	return params_str[:-1]


if __name__ == "__main__":

	args = parse_args()

	exper_name = os.path.basename(args.config).split('.')[0]
	exper_dir = os.path.join(RESULTS_DIR, exper_name)
	cfgs_dir = os.path.join(exper_dir, 'cfgs')
	checkpoint_dir = os.path.join(exper_dir, 'checkpoint')
	
	sleep(randint(1,15))

	makedir(cfgs_dir)
	makedir(checkpoint_dir)
	new_config_path = os.path.join(cfgs_dir, os.path.basename(args.config))
	if not os.path.exists(new_config_path):
		shutil.copyfile(args.config, new_config_path)

	with open(args.config, 'r') as f:
		exper_str = f.read().format(**args.exper_params)

	params_str = get_params_str(args.exper_params)
	exper_config_path = os.path.join(cfgs_dir, params_str + '.yml')

	with open(exper_config_path, 'w') as f:
		f.write(exper_str)
	
	print(params_str)
	
	makedir(checkpoint_dir)
	
	train_val(exper_config_path, args.num_epochs, args.dataset, use_cpu=False, root_checkpoint_dir=checkpoint_dir)