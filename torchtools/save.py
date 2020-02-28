from .vis import make_palette, vis_seg
import os.path
import os
import cv2
import numpy as np
import pdb
import json
import torch
import matplotlib
import shutil
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def makedir(_dir):
	if not os.path.exists(_dir):
		os.makedirs(_dir)


class VisSaver():
	def __init__(self, pred_name, label_name):
		self.pred_name = pred_name
		self.label_name = label_name

	def get_values(self, preds, data):
		preds = preds[self.pred_name]
		data = data[self.label_name]
		if len(preds.shape) > 2:
			preds = preds.squeeze(0)
		if len(data.shape) > 3:
			data = data.squeeze(0)
		return preds.numpy(), data.numpy()


class SegVisSaver(VisSaver):
	def __init__(self, n_classes, *args, **kwargs):
		self.palette = make_palette(n_classes)
		super(SegVisSaver, self).__init__(*args, **kwargs)

	def __call__(self, save_path, preds, data):
		seg_mask, img = self.get_values(preds, data)
		vis_img = vis_seg(img, seg_mask.astype(np.uint8), self.palette)
		cv2.imwrite(save_path, vis_img)


class AngleVisSaver(VisSaver):
	def __call__(self, save_path, preds, data):
		gabor_out, idx = self.get_values(preds, data)
		fig = plt.figure()
		plt.imshow(gabor_out[idx].squeeze(0))
		fig.savefig(save_path)


class ResultsSaver:

	def __init__(self, results_dir, metrics=None, vis_savers=None):

		self.results_dir = results_dir
		makedir(results_dir)
		self.metrics = metrics
		self.vis_savers = vis_savers
		self.score_path = os.path.join(results_dir, 'score.json')
		self.init_score_file()
		self.create_vis_dirs()

	def create_vis_dirs(self):
		if self.vis_savers is not None:
			for key in self.vis_savers.keys():
				vis_dir = os.path.join(self.results_dir, key)
				if os.path.exists(vis_dir):
					shutil.rmtree(vis_dir, ignore_errors=True)
				makedir(vis_dir)

	def init_score_file(self):
		if os.path.exists(self.score_path):
			os.remove(self.score_path)
		os.mknod(self.score_path)

	def update_metrics(self, preds, data):
		if self.metrics is not None:
			for metric in self.metrics.values():
				metric(preds, data)

	def save_vis(self, preds, data):
		if self.vis_savers is not None:
			for key, vis_saver in self.vis_savers.items():
				img_id = data["image_id"][0]
				save_path = os.path.join(self.results_dir, key, img_id)
				vis_saver(save_path, preds, data)

	def save_metrics(self,):

		scores = []
		for key, metric in self.metrics.items():
			scores.append({key: metric.value()})

		with open(self.score_path, 'w') as f:
			json.dump(scores, f)

class CheckpointSaver:

	def __init__(self, checkpoint_dir, start_epoch):
		self.checkpoint_dir = checkpoint_dir
		makedir(self.checkpoint_dir)
		self.start_epoch = start_epoch + 1


	def __call__(self, epoch, model, optimizer):

		epoch += self.start_epoch
		checkpoint_path = os.path.join(self.checkpoint_dir, 'epoch_{}.pth'.format(epoch))
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'learning_rate': optimizer.param_groups[0]['lr'],
			'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

class MetricSaver:

	def __init__(self, metric_name, save_dir, start_epoch):
		self.save_path = os.path.join(save_dir, '{}.json').format(metric_name)
		if os.path.exists(self.save_path):
			with open(self.save_path, 'r') as f:
				self.metrics = json.load(f)
		else:
			self.metrics = {}
		self.start_epoch = start_epoch
	
	def __call__(self, epoch, last_metric):

		epoch += self.start_epoch
		self.metrics["epoch_{}".format(epoch)] = last_metric
		with open(self.save_path, 'w') as f:
			json.dump(self.metrics, f)


class VideoSaver:

	def __init__(self, n_classes, save_path, fps=10.0):
		self.palette = make_palette(n_classes)
		self.save_path = save_path
		self.fps = int(fps)
		self.out = None
		
	def save_frame(self, img, pred):
		img = np.squeeze(img)
		pred = np.squeeze(pred)
		if self.out is None:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			self.out = cv2.VideoWriter(self.save_path, fourcc, self.fps, img.shape[:2][::-1])
		# plt.figure()
		# plt.imshow(vis_seg(img, pred, self.palette))
		# plt.show()
		self.out.write(vis_seg(img, pred, self.palette))

	def save_video(self):
		if self.out is not None:
			self.out.release()
			self.out = None




