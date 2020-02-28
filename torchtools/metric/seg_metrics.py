import numpy as np
import pdb
import torch
from .register import register

@register.attach('miou')
class JaccardIndex(object):
	def __init__(self, task, num_classes):
		self.n_classes = num_classes
		self.key = task
		self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

	def _fast_hist(self, label_true, label_pred, n_class):
		mask = (label_true >= 0) & (label_true < n_class)
		hist = np.bincount(
			n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
		).reshape(n_class, n_class)
		return hist

	def __call__(self, preds, data):
		label_preds = preds[self.key]['seg'].squeeze().unsqueeze(0)
		label_trues = data[self.key]['label'].squeeze().unsqueeze(0)
		for lt, lp in zip(label_trues, label_preds):
			self.confusion_matrix += self._fast_hist(lt.numpy().flatten(), lp.numpy().flatten(), self.n_classes)

	def value(self):
		hist = self.confusion_matrix
		pre_class_iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
		return np.mean(pre_class_iu)

	def reset(self):
		self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

@register.attach('per_class_iou')
class ClassIoU(JaccardIndex):
	def __init__(self, key, num_classes):
		super(ClassIoU, self).__init__(key, num_classes)
	def value(self,):
		hist = self.confusion_matrix
		return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
