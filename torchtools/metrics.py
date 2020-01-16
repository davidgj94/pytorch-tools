# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import pdb
import torch


class AccuracyAngleRange(object):
    def __init__(self, pred_name, label_name):
        self.pred_name = pred_name
        self.label_name = label_name
        self.reset()

    def reset(self,):
        self.preds = []
        self.labels = []

    def __call__(self, preds, data):
        preds = preds[self.pred_name]
        labels = data[self.label_name].numpy().tolist()
        for _pred, _label in zip(preds, labels):
            _pred = np.argmax(_pred.cpu().numpy())
            self.preds.append(_pred)
            self.labels.append(_label)

    def value(self,):
        return np.mean(np.array(self.preds) == np.array(self.labels))


class RunningScore(object):
    def __init__(self, n_classes, pred_name, label_name):
        self.n_classes = n_classes
        self.pred_name = pred_name
        self.label_name = label_name
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def __call__(self, preds, data):
        label_preds = preds[self.pred_name].squeeze().unsqueeze(0)
        label_trues = data[self.label_name].squeeze().unsqueeze(0)
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.numpy().flatten(), lp.numpy().flatten(), self.n_classes)

    def value(self):
        hist = self.confusion_matrix
        pre_class_iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        return pre_class_iu.tolist()

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class RunningScoreLines(object):
    def __init__(self, pred_name, label_name):
        self.n_classes = 2
        self.pred_name = pred_name
        self.label_name = label_name
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def __call__(self, preds, data):
        label_preds = (torch.sigmoid(preds[self.pred_name]) > 0.9).cpu().numpy()
        label_trues = data[self.label_name].squeeze(0).cpu().numpy()
        self.confusion_matrix += self._fast_hist(label_trues.astype(int), label_preds.astype(int), self.n_classes)

    def value(self):
        hist = self.confusion_matrix
        pre_class_iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        return pre_class_iu.tolist()

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg
