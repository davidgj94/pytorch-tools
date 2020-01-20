from torch.nn import functional as F
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def cross_entropy(preds, label):
	return F.cross_entropy(preds, label, ignore_index=255)


def binary_loss(preds, label, weights, pos_weight=None):

	if pos_weight is not None:
		bin_loss = F.binary_cross_entropy_with_logits(preds, label, 
												reduction="none", 
												pos_weight=pos_weight,)
	else:
		bin_loss = F.binary_cross_entropy_with_logits(preds, label, 
												reduction="none",)

	return (bin_loss * weights).sum() / (weights.sum() + 1.0)