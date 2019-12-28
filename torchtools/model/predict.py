import torch
from torch.nn import functional as F
import cv2
import numpy as np
import pdb
from skimage.measure import label
import deeplabv3.lines as lines
import pdb
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from skimage.transform import hough_line
from scipy.signal import find_peaks
from deeplabv3.lines import general_form, find_intesect_borders, normal_form
from sklearn.cluster import MeanShift
from functools import partial

def getLargestCC(labels, segmentation):
    return np.argmax(np.bincount(labels.flat, weights=segmentation.flat))

def get_sortedCC(labels, segmentation):
	cc_count = np.bincount(labels.flat, weights=segmentation.flat)
	return np.argsort(cc_count)[::-1], cc_count

def argmax_predict(x):
	_, pred = torch.max(x, 1)
	return pred

############ Funcion de predict de deteccion y clustering de lineas #########
#############################################################################

def line_detection(line_probs, line_coeffs, sz):
	# line_coeffs = data['line_coeffs'].cpu().numpy().squeeze()

	def _cluster_lines(_lines, _is):

		_lines = _lines[_is]

		# pdb.set_trace()

		ms = MeanShift(bandwidth=50.0)
		pred = ms.fit_predict(_lines[:,0][...,np.newaxis])
		cluster_labels = np.unique(pred)

		cluster_lines = []
		for _label in cluster_labels.tolist():
			close_lines = _lines[pred == _label]
			cluster_lines.append(close_lines.mean(0).tolist())

		return cluster_lines

	line_probs = line_probs.cpu().detach().numpy()
	selected_lines = line_coeffs[line_probs > 0.5]
	pre_mask = lines.create_grid(sz, selected_lines.tolist())

	# plt.figure()
	# plt.imshow(pre_mask)

	_cluster_lines = partial(_cluster_lines, selected_lines)
	thetas = np.rad2deg(selected_lines[:,1])
	theta_v = thetas[np.logical_and(thetas > -30.0, thetas < 30.0)].mean()
	theta_h = thetas[thetas > 30.0].mean()
	is_v = (np.abs(thetas - theta_v) < 5.0)
	is_h = (np.abs(thetas - theta_h) < 5.0)
	selected_lines = _cluster_lines(is_v) + _cluster_lines(is_h)

	post_mask = lines.create_grid(sz, selected_lines)

	# plt.figure()
	# plt.imshow(post_mask)
	# plt.show()

	return pre_mask, post_mask

#############################################################################



############ Funcion de predict cuando hago regresion de líneas #############
#############################################################################

def predict_reg_lines(proposed_lines, score, iou, reg_offset, sz):

	def _cluster_lines(_lines, iou, reg_offset, _is):

		_lines = _lines[_is]
		iou = iou[_is]
		reg_offset = reg_offset[_is]

		pdb.set_trace()

		ms = MeanShift(bandwidth=50.0)
		pred = ms.fit_predict(_lines[:,0])
		cluster_labels = np.unique(pred)

		cluster_lines = []
		for _label in cluster_labels.tolist():
			idx = np.argmax(iou[pred == _label])
			cluster_offset = reg_offset[idx] * 45
			cluster_line = _line[idx]


			cluster_line[0] += cluster_offset
			cluster_lines.append(cluster_line)

		return cluster_lines

	close_lines = (score > 0.5)
	proposed_lines = proposed_lines[close_lines]
	iou = iou[close_lines]
	reg_offset = reg_offset[close_lines]
	_cluster_lines = partial(_cluster_lines, proposed_lines, iou, reg_offset)

	thetas = np.rad2deg(proposed_lines[:,1])
	theta_v = thetas[np.logical_and(thetas > -30.0, thetas < 30.0)].mean()
	theta_h = thetas[thetas > 30.0].mean()
	is_v = (np.abs(thetas - theta_v) < 5.0)
	is_h = (np.abs(thetas - theta_h) < 5.0)

	pdb.set_trace()

	cluster_lines = _cluster_lines(is_v) + _cluster_lines(is_h)
	mask = lines.create_grid(sz, cluster_lines, width=10)
	plt.imshow(mask)
	plt.show()
	return mask

#############################################################################



############ Funcion de predict auxiliar multiframe #########################
#############################################################################

def _line_detection(x, kernel=np.ones((3,3), np.uint8), iterations=5):

	pred = np.squeeze(argmax_predict(x).cpu().numpy()).astype(np.uint8)
	pred_1 = (pred == 1).astype(np.uint8)
	APR_pred = np.logical_or(pred == 0, pred == 1)
	pred_S = pred.copy()
	pred_S[APR_pred] = 0

	if np.any(APR_pred):

		APR_pred_cc = label(APR_pred.astype(np.uint8))
		APR_pred = (APR_pred_cc == getLargestCC(APR_pred_cc, APR_pred.astype(np.uint8)))

		if np.sum(APR_pred) > 0.3 * np.prod(APR_pred.shape):

			pred_1 *= APR_pred
			pred_1_cc = label(pred_1)
			CC_indices, cc_count = get_sortedCC(pred_1_cc, pred_1)
			cc1_size = cc_count[CC_indices[0]]
			cc2_size = cc_count[CC_indices[1]]
			if (float(cc2_size) / cc1_size) > 0.1:
				pred_1 = np.logical_or(pred_1_cc == CC_indices[0], pred_1_cc == CC_indices[1]).astype(np.uint8)
			else:
				pred_1 = (pred_1_cc == CC_indices[0]).astype(np.uint8)
	
			pred_1 = cv2.erode(pred_1, kernel, iterations=iterations)

			angles_coarse = np.linspace(np.deg2rad(-45), np.deg2rad(135), 180)
			hspace, angles, distances = hough_line(pred_1, angles_coarse)
			tresh = np.max(hspace) * 0.1
			hspace[hspace < tresh] = 0

			bin_length = 5
			resol = 1.0
			hist, bins_edges = lines.compute_hist(hspace, angles, bin_length=bin_length)
			hist_peaks, prop = find_peaks(hist, height=0.1 * max(hist))
			indices = np.argsort(prop['peak_heights'])[::-1]
			hist_peaks = hist_peaks[indices] * bin_length * resol - 45.0

			main_peaks = None
			for peak in hist_peaks.tolist():
				dist = np.abs(hist_peaks - peak)
				pair_peaks_loc = np.logical_and((90 - bin_length * resol) <= dist,
				                                 dist <= (90 + bin_length * resol))
				if np.any(pair_peaks_loc):
					pair_peak = hist_peaks[pair_peaks_loc][0]
					main_peaks = (peak, pair_peak)
					break
			
			detected_lines = []
			if main_peaks is not None:

				for peak in main_peaks:
					center_angle = (2 * peak + bin_length * resol) / 2
					angles_range = (center_angle - 2 * bin_length * resol, center_angle + 2 * bin_length * resol)
					_, line_angles, line_dist = lines.search_lines(pred_1, angles_range, npoints=1000, min_distance=100, min_angle=300, threshold=None)
					detected_lines += lines.get_lines(line_dist, line_angles)

				# detected_lines = [general_form(*line_coeff) for line_coeff in detected_lines]

	return pred_S, detected_lines

#############################################################################























