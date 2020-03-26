#!/usr/bin/env python3


import math
import numpy as np
from skimage.morphology import skeletonize
import torchtools.road_utils.graph_utils as graph_utils
import torchtools.road_utils.sknw as sknw
import matplotlib.pyplot as plt
import pdb


def getKeypoints(mask, thresh=0.8, is_gaussian=True, is_skeleton=False, smooth_dist=10):
	"""
	Generate keypoints for binary prediction mask.

	@param mask: Binary road probability mask
	@param thresh: Probability threshold used to cnvert the mask to binary 0/1 mask
	@param gaussian: Flag to check if the given mask is gaussian/probability mask
					from prediction
	@param is_skeleton: Flag to perform opencv skeletonization on the binarized
						road mask
	@param smooth_dist: Tolerance parameter used to smooth the graph using
						RDP algorithm

	@return: return ndarray of road keypoints
	"""

	if is_gaussian:
		mask /= 255.0
		mask[mask < thresh] = 0
		mask[mask >= thresh] = 1

	h, w = mask.shape
	if is_skeleton:
		ske = mask
	else:
		ske = skeletonize(mask).astype(np.uint16)

	graph = sknw.build_sknw(ske, multi=True)

	segments = graph_utils.simplify_graph(graph, smooth_dist)
	linestrings_1 = graph_utils.segmets_to_linestrings(segments)
	linestrings = graph_utils.unique(linestrings_1)

	keypoints = []
	for line in linestrings:
		linestring = line.rstrip("\n").split("LINESTRING ")[-1]
		points_str = linestring.lstrip("(").rstrip(")").split(", ")
		## If there is no road present
		if "EMPTY" in points_str:
			return keypoints
		points = []
		for pt_st in points_str:
			x, y = pt_st.split(" ")
			x, y = float(x), float(y)
			points.append([x, y])

			x1, y1 = points[0]
			x2, y2 = points[-1]
			zero_dist1 = math.sqrt((x1) ** 2 + (y1) ** 2)
			zero_dist2 = math.sqrt((x2) ** 2 + (y2) ** 2)

			if zero_dist2 > zero_dist1:
				keypoints.append(points[::-1])
			else:
				keypoints.append(points)
	return keypoints

def merge_masks(concat_masks, bool_indices):
	new_mask = concat_masks[bool_indices].sum(0)
	return np.clip(new_mask, a_min=0.0, a_max=1.0)


def getVectorMapsAngles(shape, keypoints, theta=5, bin_size=10, margin=0.1):
	"""
	Convert Road keypoints obtained from road mask to orientation angle mask.
	Reference: Section 3.1
		https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf

	@param shape: Road Label/PIL image shape i.e. H x W
	@param keypoints: road keypoints generated from Road mask using
						function getKeypoints()
	@param theta: thickness width for orientation vectors, it is similar to
					thicknes of road width with which mask is generated.
	@param bin_size: Bin size to quantize the Orientation angles.

	@return: Retun ndarray of shape H x W, containing orientation angles per pixel.
	"""

	height, width = shape
	anchor_angles = np.arange(0.0, 180.0 + bin_size, bin_size)
	n_angles = len(anchor_angles)
	ori_gt = np.zeros((n_angles - 1, height, width), dtype=np.float32)
	ori_weights = np.zeros_like(ori_gt, dtype=np.float32)
	if len(keypoints) < 2:
		return ori_gt, ori_weights

	rectangle_angles = []
	rectangle_masks = []
	for j in range(len(keypoints)):
		_mask_aux = np.zeros(shape, dtype=np.float32)
		for i in range(1, len(keypoints[j])):

			a = keypoints[j][i - 1]
			b = keypoints[j][i]
			ax, ay = a[0], a[1]
			bx, by = b[0], b[1]
			bax = bx - ax
			bay = by - ay
			norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
			bax /= norm
			bay /= norm

			min_w = max(int(round(min(ax, bx) - theta)), 0)
			max_w = min(int(round(max(ax, bx) + theta)), width)
			min_h = max(int(round(min(ay, by) - theta)), 0)
			max_h = min(int(round(max(ay, by) + theta)), height)

			_theta = math.degrees(math.atan2(bay, bax))
			_theta = (_theta -90) % 180
			rectangle_angles.append(_theta)

			_mask = np.zeros(shape, dtype=np.float32)
			for h in range(min_h, max_h):
				for w in range(min_w, max_w):
					px = w - ax
					py = h - ay
					dis = abs(bax * py - bay * px)
					if dis <= theta:
						_mask[h, w] = 1.0
			_mask_aux += _mask
			rectangle_masks.append(_mask)
		# plt.imshow(np.clip(_mask_aux, a_min=0.0, a_max=1.0))
		# plt.show()

	rectangle_angles = np.array(rectangle_angles)
	rectangle_masks = np.stack(rectangle_masks, axis=0)

	for idx, anchor in enumerate(anchor_angles[:-1]):
		angle_dists = rectangle_angles - anchor
		ori_gt[idx] = merge_masks(rectangle_masks, np.logical_and(0 < angle_dists, angle_dists < bin_size))
	
	return ori_gt, np.zeros_like(ori_gt)

	# margin_h = int(margin * height) // 2
	# margin_w = int(margin * width) // 2
	# margin_mask = np.zeros_like(ori_gt, dtype=np.float32)
	# margin_mask[..., margin_h:-margin_h, margin_w:-margin_w] = 1.0
	# ori_gt *= margin_mask

	# max_idx = np.argmax(np.reshape(ori_gt, (n_angles-1, -1)).sum(1))
	# ori_weights[max_idx, margin_h:-margin_h, margin_w:-margin_w] = 1.0
	# prev_idx = max_idx - 1
	# next_idx = (max_idx + 1) % (n_angles - 1)
	# ignore = np.clip(ori_gt[prev_idx] + ori_gt[next_idx], a_min=None, a_max=1.0)
	# ignore = np.clip(ignore - ori_gt[max_idx], a_min=0.0, a_max=None)
	# ori_weights[max_idx] -= ignore

	# return ori_gt, ori_weights


def getVectorMapsAngles_v2(shape, keypoints, theta=5, bin_size=10, filter_gt=False, margin=0.1):
	"""
	Convert Road keypoints obtained from road mask to orientation angle mask.
	Reference: Section 3.1
		https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf

	@param shape: Road Label/PIL image shape i.e. H x W
	@param keypoints: road keypoints generated from Road mask using
						function getKeypoints()
	@param theta: thickness width for orientation vectors, it is similar to
					thicknes of road width with which mask is generated.
	@param bin_size: Bin size to quantize the Orientation angles.

	@return: Retun ndarray of shape H x W, containing orientation angles per pixel.
	"""

	height, width = shape
	anchor_angles = np.arange(0.0, 180.0 + bin_size, bin_size)
	n_angles = len(anchor_angles)
	ori_gt = np.zeros((n_angles,) + shape, dtype=np.float32)
	if len(keypoints) < 2:
		ori_gt = ori_gt[:-1]
		ori_weights = np.ones_like(ori_gt)
		return ori_gt, ori_weights

	rectangle_angles = []
	rectangle_masks = []
	for j in range(len(keypoints)):
		for i in range(1, len(keypoints[j])):

			a = keypoints[j][i - 1]
			b = keypoints[j][i]
			ax, ay = a[0], a[1]
			bx, by = b[0], b[1]
			bax = bx - ax
			bay = by - ay
			norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
			bax /= norm
			bay /= norm

			min_w = max(int(round(min(ax, bx) - theta)), 0)
			max_w = min(int(round(max(ax, bx) + theta)), width)
			min_h = max(int(round(min(ay, by) - theta)), 0)
			max_h = min(int(round(max(ay, by) + theta)), height)

			_theta = math.degrees(math.atan2(bay, bax))
			_theta = (_theta -90) % 180
			rectangle_angles.append(_theta)

			_mask = np.zeros(shape, dtype=np.float32)
			for h in range(min_h, max_h):
				for w in range(min_w, max_w):
					px = w - ax
					py = h - ay
					dis = abs(bax * py - bay * px)
					if dis <= theta:
						_mask[h, w] = 1.0
			rectangle_masks.append(_mask)

	rectangle_angles = np.array(rectangle_angles)
	rectangle_masks = np.stack(rectangle_masks, axis=0)
	max_angle_dist = bin_size * 0.65
	for idx, anchor in enumerate(anchor_angles):
		angle_dists = np.abs(rectangle_angles - anchor)
		ori_gt[idx] = merge_masks(rectangle_masks, angle_dists < max_angle_dist)
	ori_gt[0] = np.clip(ori_gt[0] + ori_gt[-1], a_min=None, a_max=1.0)
	ori_gt = ori_gt[:-1]
	margin_h = int(margin * height) // 2
	margin_w = int(margin * width) // 2
	ori_weights = np.zeros_like(ori_gt)
	ori_weights[..., margin_h:-margin_h, margin_w:-margin_w] = 1.0
	ori_gt *= ori_weights
	
	if filter_gt:
		hist = np.reshape(ori_gt,(n_angles-1,-1)).sum(1)
		max_idx = np.argmax(hist)
		ori_weights = np.zeros_like(ori_gt)
		ori_weights[max_idx, margin_h:-margin_h, margin_w:-margin_w] = 1.0

	# plt.figure()
	# plt.imshow(ori_gt.sum(0) > 0.0)
	# for idx, angle in enumerate(anchor_angles[:-1]):
	# 	if np.any(ori_weights[idx] > 0.0):
	# 		plt.figure()
	# 		plt.imshow(ori_gt[idx])
	# 		plt.title("angle:{}".format(angle))
	# plt.show()

	return ori_gt, ori_weights


def convertAngles2VecMap(shape, vecmapAngles):
	"""
	Helper method to convert Orientation angles mask to Orientation vectors.

	@params shape: Road mask shape i.e. H x W
	@params vecmapAngles: Orientation agles mask of shape H x W
	@param bin_size: Bin size to quantize the Orientation angles.

	@return: ndarray of shape H x W x 2, containing x and y values of vector
	"""

	h, w = shape
	vecmap = np.zeros((h, w, 2), dtype=np.float)

	for h1 in range(h):
		for w1 in range(w):
			angle = vecmapAngles[h1, w1]
			if angle < 36.0:
				angle *= 10.0
				if angle >= 180.0:
					angle -= 360.0
				vecmap[h1, w1, 0] = math.cos(math.radians(angle))
				vecmap[h1, w1, 1] = math.sin(math.radians(angle))

	return vecmap


def convertVecMap2Angles(shape, vecmap, bin_size=10):
	"""
	Helper method to convert Orientation vectors to Orientation angles.

	@params shape: Road mask shape i.e. H x W
	@params vecmap: Orientation vectors of shape H x W x 2

	@return: ndarray of shape H x W, containing orientation angles per pixel.
	"""

	im_h, im_w = shape
	angles = np.zeros((im_h, im_w), dtype=np.float)
	angles.fill(360)

	for h in range(im_h):
		for w in range(im_w):
			x = vecmap[h, w, 0]
			y = vecmap[h, w, 1]
			angles[h, w] = (math.degrees(math.atan2(y, x)) + 360) % 360

	angles = (angles / bin_size).astype(int)
	return angles
