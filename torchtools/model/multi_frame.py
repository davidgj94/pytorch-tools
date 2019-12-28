import collections
import cv2
from deeplabv3.lines import points2line_eq, create_grid_intersect, find_intesect_borders, normal_form, create_grid, general_form
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from .predict import _line_detection
import numpy as np
import pdb


class MultiFrameMerge():

	def __init__(self, nframes=3):
		self.nframes = nframes
		self.central_idx = self.nframes // 2
		self.buffer = collections.deque(maxlen=self.nframes)
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.bf = cv2.BFMatcher()

	def homography(self, sift_src, sift_dst):

		kp_a, des_a = sift_src
		kp_b, des_b = sift_dst

		matches = self.bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

		# Lowes Ratio
		good_matches = []
		for m, n in matches:
			if m.distance < .75 * n.distance:
				good_matches.append(m)

		src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

		if len(src_pts) > 4:
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
		else:
			M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

		return M

	def transform_point(self, point, M):

		p = np.array(point + (1,))
		p_prime = np.dot(M, p)

		y_t = p_prime[1] / p_prime[2]
		x_t = p_prime[0] / p_prime[2]

		return x_t, y_t

	# def transform_line(self, line_eq, M):

	# 	p = np.array(line_eq)
	# 	p_prime = np.dot(M, p)
	# 	line_eq_t = tuple(p_prime.tolist())
	# 	return line_eq_t

	def angle_diff(self, anchor_angle, angles):
		unit_vertor_anchor = np.array([np.cos(anchor_angle), np.sin(anchor_angle)]).reshape(-1,2)
		unit_vertors = np.hstack([np.cos(angles).reshape(-1,1), np.sin(angles).reshape(-1,1)])
		dot_product = np.dot(unit_vertor_anchor, unit_vertors.T)
		dot_product = np.clip(dot_product, a_max=1.0, a_min=-1.0)
		diff_angles = np.abs(np.arccos(dot_product)).squeeze()
		diff_angles = np.minimum(diff_angles, np.pi - diff_angles)
		return diff_angles

	def angle_mean(self, angles):
		return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

	def cluster_lines(self, _lines):

		if np.size(_lines) == 0:
			return []

		ms = MeanShift(bandwidth=50.0)
		pred = ms.fit_predict(_lines[:,0].reshape(-1,1))
		cluster_labels = np.unique(pred)

		cluster_lines = []
		for _label in cluster_labels.tolist():
			res = _lines[pred == _label]
			res = (res[:,0].mean(), self.angle_mean(res[:,1]))
			cluster_lines.append(res)

		return cluster_lines

	def vis_lines(self, _lines):

		sz = self.buffer[0][0].shape
		plt.figure()
		plt.imshow(create_grid(sz, _lines, width=10))



	def filter_lines(self, lines1, lines2):
			

		if len(lines2) == 0:
			# print("lines2 empty")
			return lines1
		else:

			lines2 = np.array(lines2)
			ori_lines2 = lines2[:,1]
			anchor_ori = ori_lines2[0]
			angle_diff = np.rad2deg(self.angle_diff(anchor_ori, ori_lines2))
			ori1 =  self.angle_mean(ori_lines2[angle_diff > 20])
			ori2 =  self.angle_mean(ori_lines2[angle_diff < 20])

			lines1 = np.array(lines1)
			angles = lines1[:,1]
			angles_diff_ori1 =  np.rad2deg(self.angle_diff(ori1, angles))
			angles_diff_ori2 =  np.rad2deg(self.angle_diff(ori2, angles))
			is_not_outlier = np.logical_or(angles_diff_ori1 < 3.0, angles_diff_ori2 < 3.0)
			self.vis_lines(lines1)
			lines1 = lines1[is_not_outlier]
			self.vis_lines(lines1)

			for _line in lines2:
				rhos = lines1[:,0]
				angles = lines1[:,1]
				rhos_diff = np.abs(_line[0] - rhos)
				angles_diff = np.rad2deg(self.angle_diff(_line[1], angles))
				keep = np.logical_and(rhos_diff < 75.0, angles_diff < 5.0)
				lines1 = lines1[np.invert(keep)]

			lines1 = lines1.tolist()
			self.vis_lines(lines1)

			if len(lines1) == 0:
				# print("lines1 empty")
				return lines2

			lines1 = np.array(lines1)
			ori_lines1 = lines1[:,1]

			angle_diff_ori1 = np.rad2deg(self.angle_diff(ori1, ori_lines1))
			angle_diff_ori1 = np.atleast_1d(angle_diff_ori1)
			lines1_1 = lines1[angle_diff_ori1 < 5]
			self.vis_lines(lines1_1)

			angle_diff_ori2 = np.rad2deg(self.angle_diff(ori2, ori_lines1))
			angle_diff_ori2 = np.atleast_1d(angle_diff_ori2)
			lines1_2 = lines1[angle_diff_ori2 < 5]
			self.vis_lines(lines1_2)

			cluster_lines1_1 = self.cluster_lines(lines1_1)
			cluster_lines1_2 = self.cluster_lines(lines1_2)
			self.vis_lines(cluster_lines1_1)
			self.vis_lines(cluster_lines1_1)
			plt.show()

			return cluster_lines1_1 + cluster_lines1_2 + lines2.tolist()


	def get_lines_homography(self,):

		for idx in np.arange(self.nframes).tolist():
			if idx != self.central_idx:
				M = self.homography(self.buffer[idx][2], self.buffer[self.central_idx][2])
				frame_lines = self.buffer[idx][1]
				yield (frame_lines, M)


	def merge_lines(self,):

		sz = self.buffer[0][0].shape

		lines1 = []
		for frame_lines, M in self.get_lines_homography():

			for _frame_line in frame_lines:

				line_points = find_intesect_borders(_frame_line, sz)
				line_point_0 = self.transform_point(line_points[0], M)
				line_point_1 = self.transform_point(line_points[1], M)
				line_eq = points2line_eq(line_point_0, line_point_1)

				if find_intesect_borders(line_eq, sz) is not None:
					lines1.append(normal_form(*line_eq))

		lines2  = [normal_form(*line_eq) for line_eq in self.buffer[self.central_idx][1]]
		return create_grid(sz, self.filter_lines(lines1, lines2), width=10)


	def __call__(self, _input, frame):

		pred_S, detected_lines = _line_detection(_input)
		detected_lines = [general_form(*line_coeff) for line_coeff in detected_lines]

		frame_bgr = frame.copy().astype(np.uint8)
		frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		sift_out = self.sift.detectAndCompute(frame, None)
		self.buffer.append((pred_S, detected_lines, sift_out, frame_bgr))

		if len(self.buffer) == self.nframes:
			grid = self.merge_lines()
			out = self.buffer[self.central_idx][0].copy()
			mask = self.buffer[self.central_idx][0] == 0
			out[mask] = grid[mask]
			return out, self.buffer[self.central_idx][-1]
		else:
			return None