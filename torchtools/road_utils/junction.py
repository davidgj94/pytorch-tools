import numpy as np
import matplotlib.pyplot as plt
from mahotas.morph import hitmiss as hit_or_miss
from skimage.morphology import medial_axis as skeletonize
from matplotlib.patches import Circle
from sklearn.cluster import MeanShift
import pdb


def compute_patterns():
	X=[]
	#cross X
	X0 = np.array([[0, 1, 0],
				   [1, 1, 1],
				   [0, 1, 0]])
	X1 = np.array([[1, 0, 1],
				   [0, 1, 0],
				   [1, 0, 1]])
	X.append(X0)
	X.append(X1)
	#T like
	T=[]
	#T0 contains X0
	T0=np.array([[2, 1, 2],
				 [1, 1, 1],
				 [2, 2, 2]])

	T1=np.array([[1, 2, 1],
				 [2, 1, 2],
				 [1, 2, 2]])  # contains X1

	T2=np.array([[2, 1, 2],
				 [1, 1, 2],
				 [2, 1, 2]])

	T3=np.array([[1, 2, 2],
				 [2, 1, 2],
				 [1, 2, 1]])

	T4=np.array([[2, 2, 2],
				 [1, 1, 1],
				 [2, 1, 2]])

	T5=np.array([[2, 2, 1],
				 [2, 1, 2],
				 [1, 2, 1]])

	T6=np.array([[2, 1, 2],
				 [2, 1, 1],
				 [2, 1, 2]])

	T7=np.array([[1, 2, 1],
				 [2, 1, 2],
				 [2, 2, 1]])
	T.append(T0)
	T.append(T1)
	T.append(T2)
	T.append(T3)
	T.append(T4)
	T.append(T5)
	T.append(T6)
	T.append(T7)
	#Y like
	Y=[]
	Y0=np.array([[1, 0, 1],
				 [0, 1, 0],
				 [2, 1, 2]])

	Y1=np.array([[0, 1, 0],
				 [1, 1, 2],
				 [0, 2, 1]])

	Y2=np.array([[1, 0, 2],
				 [0, 1, 1],
				 [1, 0, 2]])

	Y2=np.array([[1, 0, 2],
				 [0, 1, 1],
				 [1, 0, 2]])

	Y3=np.array([[0, 2, 1],
				 [1, 1, 2],
				 [0, 1, 0]])

	Y4=np.array([[2, 1, 2],
				 [0, 1, 0],
				 [1, 0, 1]])
	Y5=np.rot90(Y3)
	Y6 = np.rot90(Y4)
	Y7 = np.rot90(Y5)
	Y.append(Y0)
	Y.append(Y1)
	Y.append(Y2)
	Y.append(Y3)
	Y.append(Y4)
	Y.append(Y5)
	Y.append(Y6)
	Y.append(Y7)

	return X, T, Y

X, Y, T = compute_patterns()

def find_branch_points(skel):

	bp = np.zeros(skel.shape, dtype=int)
	for x in X:
		bp = bp + hit_or_miss(skel,x)
	for y in Y:
		bp = bp + hit_or_miss(skel,y)
	for t in T:
		bp = bp + hit_or_miss(skel,t)

	return bp


def extract_coords(mask, normalize=False):
	y, x = np.where(mask)
	coords = np.array(list(zip(x, y)))
	if normalize:
		H, W = mask.shape
		coords = 2 * (coords / np.array([(W-1), (H-1)])) - 1.0
	return coords


def test_branch_points(label):
	skel = skeletonize(label)
	bp = find_branch_points(skel)
	coords = extract_coords(bp)
	
	_, ax = plt.subplots(1)
	ax.imshow(label)
	for idx in np.arange(len(coords)):
		ax.add_patch(Circle(coords[idx], 20))


def compute_junction_gt(label, sigma=50, tresh=0.5):

	test_branch_points(label)

	H, W = label.shape
	sigma /= np.sqrt(H * W)

	def cluster_coords(coords,):
		bandwidth = sigma / 3
		ms = MeanShift(bandwidth=bandwidth)
		ms.fit(coords)
		cluster_labels = ms.labels_
		cluster_centers = ms.cluster_centers_
		tresholds = []
		new_coords = []
		for label_idx in np.unique(cluster_labels).tolist():
			_coords = coords[cluster_labels == label_idx]
			new_coords.append(_coords)
			num_coords = len(_coords)
			tresholds += [tresh / num_coords]*num_coords

			# if n_coords == 1:
			# 	tresholds.append(tresh)
			# else:

				# _cluster_center = cluster_centers[label_idx]
				# diff = _coords - _cluster_center
				# new_sigma = np.sqrt((diff ** 2).sum(1)).mean()
				# sigmas += [new_sigma]*n_coords
		return tresholds, np.vstack(new_coords)

	skel = skeletonize(label)
	bp = find_branch_points(skel)
	if np.all(bp == 0):
		return np.zeros_like(label, dtype=np.float32), np.zeros_like(label, dtype=np.float32)

	coords = extract_coords(bp, normalize=True)
	junction_gt = np.zeros_like(label, dtype=np.float32)
	X, Y = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(-1.0, 1.0, H))
	tresholds, coords = cluster_coords(coords)

	for idx, _sigma in enumerate(sigmas):
		mu_x, mu_y = coords[idx].tolist()
		G = np.exp(-1.0 * ((X-mu_x)** 2 / (2 * _sigma**2) + (Y-mu_y)**2 / (2 * _sigma**2)) ) / (2 * np.pi * _sigma**2)
		G /= G.max()
		G[G<tresh] = 0.0
		junction_gt += G
	junction_gt = np.clip(junction_gt, a_min=0.0, a_max=1.0)

	plt.figure()
	plt.imshow(junction_gt)
	plt.show()
	
	return junction_gt, np.ones_like(junction_gt, dtype=np.float32)
	

