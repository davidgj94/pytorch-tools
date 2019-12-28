import matplotlib; matplotlib.use('tkagg')
import random
import numpy as np
import torch
from torch.utils import data
import os.path
import os
import pdb
from torchvision import transforms
from skimage.io import imread
import torchvision.transforms.functional as TF
from torchtools.augmentation import Compose
from .register import register
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import torchtools.lines as lines
import pickle
import torchtools.vis as vis
from scipy import ndimage
import pickle
from pathlib import Path


def string2msec(time_string):
    time_min = int(time_string.split(':')[0])
    time_sec = int(time_string.split(':')[1])
    time_sec += time_min * 60
    time_msec = 1000 * time_sec
    return time_msec


def msec2string(time_msec):
    time_sec = time_msec // 1000
    time_min = time_sec // 60
    time_string = "{}:{:02d}".format(time_min, time_sec - time_min * 60)
    return time_string

def downscale(img, max_dim):

    height, width = img.shape[:2]

    if max_dim < height or max_dim < width:
        scaling_factor = min(max_dim / float(width), max_dim / float(height))
        img_down = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return img_down
    else:
        return None

def undistort(img, dist, mtx):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst

class VideoLoader:

    def __init__(self, video_path, camera_name, max_dim=1000):

        self.vidcap = cv2.VideoCapture(video_path)
        camera_dir = os.path.join("calib_data", camera_name)
        self.dist = np.load(os.path.join(camera_dir, 'dist.npy'))
        self.mtx = np.load(os.path.join(camera_dir, 'mtx.npy'))
        self.max_dim = max_dim

    def frame_at(self, time_msec):

        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
        success, frame = self.vidcap.read()
        if success:
            frame = self.process_frame(frame)
            return frame
        else:
            return None

    def process_frame(self, frame):
        frame = undistort(frame, self.dist, self.mtx)
        frame = downscale(frame, self.max_dim)
        return frame


@register.attach('video_dataset')
class VideoDataset(data.Dataset):

    def __init__(self, video_path, camera_name, start_time, end_time, fps=10.0):
        self.video_loader = VideoLoader(video_path, camera_name)
        start_time_msec = string2msec(start_time)
        end_time_msec = string2msec(end_time)
        step_msec = int(1000 / fps)
        self.time_msec_array = np.arange(start_time_msec, end_time_msec + step_msec, step_msec)
        self.mean = [0.485, 0.456, 0.406]
        self.var = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.time_msec_array)

    def __getitem__(self, index):

        time_msec = self.time_msec_array[index]
        frame = self.video_loader.frame_at(time_msec)
        image = TF.to_tensor(frame[...,::-1].copy())
        image = TF.normalize(image, self.mean, self.var)
        image = image.numpy()
        return image, frame.astype(np.int64)