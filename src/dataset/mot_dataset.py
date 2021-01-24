import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from utils.config import opt
from dataset.util import gaussian_radius, draw_umich_gaussian


os.environ['KMP_DUPLICATE_LIB_OK']='True'
class LoadImages:  # for inference
    default_resolution = [640, 480]
    mean = None
    std = None
    num_classes = 1
    def __init__(self, path, img_size=(640, 480)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = (img_size[0]//32) * 32
        self.height = (img_size[1]//32) * 32
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class MotDataset:  # for training
    default_resolution = [640, 480]
    mean = None
    std = None
    num_classes = 1

    def __len__(self):
        return self.nF  # number of batches

    def __init__(self, opt, img_size=(640, 480), augment=False, transforms=None):
        self.opt = opt

        self.num_classes = 1
        self.fr = 1
        self.img_files = sorted(glob.glob('%s/*/*/*.png' % opt.images_dataset))
        self.label_files = sorted(glob.glob('%s/*/*/*.txt' % opt.labels_dataset))

        self.nID = int(1 + 1) # fot no
        self.nF = len(self.label_files)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        index = files_index
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path)

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio

        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        
        wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[1:]
            cls_id = int(label[0])

            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h

            # x, y , w, h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            # x1, y1, x2, y2
            # bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            # bbox[1] = np.clip(bbox[1], 0, output_h - 1)

            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]
            # x1, y1, x2, y2

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                        bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]

                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = 0
                bbox_xys[k] = bbox_xy

        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        return ret
    
    def xyxy2xywh(self, x):
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y
    
    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
            labels = labels0.copy()
            labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
            labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
            labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
            labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh

        img, labels, M = random_affine(img, self.fr, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
        self.fr +=1
        img00 = img
        if self.transforms is not None:
            img = self.transforms(img)
        nL = len(labels)

        if nL > 0:
            labels[:, 1:5] = self.xyxy2xywh(labels[:, 1:5].copy())
            labels[:, 1] /= width
            labels[:, 2] /= height
            labels[:, 3] /= width
            labels[:, 4] /= height
        return img, labels, img_path, (h, w)


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def random_affine(img, frame, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2), borderValue=(127.5, 127.5, 127.5)):

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]
    

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)

    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 0) & (h > 0) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]
            vis(imw, str(frame) + '.png', xy[:, 0], xy[:, 1], xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1])
            frame +=1
        return imw, targets, M
    else:
        return imw


def vis(img, nm, lowerx = None, lowery = None, bbox_width = None, bbox_height = None):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if lowerx:
        for i in range(lowerx.shape[0]):
            if i > 4:
                continue
            rect = patches.Rectangle((lowerx[i],lowery[i]),bbox_width[i],bbox_height[i],linewidth=1,edgecolor='y',facecolor='none')
            ax.add_patch(rect)
    plt.savefig(nm)