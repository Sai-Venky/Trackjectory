from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import numpy as np
import torch

from dataset.mot_dataset import LoadImages
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.timer import Timer
from utils.config import opt as options


"""
    eval_seq is used for running the JDE tracker.
    It also computes the results needed for training/testing GCN for trajectory forcasting.
"""
def eval_seq(opt, save_dir='', frame_rate=30):
    
    dataloader = LoadImages(opt.test_images_dataset)
    opt = opt.update_dataset_info_and_set_heads(opt, dataloader)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    fd = os.open(save_dir + '/args.txt', os.O_RDWR)
    for i, (path, img, img0) in enumerate(dataloader):

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            centre = tlwh[:2] + tlwh[2:]/2
            text = str(frame_id) + ' ' + str(tid) + ' ' + str(centre[1]) + ' '+ str(centre[0]) + '\n'
            os.write(fd, str.encode(text))
        timer.toc()
        # save results

        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                        fps=1. / timer.average_time)
        cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    os.close(fd)

    return frame_id, timer.average_time, timer.calls


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = options
    opt.device = torch.device('cuda' if opt.gpus >= '0' else 'cpu')
    eval_seq(opt, opt.save_dir)
