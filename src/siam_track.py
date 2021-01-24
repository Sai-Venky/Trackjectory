import argparse
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import os 
from tracker.singletracker import SiamRPNPP, SiamRPN_init, SiamRPN_track
from utils.config import opt as options
from utils.utils  import load_net
from dataset.util  import get_axis_aligned_bbox, cxy_wh_2_rect
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def eval_seq(opt, save_dir='/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/output'):
    # load net
    net = SiamRPNPP()
    load_net(net, opt.single_track_load_model)
    net.eval()#.cuda()

    # image and init box
    image_files = sorted(glob.glob('/Users/ecom-v.ramesh/Desktop/kabadi/frames/test/*.png'))
    init_rbox = [698, 141, 876, 141, 698, 554, 876, 554]
    [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

    # tracker init
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im = cv2.imread(image_files[0])  # HxWxC
    state = SiamRPN_init(im, target_pos, target_sz, net)

    # tracking and visualization
    toc = 0
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        # print(im.shape)
        tic = cv2.getTickCount()
        state = SiamRPN_track(state, im)  # track
        toc += cv2.getTickCount()-tic
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        # print(res)
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        cv2.imshow('SiamRPN', im)
        cv2.waitKey(1)

    print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = options
    opt.device = torch.device('cuda' if opt.gpus >= '0' else 'cpu')
    eval_seq(opt)