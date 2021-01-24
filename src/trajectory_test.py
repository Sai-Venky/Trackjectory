import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.module import Module
import torch.optim as optim
from torch.utils.data import DataLoader
import networkx as nx
import pickle
import torch.distributions.multivariate_normal as torchdist
import copy
import cv2 

from models.gcn import *
from dataset.trajectory_dataset import TrajectoryDataset
from utils.config import opt as options

#Data prep    
def main(**kwargs):
    options._parse(kwargs)        
    args = options
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    dataset = TrajectoryDataset(
            args.trajectory_dataset,
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            skip=1,norm_lap_matr=True)

    args.loader_test = DataLoader(
            dataset,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =True,
            num_workers=0)

    #Defining the model 
    model = social_stgcnn(n_stgcnn =args.n_stgcnn, n_txpcnn=args.n_txpcnn,
    output_feat=5, seq_len=args.obs_seq_len,
    kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
    model.load_state_dict(torch.load('./checkpoint/best.pth'))

    # checkpoint_dir = './checkpoint/'
    # with open(args_path,'rb') as f: 
    #     args = pickle.load(f)


    print('Data and model loaded')

    #Training 
    args.metrics = {'train_loss':[],  'val_loss':[]}
    print('Testing started ...')
    for epoch in range(args.trajectory_num_epochs):
        test(model, epoch, args)

        print('Epoch:', epoch)
        


def test(model, epoch, args, KSTEPS=20):
    model.eval()
    loader_test = args.loader_test
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 
    for batch in loader_test: 
        step+=1
        #Get data
        # batch = [tensor.cuda() for tensor in batch]
        batch = [tensor for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask, V_obs, A_obs, V_tr, A_tr, frame_id = batch


        num_of_objs = obs_traj_rel.shape[1]

        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())

        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]

        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr

        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2)#.cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        mvnormal = torchdist.MultivariateNormal(mean,cov)

        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                    V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                    V_x[-1,:,:].copy())

        frame_id = '{0:05d}'.format(int(frame_id) + 2014)
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        image = cv2.imread('/Users/ecom-v.ramesh/Desktop/kabadi/frames/frames2/' + frame_id + '.png')
        for k in range(20):

            V_pred = mvnormal.sample()
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                        V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

        image = vis(raw_data_dict[step]['obs'], raw_data_dict[step]['pred'], raw_data_dict[step]['trgt'], image)
        cv2.imwrite('/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/output2/'+ frame_id + '.png', image)

def vis(obs, pred, tgt, image):

    colors = (255, 0, 0)
    pred = np.array(pred)
    pred = pred.transpose(2, 0, 1, 3)
    image = vis_change(pred, image)
        # for j in range(len(k)):
        #     image = cv2.circle(image, (int(k[j][1]), int(k[j][0])), radius=2, color= colors, thickness=2)
    
    # colors = (0, 0, 255)
    # for i in range(len(obs)):
    #     for j in range(len(obs[i])):
    #         image = cv2.circle(image, (int(obs[i][j][1]), int(obs[i][j][0])), radius=2, color= colors, thickness=2)
 
    # colors = (0, 255, 0)
    # for i in range(len(tgt)):
    #     for j in range(len(tgt[i])):
    #         image = cv2.circle(image, (int(tgt[i][j][1]), int(tgt[i][j][0])), radius=2, color= colors, thickness=2)

    return image

# traj is a list of xy tuple
def plot_traj(img, traj, color):
  """Plot a trajectory on image."""
  traj = np.array(traj, dtype="float32")
  points = zip(traj[:-1], traj[1:])

  for p1, p2 in points:
    p1 = p1.T
    img = cv2.line(img, tuple(p1[::-1]), tuple(p2[::-1]), color=color, thickness=2)

  return img

def vis_change(trajs_indexeds, image):
    h,w = image.shape[:2]
    num_between_line = 40
    max_gt_pred_length = 70
    frame_data = image
    new_layer = np.zeros((h, w), dtype="float")
    for i in range(len(trajs_indexeds)):
        trajs_indexed = trajs_indexeds[i]
        trajs_indexed = trajs_indexed.reshape(-1, 2)
        for (y1, x1), (y2, x2) in zip(trajs_indexed[:-1], trajs_indexed[1:]):
        # all x,y between
            xs = np.linspace(x1, x2, num=num_between_line, endpoint=True)
            ys = np.linspace(y1, y2, num=num_between_line, endpoint=True)
            points = zip(xs, ys)
            for x, y in points:
                x = int(x)
                y = int(y)
                new_layer[y, x] = 1.0
        
        from scipy.ndimage import gaussian_filter
        f_new_layer = gaussian_filter(new_layer, sigma=10)
        f_new_layer = np.uint8(f_new_layer*255)
        ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)

        heatmap_img = cv2.applyColorMap(f_new_layer, cv2.COLORMAP_AUTUMN)
        heatmap_img_masked = cv2.bitwise_and(heatmap_img,heatmap_img, mask=mask)
        frame_data = cv2.addWeighted(frame_data, 1.0, heatmap_img_masked, 1.0, 0)
        # plot the predicted trajectory
        # for y, x in trajs_indexed[:max_gt_pred_length]:
        #     frame_data = cv2.circle(frame_data, (int(x), int(y)), radius=5,
        #                             color=(255, 0, 0), thickness=1)
        # frame_data = plot_traj(frame_data, trajs_indexed[:max_gt_pred_length],
        #                         (255, 0, 0))
    return frame_data

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

if __name__ == '__main__':
    main()