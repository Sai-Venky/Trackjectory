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

from models.gcn import *
from dataset.trajectory_dataset import TrajectoryDataset
from models.loss import bivariate_loss
from utils.config import opt as options

def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)

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

    args.loader_train = DataLoader(
            dataset,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =True,
            num_workers=0)

    #Defining the model 
    model = social_stgcnn(n_stgcnn =args.n_stgcnn, n_txpcnn=args.n_txpcnn,
    output_feat=5, seq_len=args.obs_seq_len,
    kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

    checkpoint_dir = './checkpoint/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    #Training settings 
    optimizer = optim.SGD(model.parameters(),lr = args.trajectory_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_sh_rate, gamma = 0.2)
         
    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    #Training 
    args.metrics = {'train_loss':[],  'val_loss':[]}
    print('Training started ...')
    torch.save(model.state_dict(), checkpoint_dir + 'bests.pth')
    for epoch in range(args.trajectory_num_epochs):
        train(model, optimizer, epoch, args)
        scheduler.step()

        print('Epoch:', epoch)
        for k,v in args.metrics.items():
            if len(v)>0:
                print(k,v[-1])
        
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(args.metrics, fp)
        torch.save(model.state_dict(), checkpoint_dir + 'best.pth')

def train(model, optimizer, epoch, args):
    metrics = args.metrics
    loader_train = args.loader_train
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len/args.trajectory_batch_size)*args.trajectory_batch_size + loader_len%args.trajectory_batch_size -1


    for cnt, batch in enumerate(loader_train): 
        batch_count+=1
        batch = [tensor for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask,V_obs,A_obs,V_tr,A_tr, fr = batch

        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp, A_obs.squeeze())
        
        V_pred = V_pred.permute(0,2,3,1)
    
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.trajectory_batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss/args.trajectory_batch_size
            is_fst_loss = True
            loss.backward()

            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch/batch_count)
                    
    metrics['train_loss'].append(loss_batch/batch_count)

if __name__ == '__main__':
    main()