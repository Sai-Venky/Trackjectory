import _init_paths

import os

import json
import time

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import transforms as T

from utils.config import opt as options
from dataset.mot_dataset import MotDataset
from models.loss import MotLoss
from utils.utils import AverageMeter, load_model, save_model
from models.pose_dla_dcn import get_pose_net as get_dla_dcn

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.optimizer.add_param_group({'params': self.loss.parameters()})

  def set_device(self, gpus, chunk_sizes, device):
    
    self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      print('Loss for Iteration ', iter_id, 'is ', loss)
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      del output, loss, loss_stats, batch
    
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    return ret, results

  def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)

def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    model = get_dla_dcn(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model

def main(**kwargs):
    options._parse(kwargs)
    opt = options
    torch.manual_seed(317)

    print('Setting up data...')

    transforms = T.Compose([T.ToTensor()])
    dataset = MotDataset(opt, (640, 480), augment=True, transforms=transforms)
    opt = opt.update_dataset_info_and_set_heads(opt, dataset)
    print(opt)


    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    opt.device = torch.device('cuda' if opt.gpus >= '0' else 'cpu')

    print('Creating model...')
    model = create_model('dla_34', opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    trainer = BaseTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, -1, opt.device)

    if opt.load_model != '':
      model, optimizer = load_model(model, opt.load_model, trainer.optimizer)

    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        
        log_dict_train, _ = trainer.train(epoch, train_loader)
        if epoch % opt.save_every == 0:
            lr = opt.lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)


if __name__ == '__main__':
    main()
