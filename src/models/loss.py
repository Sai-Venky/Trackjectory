import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from utils.utils import _tranpose_and_gather_feat, _sigmoid
from numpy import linalg as LA

# Siamese

def rpn_cross_entropy(input, target):
    mask_ignore = target == -1
    mask_calcu =  ~mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu], size_average=False)
    return loss


def rpn_smoothL1(input, target, label):
    pos_index = np.where(label.cpu() == 1)
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], size_average=False)
    return loss

# MOT

def _neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask
        
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)


class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.opt = opt
        self.emb_dim = 128
        self.nID = 2
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0

        output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])

        hm_loss += self.crit(output['hm'], batch['hm'])
        if opt.wh_weight > 0:
            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'],
                batch['ind'], batch['wh'])

        if opt.off_weight > 0:
            off_loss += self.crit_wh(output['reg'], batch['reg_mask'],
                                      batch['ind'], batch['reg'])

        if opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
            id_head = id_head[batch['reg_mask'] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][batch['reg_mask'] > 0]

            id_output = self.classifier(id_head).contiguous()
            id_loss += self.IDLoss(id_output, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats

# Trajectory

def bivariate_loss(V_pred, V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result
   