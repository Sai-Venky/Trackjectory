
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from dataset.util import *

class SiameseDataset(Dataset):
    def __init__(self, data, train=False):
        
        self.nSamples = None
        self.transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self.lines = []

        if train and (data is not None):
            data_sample = [data[i] for i in sorted(random.sample(range(len(data)), 5))]
            all_samples = data_sample
            
        self.nSamples = len(all_samples)

        for i in range(self.nSamples):
            track_obj = all_samples
            ran_f1 = random.randint(0, len(track_obj)-1)
            ran_f2 = random.randint(0, len(track_obj)-1)
            self.lines.append([track_obj[ran_f1], track_obj[ran_f2]])
        random.shuffle(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        
        pair_infos = self.lines[index]

        z, x, gt_box, regression_target, label = load_data_rpn(pair_infos)

        z = self.transform(z)
        x = self.transform(x)

        regression_target = torch.from_numpy(regression_target)
        label = torch.from_numpy(label)

        return z, x, regression_target, label