import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import wandb


myseed = 1
np.random.seed(myseed)
torch.manual_seed(myseed)

tr_path = val_path = test_path = "/home/jie/program/cic-iot/data/"

config = {
    'n_epochs': 2,
    'batch_size': 128,
    'flow_len': 50,
    'd_dim': 36,
    'hidden_size': 50, 
    'save_name': 'HyL-{epoch:02d}-{val_acc:.2f}',
    'log_path1': 'HyL_logs', 
    'log_path2': 'Mix_advTrain', 
    'num_class': 6 
}
isIat = False 



if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    def __init__(self, path, mode):
        self.mode = mode
       
        if 'csv' in path:
            with open(path, 'r') as fp:
                data = list(csv.reader(fp))
                data = np.array(data[:])[:, 0:].astype(float)
                
                feats = list(range(256))
                label = data[:, -1]
                data = data[:, feats]
        else:  # npy
            data_label = np.load(path + mode + '.npy')
            # label = np.load(path + mode + '.npy')
            if data_label.shape[0] < config['flow_len']:
                data = data_label[:, :10]
                pad_width = ((0, 0), (0, config['flow_len'] - 10))  
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
                label = data_label[:, -1]
            else:
                data = data_label[:, :config['flow_len']]
                label = data_label[:, -1]
        
        
        self.data = torch.LongTensor(data)
        self.label = torch.LongTensor(label.astype(np.uint8))

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def prep_dataloader(path, mode, batch_size, njobs=12):
    dataset = MyDataset(path, mode)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=njobs,
                            pin_memory=True)
    return dataloader

class HyLIDS(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Embedding layers
        self.byte_embedding = nn.Embedding(256, config['d_dim'])
    
        # Normalization and regularization
        # Classification layers
        
        
        self.classifier1 = nn.Sequential(
            # test avg 
            nn.LayerNorm(config['d_dim']),  
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(config['flow_len'], config['hidden_size']),
            nn.ReLU()
        )
        
        
       
        
        self.classifier2 = nn.Sequential(
            # test avg 
            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(config['flow_len'], config['hidden_size']),
            nn.ReLU()
        )
        
        self.out = nn.Linear(config['hidden_size']*2, config['num_class'])
        

    def forward(self, x):
        # Embedding
        embedded = self.byte_embedding(x)
        spatial_avg = self.classifier1(embedded)
        
        fft = torch.fft.fft(torch.fft.fft(embedded, dim=-1), dim=-2)
        fft = torch.abs(fft)#[:, :, :config['d_dim']//2 + 1]

        #fft = F.normalize(fft, p=2, dim=-1)
        fft = self.classifier2(fft)

        out = torch.cat((spatial_avg, fft), dim=-1)
        out = self.out(out)
        return out
    
    
