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
config = {
    'n_epochs': 5,
    'batch_size': 128,
    'flow_len': 50,
    'd_dim': 36,
    'hidden_size': 50, 
    'save_name': 'HyL-{epoch:02d}-{val_acc:.2f}',
    'log_path1': 'HyL_logs',  # 修改
    'log_path2': 'Mix_advTrain',  # 修改
    'num_class': 6 # 修改
}
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