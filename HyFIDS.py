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
    'log_path1': 'HyL_logs',  # 修改
    'log_path2': 'Mix_advTrain',  # 修改
    'num_class': 6 # 修改
}
isIat = False  # 修改



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
                pad_width = ((0, 0), (0, config['flow_len'] - 10))  # 在第二个维度上填充40列0
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
                label = data_label[:, -1]
            else:
                data = data_label[:, :config['flow_len']]
                label = data_label[:, -1]
        
        
        self.data = torch.LongTensor(data)
        self.label = torch.LongTensor(label.astype(np.uint8))

    def __len__(self):
        return len(self.data)  # 返回数据的总个数

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
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) #weight_decay=0.01)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=1e-6)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}
       

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log('training_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        loss = {'loss': loss}
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        entropy = F.cross_entropy(preds, y)
        self.log('val_loss', entropy, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_pre = torch.argmax(F.log_softmax(preds, dim=1), dim=1)
        
        # 获取概率值用于ROC曲线
        probs = F.softmax(preds, dim=1)[:, 1].cpu().numpy()
        y_true = y.cpu().numpy()
        y_pred = y_pre.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        

        # 记录所有指标
        self.log('test_acc', acc)
        self.log('test_pre', pre)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
    

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=config['save_name'],
    save_top_k=1,
    mode='min',
    save_last=True
)


    
if __name__ == '__main__':
    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], njobs=6)
    val_set = prep_dataloader(val_path, 'val', config['batch_size'], njobs=6)
    test_set = prep_dataloader(test_path, 'test', config['batch_size'], njobs=1)

    model =  HyLIDS().to(device)
    
    wandb_logger = WandbLogger(project="FFT-model-analysis", name="FFT-timing", log_model=True)

    wandb_logger.experiment.config.update({
        "model_type": "HyLIDS",
        "dataset": "cic-iot",
        "epochs": config['n_epochs'],
        "batch_size": config['batch_size'],
        "flow_length": config['flow_len'],
        "embedding_dim": config['d_dim'],
        "hidden_size": config['hidden_size'],
        "num_classes": config['num_class'],
        "device": device
    })
    
    trainer = Trainer(
    val_check_interval=1.0,
    max_epochs=config['n_epochs'],
    accelerator='cpu',      # 👈 使用 CPU
    logger=wandb_logger,
    deterministic=True,     # 禁用非确定性优化
    callbacks=[
        checkpoint_callback
    ])
    wandb_logger.watch(model, log="all")

    start_time = time.time()
    trainer.fit(model, train_dataloaders=tr_set, val_dataloaders=val_set)
    # trainer.fit(model, train_dataloaders=tr_set, ckpt_path=r'D:\project\FS-NET_my\model\pktl_v2ray_all.ckpt')
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练总时间: {training_time:.2f} 秒")
    print(f"每轮平均时间: {training_time / config['n_epochs']:.2f} 秒")
    
    trainer.test(model, dataloaders=test_set)
    # 保存模型为pt格式
    torch.save(model.state_dict(), "/home/jie/program/cic-iot/data/hylids_model.pt")
    print("模型已保存为pt格式：/home/jie/program/cic-iot/data/hylids_model.pt")
    # # save model
    #trainer.save_checkpoint("/home/jie/Program/AdvTrafFeat/SimuModel/saved_dict/sub_VPN.ckpt")
