# import pcapy  # 删除pcapy
import numpy as np
from collections import deque
import sys
import warnings
from Pkt_filter import packet_parse,packet_filter
from scipy.sparse import csr_matrix
import scapy.all as scapy  # 使用scapy替代
from model import FFT, CNN, GateKeeper
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

model = FFT.HyLIDS()
# 加载模型权重
model.load_state_dict(torch.load("/home/jie/program/cic-iot/data/hylids_model.pt"))
model.eval()  # 设置为评估模式

#####CNN
# config = CNN.config
# model = CNN.CNNBiLSTMAttentionModel(vocab_size=256, embed_dim=config['d_dim'], num_classes=config['num_class'], lstm_hidden_dim=config['hidden_size'], lstm_layers=1, cnn_out_channels=128, kernel_sizes=[3, 4, 5], dropout=0.5, embedding_weights=None) 
# model.load_state_dict(torch.load("/home/jie/program/Deploy/model/CNN.pt"))
# model.eval()  # 设置为评估模式

model = GateKeeper.GateKeeper()
# 加载模型权重
model.load_state_dict(torch.load("/home/jie/program/Deploy/model/GateKeeper.pt"))
model.eval()  # 设置为评估模式

   
def bytes_to_numpy_array(byte_stream, max_length=50):
    arr = np.frombuffer(byte_stream, dtype=np.uint8)[0: max_length] 
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    #arr = csr_matrix(arr)
    return arr


def main():
    total_bytes = 0
    count = 0
    
    f = open("FFT_th.csv",'a')
    f.write("PT,IT\n")
    with torch.no_grad():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print("\n开始抓包分析，按 Ctrl+C 停止程序...")
        start_time = time.time()  # 记录开始时间
        def packet_callback(packet):
            nonlocal total_bytes, start_time, count
            # 检查是否超过3分钟
            if time.time() - start_time >= 300:  # 180秒 = 3分钟
                TH = total_bytes/(time.time() - start_time)
                print(f"\n运行时间达到300秒，总吞吐量: {TH * 8 / 10E6*300 } Mbps")
                print(f"\n运行时间达到300秒，总吞吐量: {count / 300 } pps")
                sys.exit(0)  # 正常退出程序
                
            try:
                parse_start_time = time.time()
                raw_packet = bytes(packet)
                keep, ip_packet = packet_parse(raw_packet)
                Pkt_in = np.copy(bytes_to_numpy_array(bytes(ip_packet)))
                parse_end_time = time.time()
                pkt_process_times = parse_end_time - parse_start_time
                
                predict_start_time = time.time()
                input_tensor = torch.LongTensor(Pkt_in).unsqueeze(0)
                outputs = model(input_tensor)
                predict_end_time = time.time()
                pkt_predict_times = predict_end_time - predict_start_time
                
                total_bytes += len(raw_packet)
                count += 1
                    
                f.write(f"{pkt_process_times:.4f},{pkt_predict_times:.4f}\n")
            except:
                total_bytes += len(bytes(packet))
                #print(len(bytes(packet)))
                pass
        
        try:
            packets = scapy.sniff(iface="wlan0", prn=packet_callback)
            count += 1
        except:
            pass

if __name__ == "__main__":
    main()