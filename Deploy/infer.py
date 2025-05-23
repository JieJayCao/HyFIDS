from  time import perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Pkt_process import main
# 读取 ONNX 模型
from model import FFT

model = FFT.HyLIDS()
# 加载模型权重
model.load_state_dict(torch.load("/home/jie/program/cic-iot/data/hylids_model.pt"))
model.eval()  # 设置为评估模式


# 获取数据包输入
Pkt_in = main()
print("输入数据形状:", Pkt_in.shape)

# 数据准备
if Pkt_in is not None:
    # 转换为PyTorch张量
    input_tensor = torch.LongTensor(Pkt_in)
    
    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 获取预测类别和概率
        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # 输出结果
        print("预测类别统计:",predicted_classes)
        
        # for i in range(FFT.config['num_class']):
        #     count = (predicted_classes == i).sum().item()
        #     if count > 0:
        #         percentage = count / len(predicted_classes) * 100
        #         print(f"类别 {i}: {count} 个数据包 ({percentage:.2f}%)")
        
        # # 显示最可能的类别
        # most_common_class = predicted_classes.bincount().argmax().item()
        # print(f"\n流量最可能属于类别: {most_common_class}")
else:
    print("未捕获到数据包")
    



####################################
"""for i in tqdm(range(10000)):
   
    ort_output = session.run( None, {input_name: session_input})
    probabilities = softmax(ort_output)
    predicted_class = np.argmax(probabilities)
e = perf_counter()
print((e-s)/10000)
"""

"""
out = torch.randn([256,20,39])
s1 = perf_counter()
torch_fft = torch.fft.fft(torch.fft.fft(out, dim=-1), dim=-2).real
e1 = perf_counter()
print(e1-s1)
s2 = perf_counter()
np_fft = np.fft.fft(out, axis=-1)
np_fft = np.fft.fft(np_fft, axis=-2)
real_part = np.real(np_fft)
e2 = perf_counter()
print(e2-s2)"""