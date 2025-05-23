
from importlib import import_module
import time
import torch
import numpy as np
import torch.nn.functional as F
import onnx
print('ONNX 版本', onnx.__version__)
import onnxscript
print(onnxscript.__version__)
import onnxruntime
from FFT import HyLIDS

device = 'cpu'
model = HyLIDS().to(device)
model.load_state_dict(torch.load("/home/jie/program/HyL_logs/Mix_advTrain/version_3/checkpoints/HyL-epoch=04-val_acc=0.00.ckpt"),strict=False)
model.eval()

Pkt_in = torch.randint(low=0, high=256, size=(200, 50), dtype=torch.int).to(device)
Pkt_in = torch.LongTensor(Pkt_in).to(device)

#onnx_program = torch.onnx.dynamo_export(model,Pkt_in).save("Ton_model.onnx")
#onnx_program = torch.onnx.export(model,Pkt_in).save("Ton_model.onnx")
onnx_program = torch.onnx.export(
    model,
    Pkt_in,
    "FFT_cic_model.onnx",
    opset_version=20  # 尝试使用更高版本的opset
)

# 读取 ONNX 模型
onnx_model = onnx.load('Ton_model.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')

"""torch.onnx.export(model,               # 待导出的模型
                  Pkt_in,                # 模型的输入数据（张量或张量的元组）
                  "model.onnx",      # 导出的模型文件名
                  export_params=True,  # 是否导出模型参数
                  opset_version=17,    # ONNX操作集的版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],     # 输入数据的名称
                  output_names=['output'],   # 输出数据的名称
                  dynamic_axes={'input': {0: 'batch_size'},  # 动态轴的指定
                                'output': {0: 'batch_size'}})"""


                   
"""onnx_input = onnx_program.adapt_torch_inputs_to_onnx(Pkt_in)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")

ort_session = onnxruntime.InferenceSession("my_image_classifier.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)"""
