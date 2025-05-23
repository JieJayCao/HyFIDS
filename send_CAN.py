import can
import time
import random
import threading

"""
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
"""

# CAN 消息配置
CAN_MESSAGES = {
    'ENGINE': {
        'id': 0x100,
        'interval': (0.01, 0.02),  # 10-20ms, 发动机转速、温度等高优先级信息
        'data_range': (2, 8),      # 数据长度范围
        'data_template': lambda: [
            random.randint(0, 255),  # 发动机转速高字节
            random.randint(0, 255),  # 发动机转速低字节
            random.randint(80, 120), # 发动机温度
            random.randint(0, 100),  # 节气门位置
            random.randint(0, 255),  # 其他发动机参数
        ]
    },
    'TRANSMISSION': {
        'id': 0x200,
        'interval': (0.05, 0.1),   # 50-100ms, 变速箱、车速等中优先级信息
        'data_range': (3, 6),
        'data_template': lambda: [
            random.randint(0, 6),    # 档位
            random.randint(0, 255),  # 车速高字节
            random.randint(0, 255),  # 车速低字节
        ]
    },
    'BODY': {
        'id': random.randint(0x300, 0x3FF),
        'interval': (0.2, 0.5),    # 200-500ms, 车身控制等低优先级信息
        'data_range': (2, 4),
        'data_template': lambda: [
            random.randint(0, 1),    # 车门状态
            random.randint(0, 100),  # 车内温度
            random.randint(0, 255),  # 其他车身信息
        ]
    }
}

# 初始化CAN总线（vcan0 为虚拟接口）
bus = can.interface.Bus(channel='vcan0', bustype='socketcan')

def send_frames(device_name, message_config):
    while True:
        data = message_config['data_template']()
        dlc = random.randint(*message_config['data_range'])
        data = data[:dlc]  # 截取指定长度的数据

        msg = can.Message(
            arbitration_id=message_config['id'],
            data=data,
            is_extended_id=False
        )

        try:
            bus.send(msg)
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] [{device_name}] Sent: ID={message_config['id']:03X}, Data={data}")
        except can.CanError:
            print(f"[{device_name}] Message NOT sent")

        time.sleep(random.uniform(*message_config['interval']))

# 启动多个发送线程，模拟不同设备
threads = [
    threading.Thread(target=send_frames, args=("ENGINE-ECU", CAN_MESSAGES['ENGINE']), daemon=True),
    threading.Thread(target=send_frames, args=("TRANS-ECU", CAN_MESSAGES['TRANSMISSION']), daemon=True),
    threading.Thread(target=send_frames, args=("BODY-ECU", CAN_MESSAGES['BODY']), daemon=True),
]

for t in threads:
    t.start()

# 主线程保持运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Simulation stopped.")