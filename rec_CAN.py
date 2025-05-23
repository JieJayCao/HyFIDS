import can
import torch
import time
bus = can.interface.Bus(channel='vcan0', bustype='socketcan')
print("Listening on vcan0...")

f = open("can_preprocess.csv",'a')
f.write("CAN Pre time\n")
while True:
    
    msg = bus.recv()
    start_time = time.time()
    if msg:
        arbitration_id = msg.arbitration_id & 0x7FF
        id_high = (arbitration_id >> 8) & 0xFF
        id_low = arbitration_id & 0xFF
        data = [x & 0xFF for x in msg.data]
        #print(f"ID高字节={id_high}, ID低字节={id_low}, Data={data}")
        input_tensor = torch.LongTensor([id_high, id_low] + data + [0]*(10-len(data)))
        f.write(f"{time.time()-start_time}\n")
        #print(input_tensor)