import dpkt


def packet_parse(packet):
    eth = dpkt.ethernet.Ethernet(packet)
    if not isinstance(eth.data, dpkt.ip.IP):
        return False, None  # 非IP数据包，不予处理

    ip = eth.data
    # 过滤出非DHCP (UDP 67, 68), DNS (UDP 53), ICMP, ARP 的数据包
    if isinstance(ip.data, dpkt.udp.UDP):
        udp = ip.data
        udp.data = (b'\x00' * 12) + udp.data
        ip.data = udp
        return True, ip
    return True, ip 

def packet_filter(packet):
    eth = dpkt.ethernet.Ethernet(packet)
    if not isinstance(eth.data, dpkt.ip.IP):
        return False, None  # 非IP数据包，不予处理

    ip = eth.data

    # 过滤出非DHCP (UDP 67, 68), DNS (UDP 53), ICMP, ARP 的数据包
    if isinstance(ip.data, dpkt.udp.UDP):
        # udp = ip.data
        # if udp.sport in (67, 68) or udp.dport in (67, 68):  # DHCP
        #     return False, None
        # if udp.sport == 53 or udp.dport == 53:  # DNS
        #     return False, None
        
        # 对UDP首部填充至20字节
        udp.data = (b'\x00' * 12) + udp.data
        ip.data = udp
        return True, ip
    # elif isinstance(ip.data, dpkt.icmp.ICMP):
    #     return False, None  # ICMP
    elif isinstance(ip.data, dpkt.tcp.TCP):
        tcp = ip.data
        # # 检查TCP标志位，过滤掉SYN, FIN, RST包
        # if (tcp.flags & (dpkt.tcp.TH_SYN | dpkt.tcp.TH_FIN | dpkt.tcp.TH_RST)) != 0:
        #     return False, None
        return True, ip

    return True, ip  # 默认保留此数据包进行进一步处理