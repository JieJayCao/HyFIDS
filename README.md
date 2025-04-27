# HyFIDS: Hybrid Frequency-Aware Lightweight Intrusion Detection for Internet of Vehicles

This repository contains the official implementation of the paper:  
**HyFIDS: Hybrid Frequency-Aware Lightweight Intrusion Detection for Internet of Vehicles**.

> **Citation**:  
> Jie Cao, Zelin Zhao, and Jianbing Ni.  
> *HyFIDS: Hybrid Frequency-Aware Lightweight Intrusion Detection for Internet of Vehicles.*  
> [Under review of IEEE GlobeCom 2025].

## Overview

HyFIDS is a lightweight Intrusion Detection System (IDS) designed specifically for Internet of Vehicles (IoV) scenarios.  
It integrates spatial and frequency domain features to efficiently detect anomalies in both intra-vehicle and inter-vehicle networks, achieving high detection performance with minimal computational overhead.

Key Features:
- Hybrid feature extraction combining raw byte embeddings and 2D Fourier representations.
- Extremely low FLOPs and model size for resource-constrained IoV devices.
- Competitive accuracy compared to state-of-the-art (SOTA) methods.

## Project Structure

```
HyFIDS/
├── dataset/          # Sample datasets and data processing scripts
├── model/            # Model structure and training
├── deploy/           # Deployment simulation program
├── requirements.txt  # Python dependencies
└── README.md         # Project description
```

## Installation

Clone this repository:
```bash
git clone https://github.com/JieJayCao/HyFIDS.git
cd HyFIDS
```

Create a virtual environment (optional but recommended):
```bash
python3 -m venv hyfids-env
source hyfids-env/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

We use both CAN traffic and IP packet datasets for evaluation.  
For demonstration purposes, a small sample dataset is provided in the `dataset/` folder.

If you wish to reproduce full-scale experiments, please refer to:
- [Ton-IoT](https://research.unsw.edu.au/projects/toniot-datasets) for IP traffic
- [IoT-23](https://www.stratosphereips.org/datasets-iot23) for IP traffic
- [Carhacking](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) for CAN bus attacks
- [CIC-IoV](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) for CAN bus attacks


## Usage

**Training and testing the model**:
```bash
python HyFIDS.py
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
