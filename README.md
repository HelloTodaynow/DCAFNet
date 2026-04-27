# DCAFNet

Official implementation of **"Difference-Gated Interaction and Change-Aware Cross-Temporal Fusion Network for Remote Sensing Image Change Detection"**.

## Overview

DCAFNet introduces a unified coarse-to-fine architecture designed for remote sensing change detection, mitigate two long-standing challenges: false alarms from environmental variations and overlooked subtle changes.

The framework is built upon two core modules:

- **DGFI (Difference-Gated Feature Interaction):** Computes channel-wise mean absolute differences between bitemporal features to produce a change correlation gate. This gate drives adaptive background suppression in stable regions and response amplification where genuine changes occur.
- **CCTF (Change-Aware Cross-Temporal Fusion):** Consists of two sub-modules — CGFR, which uses preliminary predictions to semantically recalibrate encoder features via learnable foreground-background weighting, and CTAF, which performs bidirectional cross-temporal querying followed by level-specific adaptive fusion to capture complementary temporal cues and recover initially missed changes.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- einops
- numpy
- tqdm

## Datasets

Organize the datasets as follows:

```
data/
├── LEVIR-CD/
│   ├── train/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   ├── val/
│   └── test/
├── WHU-CD/
└── CDD/
```

## Usage

### Training

```bash
python train.py --data_name LEVIR --gpu_id 0
```

### Testing

```bash
python test.py --data_name LEVIR --gpu_id 0
```

## Code Availability

The complete source code, including training scripts, model definitions, will be released upon acceptance of the paper.
