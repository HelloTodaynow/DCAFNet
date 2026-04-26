# DCAFNet

Official implementation of **"Difference-Gated Interaction and Change-Aware Cross-Temporal Fusion Network for Remote Sensing Image Change Detection"**.

## Overview

DCAFNet is a two-stage coarse-to-fine framework for remote sensing image change detection that simultaneously addresses pseudo-change suppression and missed detection recovery. The network comprises two key components:

- **DGFI (Difference-Gated Feature Interaction):** Employs a change correlation gate derived from bitemporal difference magnitude to adaptively smooth background in unchanged regions while amplifying genuine change signals.
- **CCTF (Change-Aware Cross-Temporal Fusion):** Leverages coarse predictions as semantic guidance to recalibrate backbone features (CGFR) and employs bidirectional cross-temporal attention with learnable adaptive fusion (CTAF) to recover missed detections.

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

