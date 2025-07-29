# DeSegaMamba: Rethinking Vision Mamba Design for Medical Image Segmentation

This repository contains the implementation of **DeSegaMamba** (Deformable Semantic Aggregation Mamba) for medical image segmentation.

![DeSegaMamba Architecture](./images/352a6bd276e8cef1699b7a4decf01cd2.png)

## 🔍 Overview

DeSegaMamba introduces a novel paradigm for Vision Mamba in medical image segmentation by addressing the fundamental limitations of spatial locality in traditional scanning mechanisms. Our key insight is that semantic similarity should guide the sequence reorganization process rather than rigid spatial ordering.


## 📊 Experimental Results

### ISIC 2018 Dataset
| Model | Dice Score | IoU | Sensitivity | Specificity |
|-------|-------------|-----|-------------|-------------|
nU-Net | 0.896 | 0.821 | 0.891 | 0.973 |
TransUNet | 0.901 | 0.829 | 0.894 | 0.978 |
Swin-Unet | 0.904 | 0.833 | 0.902 | 0.981 |
**DeSegaMamba** | **0.918** | **0.851** | **0.915** | **0.985** |

### ISIC 2017 Dataset
| Model | Dice Score | IoU | Sensitivity | Specificity |
|-------|-------------|-----|-------------|-------------|
nU-Net | 0.882 | 0.804 | 0.878 | 0.971 |
TransUNet | 0.887 | 0.811 | 0.884 | 0.974 |
Swin-Unet | 0.891 | 0.816 | 0.889 | 0.977 |
**DeSegaMamba** | **0.905** | **0.837** | **0.902** | **0.982** |

## 🛠️ Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Data Preparation
```bash
# Organize dataset structure
data/
├── ISIC2018/
│   ├── Training/
│   │   ├── Images/
│   │   └── Masks/
│   └── Validation/
│       ├── Images/
│       └── Masks/
```

### Training
```bash
# Start training
python train.py

# Monitor training progress
tensorboard --logdir runs/
```

### Configuration
Key hyperparameters in `configs/config_setting_DSAM.py`:
- **Input size**: 512×512
- **Batch size**: 8
- **Learning rate**: 1e-4 (Adam optimizer)
- **Epochs**: 120
- **Scheduler**: CosineAnnealingLR

## 🔧 Model Variants

| Model | Parameters | FLOPs | FPS | Dice Score |
|-------|------------|-------|-----|-------------|
DeSegaMamba-T | 8.7M | 4.2G | 45 | 0.905 |
DeSegaMamba-S | 15.2M | 7.8G | 38 | 0.912 |
DeSegaMamba-B | 27.8M | 14.3G | 32 | 0.918 |

## 📄 License

This project is licensed under the MIT License.