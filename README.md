# DeSegaMamba

## DeSegaMamba: Rethinking Vision Mamba Design for Medical Image Segmentation

This repository contains the implementation of **DeSegaMamba** (Deformable Semantic Aggregation Mamba) for medical image segmentation.

![DeSegaMamba Architecture](./images/352a6bd276e8cef1699b7a4decf01cd2.png)



## Dataset Support

- ISIC 2018 (default)
- ISIC 2017
- Configurable for other medical imaging datasets

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Configuration
Modify `configs/config_setting_DSAM.py` for your specific requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.