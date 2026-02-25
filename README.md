# DualStreamSeg-Boundary-Guided-Dual-Encoder-Network-for-Blood-Cell-Segmentation
DualStreamSeg: Boundary-Guided Dual-Encoder Network for Blood Cell Segmentation


Official PyTorch implementation of the DualStreamSeg model for accurate and efficient blood cell segmentation.
## Features

* Dual encoder architecture (ResNet50 + ResNet18, [He et al., 2016])
* Boundary-guided fusion module
* Lightweight decoder [Ronneberger et al., 2015]
* High segmentation accuracy with low computational cost

## Architecture


## Installation

```bash
pip install -r requirements.txt
```

## Dataset

![image](https://github.com/yourusername/yourrepo/assets/12345678/abcd-image.png)<img width="4267" height="2167" alt="22222" src="https://github.com/user-attachments/assets/17d6d2c3-d4c9-456a-9020-9c43d7015510" />


```
dataset/
  train/
    original/
    mask/
  test/
    original/
    mask/
```

## Training

```bash
python train.py
```

## Testing

```bash
python test.py --model pretrained/model.pth --image test.png
```

## Results
## Model Comparison on BCCD Dataset

| Model | Parameters (M) | Size (MB) | FLOPs (G) | Dice (%) | Accuracy (%) |
|------|----------------|-----------|-----------|----------|--------------|
| U-Net | 7.76 | 30 | 15.2 | 94.85 | 97.31 |
| Attention U-Net | 10.4 | 140 | 18.1 | 94.51 | 96.78 |
| DeepLabV3-ResNet50 | 39.6 | 150 | 48.7 | 94.69 | 97.03 |
| nnU-Net | 32 | 160 | 50 | 94.79 | 97.26 |
| TransUNet | 105 | 400 | 85.3 | 94.88 | 97.34 |
| SAM Encoder | 91.7 | 375 | 92.5 | 94.71 | 97.10 |
| **DualStreamSeg (Proposed)** | **12.70** | **48.6** | **36.86** | **95.00** | **97.47** |

## Citation
```
@article{DualStreamSeg2026,
  title={Boundary-Guided Dual-Encoder Network for Blood Cell Segmentation},
  author={},
  journal={},
  year={2026}
}
```
