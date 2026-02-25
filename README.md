# DualStreamSeg-Boundary-Guided-Dual-Encoder-Network-for-Blood-Cell-Segmentation
DualStreamSeg: Boundary-Guided Dual-Encoder Network for Blood Cell Segmentation
# DualStreamSeg: Boundary-Guided Dual-Encoder Network for Blood Cell Segmentation

Official PyTorch implementation of the DualStreamSeg model for accurate and efficient blood cell segmentation.

## Features

* Dual encoder architecture (ResNet50 + ResNet18)
* Boundary-guided fusion module
* Lightweight decoder
* High segmentation accuracy with low computational cost

## Architecture


## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Example structure:

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

| Model         | Dice   | Accuracy |
| ------------- | ------ | -------- |
| DualStreamSeg | 95.00% | 97.47%   |

## Citation

```
@article{DualStreamSeg2026,
  title={Boundary-Guided Dual-Encoder Network for Blood Cell Segmentation},
  author={Your Name},
  journal={},
  year={2026}
}
```
