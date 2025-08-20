# HMB Helpers Package

A collection of helper modules for image processing, segmentation, and deep learning workflows in PyTorch and beyond.

## Features
- Custom PyTorch loss functions for segmentation
- Image comparison and normalization utilities
- PDF and text helpers
- Performance metrics
- And more!

## Quickstart

### Installation

```bash
pip install -e .
```
Or, if you want to install dependencies:
```bash
pip install -r requirements.txt
```

### Usage Example

#### PyTorch Segmentation Losses
```python
import torch
from HMB.PyTorchSegmentationLosses import DiceLoss, DiceBCELoss

# Simulate model output (logits) and ground truth
predictions = torch.randn(1, 1, 256, 256).float()
targets = torch.randint(0, 2, (1, 1, 256, 256)).float()

criterion = DiceBCELoss()
loss = criterion(predictions, targets)
print(f"DiceBCE Loss: {loss.item()}")
```

### Other Modules
Refer to the documentation in the `source/` folder for details on other modules and their usage.

## Documentation
- HTML docs: `build/html/index.html`
- Source docs: `source/`

## Citation & License
Refer to the header in each module and the LICENSE file for citation and usage permissions.

---
For questions or contributions, please contact the author or open an issue.
