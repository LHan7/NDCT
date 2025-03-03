## NDCT
A compressor for nd-tensor with discrete cosine transform
## Install
```python
pip install torchpress
```
## Usage
```python
import torch
from torchpress import ndct

x= torch.randn(64, 16, 8, 8, 4, 2)
kernel = [8, 16, 4, 4, 2, 2]
energy = 0.99

cp = ndct.CompressorDCT()

y, compress_ratio = cp.compress(x, kernel, energy)

z = cp.decompress(y, kernel)
```
