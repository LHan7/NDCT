## NDCT-II
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
dct_kernel = [8, 16, 4, 4, 2, 2]
save_energy = 0.99

cp = ndct.CompressorDCT()

y, compress_ratio = cp.compress(x, dct_kernel, save_energy)

z = cp.decompress(y, kernel)
```
