from scipy.fft import dct
import torch
from einops import rearrange
from torch_dct import dct, idct


class CompressorDCT:
    def __init__(self):
        pass
    
    def compress(self, x, kernel, energy):
        self.num_blocks = []
        for x_dim, kernel_dim in zip(x.shape, kernel):
            assert x_dim % kernel_dim == 0, f"维度 {x_dim} 无法被步长 {kernel_dim} 整除"
            # 每个维度分的块数
            num_block = x_dim // kernel_dim
            self.num_blocks.append(num_block)
        
        x = x.unsqueeze(0)
        x = self.rearrange_nd(x, kernel)
        
        # DCT-II for nd
        x = self.dct_nd(x)

        # 展成一维的，展前注意保留原shape以便后面恢复
        self.origin_shape = x.shape
        x = x.flatten()
        total_len = len(x)

        y = torch.zeros_like(x)

        # 排序索引(升序)
        sort = torch.argsort(torch.abs(x), descending=False)

        # 按索引排序，并保留前n个数
        x = x[sort]

        # compress on energy
        idx = self.save_energy(torch.abs(x), energy)
        s = x[idx:]

        # 计算保存的能量比例和压缩率
        total_energy = torch.sum(torch.abs(x))
        s_energy = torch.sum(torch.abs(s))
        energy_ratio = (s_energy/total_energy)*100
        compress_ratio = (idx/total_len)*100
        print('保留的DCT系数:', idx.item(), '保存的能量:', energy_ratio.item(), ' 压缩率:', compress_ratio.item(),'%')

        y[idx:] = x[idx:]

        iterate = torch.argsort(sort)
        y = y[iterate]

        return y, compress_ratio
    
    def decompress(self, x, kernel):

        # (262144*32768) -> (262144, 64, 8, 8, 8)
        x = x.view(self.origin_shape)

        # DCT-II逆变换torch version
        x = self.idct_nd(x)

        # y: (262144, 64, 8, 8, 8) -> (512, 256, 256, 256)
        y = self.inverse_rearrange_nd(x, kernel, self.num_blocks)

        return y
    
    def rearrange_nd(self, x: torch.Tensor, kernel):

        assert len(kernel) == x.dim() - 1, "步长列表长度需与张量维度数减一一致"
        # 左->
        left_pattern = ["b"]
        for i in range(len(kernel)):
            left_pattern.append(f"(b{i}_num b{i}_size)")
        
        # ->右
        right_blocks = ["b"] + [f"b{i}_num" for i in range(len(kernel))]
        right_sizes = [f"b{i}_size" for i in range(len(kernel))]
        right_pattern = f"({' '.join(right_blocks)}) {' '.join(right_sizes)}"
        
        shape_info = {}
        for i, step in enumerate(kernel):
            shape_info[f"b{i}_size"] = step
        
        x = rearrange(x, " ".join(left_pattern) + " -> " + right_pattern, **shape_info)
        
        return x

    def inverse_rearrange_nd(self, x: torch.Tensor, kernel, num_blocks):

        assert len(kernel) == x.dim() - 1, "步长列表长度需与张量维度数减一一致"

        left_blocks = ["b"] + [f"block_num{i}" for i in range(len(kernel))]
        left_sizes = [f"kernel_dim{i}" for i in range(len(kernel))]
        left_pattern = f"({' '.join(left_blocks)}) {' '.join(left_sizes)}"

        right_pattern = ["b"]
        for i in range(len(kernel)):
            right_pattern.append(f"(block_num{i} kernel_dim{i})")

        shape_info2 = {}
        for i, num_block in enumerate(num_blocks):
            shape_info2[f"block_num{i}"] = num_block 

        x = rearrange(x, left_pattern + " -> " + " ".join(right_pattern), **shape_info2)

        return x.squeeze(0)
    
    def save_energy(self, x, ratio):
        # 前n项和的升序数组
        cumulative_sum = torch.cumsum(x, dim=0)
        threshold = (1-ratio) * torch.sum(x)
        index = torch.searchsorted(cumulative_sum, threshold) + 1
        return index

    def dct_nd(self, x, norm=None):
        # 每个维度做一维DCT-II
        for dim in range(1, x.ndim):  
            x = dct(x.transpose(-1, dim), norm=norm).transpose(-1, dim)
        return x

    def idct_nd(self, x, norm=None):
        for dim in range(1, x.ndim): 
            x = idct(x.transpose(-1, dim), norm=norm).transpose(-1, dim)
        return x
