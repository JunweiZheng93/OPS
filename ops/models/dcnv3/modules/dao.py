from torch import nn
import torch
from einops import rearrange, repeat
from .dcnv3 import DCNv3


class DAO(nn.Module):
    def __init__(self,
                 channels=3,  # both input and output channels
                 group=3,  # channel has to be divisible by group
                 kernel_size=3,  # kernel size of DCNv3
                 stride=1,  # stride of DCNv3
                 pad=1,  # padding of DCNv3
                 dilation=1,  # dilation of DCNv3
                 offset_scale=1.0,  # offset scale of DCNv3
                 act_layer='GELU',  # activation layer of DCNv3
                 norm_layer='LN',  # normalization layer of DCNv3
                 dw_kernel_size=5,  # kernel size of depthwise conv of DCNv3
                 center_feature_scale=True,  # group feature subtract center feature before output projection in DCNv3
                 pa_kernel_size=3,  # patch attention kernel size
                 pa_norm_layer='Softmax',  # patch attention normalization layer, Softmax or Norm
                 ):
        super().__init__()
        self.pa_kernel_size = pa_kernel_size
        self.pa_norm_layer = pa_norm_layer
        self.dcn_v3 = DCNv3(channels, kernel_size, dw_kernel_size, stride,
                            pad, dilation, group, offset_scale, act_layer,
                            norm_layer, center_feature_scale)

    def forward(self, x):
        """
        input.shape == (N, H, W, C)
        output.shape == (N, H, W, C)
        """
        x1 = self.dcn_v3(x)  # x1.shape == (N, H, W, C)
        mask = self._patch_attention(x1, self.pa_kernel_size, self.pa_norm_layer)  # mask.shape == (N, C, H, W)
        x1 = x1 * mask  # x1.shape == (N, H, W, C)
        x = x + x1  # x.shape == (N, H, W, C)
        return x

    def _patch_attention(self, x, kernel_size, norm_layer):
        """
        input: (N, H, W, C)
        output: (N, H, W, C)
        """
        x = rearrange(x, 'N H W C -> N C H W')
        N, C, H, W = x.shape
        pad = (kernel_size - 1) // 2
        neighbors = torch.nn.functional.unfold(x, kernel_size=kernel_size, padding=pad, stride=1)
        neighbors = rearrange(neighbors, 'N (C K1 K2) (H W) -> N H W C (K1 K2)', C=C, K1=kernel_size, K2=kernel_size, H=H, W=W)
        x = rearrange(x, 'N C H W -> N H W 1 C')
        if norm_layer == 'Softmax':
            x = x @ neighbors  # x.shape == (N, H, W, 1, K1K2)
            x = torch.nn.functional.softmax(x, dim=-1)
        elif norm_layer == 'Norm':
            norm1 = torch.norm(x, dim=-1)  # norm1.shape == (N, H, W, 1)
            norm2 = torch.norm(neighbors, dim=-2)  # norm2.shape == (N, H, W, K1K2)
            norm = rearrange(norm1 * norm2, 'N H W K1K2 -> N H W 1 K1K2')  # norm.shape == (N, H, W, 1, K1K2)
            x = x @ neighbors  # x.shape == (N, H, W, 1, K1K2)
            x = x / norm
        else:
            raise NotImplementedError
        mask = rearrange(torch.std(x, dim=-1), 'N H W 1 -> N H W')
        mask = repeat(mask, 'N H W -> N H W C', C=C)
        return mask
