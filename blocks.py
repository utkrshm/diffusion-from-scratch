import torch
import torch.nn as nn

from layers import (
    ResNetBlock,
    AttentionBlock,
)

class DownBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        d_embed: int,
        has_attn: bool = False,
        n_groups: int = 8,
    ):
        super().__init__()

        self.res_block_1 = ResNetBlock(in_channels, out_channels, d_embed, n_groups)
        self.attn_1 = AttentionBlock(out_channels, n_groups) if has_attn else nn.Identity()

        self.res_block_2 = ResNetBlock(out_channels, out_channels, d_embed, n_groups)
        self.attn_2 = AttentionBlock(out_channels, n_groups) if has_attn else nn.Identity()


    def forward(self, x, time_embed):
        x = self.res_block_1(x, time_embed)
        x = self.attn_1(x)
        x = self.res_block_2(x, time_embed)
        x = self.attn_2(x)
        return x


class MidBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
        return x



class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
        return x
