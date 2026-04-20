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

        self.downsample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)


    def forward(self, x, time_embed):
        skips = []          # For residual connections with the Up block

        x = self.res_block_1(x, time_embed)
        x = self.attn_1(x)
        skips.append(x)

        x = self.res_block_2(x, time_embed)
        x = self.attn_2(x)
        skips.append(x)

        x = self.downsample_layer(x)
        skips.append(x)

        return x, skips


class MidBlock(nn.Module):
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

    def forward(self, x, time_embed):
        x = self.res_block_1(x, time_embed)
        x = self.attn_1(x)
        x = self.res_block_2(x, time_embed)
        return x


class UpBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        d_embed: int,
        n_groups: int = 8,
        has_attn: bool = False,
        up_sample: bool = True,
    ):
        super().__init__()

        self.res_block_1 = ResNetBlock(in_channels * 2, out_channels, d_embed, n_groups)
        self.attn_1 = AttentionBlock(out_channels, n_groups) if has_attn else nn.Identity()

        self.res_block_2 = ResNetBlock(out_channels * 2, out_channels, d_embed, n_groups)
        self.attn_2 = AttentionBlock(out_channels, n_groups) if has_attn else nn.Identity()

        self.upsample_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity()


    def forward(self, x, time_embed, down_skips):
        down_x = down_skips.pop()
        x = torch.cat([x, down_x], dim=1)
        x = self.res_block_1(x, time_embed)
        x = self.attn_1(x)

        down_x = down_skips.pop()
        x = torch.cat([x, down_x], dim=1)
        x = self.res_block_2(x, time_embed)
        x = self.attn_2(x)

        x = self.upsample_layer(x)

        return x
