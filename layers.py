import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed: int):
        super().__init__()

        assert d_embed % 2 == 0, "Embedding dimension must be an even number"
        self.d_embed = d_embed

        base_idxs = torch.arange(d_embed // 2)
        inv_freqs = torch.exp(- math.log(10000) * 2 * base_idxs / d_embed)

        self.register_buffer("inverse_frequencies", inv_freqs)


    def forward(self, timestep: torch.Tensor):
        t_divided_by_inv_freqs = self.inverse_frequencies.unsqueeze(0) * timestep.unsqueeze(1)

        even_embeds = torch.sin(t_divided_by_inv_freqs)   # Apply sin on even positions
        odd_embeds = torch.cos(t_divided_by_inv_freqs)   # Apply cos on odd positions

        stacked_embeds = torch.cat((even_embeds, odd_embeds), dim=-1)
        pos_embeds = stacked_embeds.view(timestep.shape[0], self.d_embed)

        return pos_embeds


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_embed: int, n_groups: int):
        super().__init__()

        self.img_prep_layers = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.embed_prep_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_embed, out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # Input and output shape matching
        if in_channels != out_channels:
            self.channel_match_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_match_layer = nn.Identity()


    def forward(self, x, time_embeds):
        org_data = x

        time_embeds = self.embed_prep_layers(time_embeds)
        time_embeds = time_embeds.unsqueeze(2).unsqueeze(3)

        x = time_embeds + self.img_prep_layers(x)

        x = self.out_layers(x)
        x = x + self.channel_match_layer(org_data)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_groups: int):
        super().__init__()

        self.norm_layer = nn.GroupNorm(n_groups, n_channels)

        self.attn_layer = nn.MultiheadAttention(embed_dim=n_channels, num_heads=1, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        org_x = x

        x = self.norm_layer(x)

        # x is an image with (B, C, H, W) that needs to be rolled into (B, H*W, C) for Attention
        # Attention input for NLP is (B, seq_len, num_embeds)
        x = x.view(b, c, h*w).transpose(1, 2)
        attn_out, _ = self.attn_layer(x, x, x)

        # Reverse the input operations of swap and flatten
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)

        out = org_x + attn_out
        return out
