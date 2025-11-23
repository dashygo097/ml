import math
from typing import List, Optional, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BaseDepthHead(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        input_res: Tuple[int, int],
        decoder_channels: Optional[List[int]] = None,
        out_act: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.input_res = input_res
        
        h, w = input_res
        self.feat_h = h // patch_size
        self.feat_w = w // patch_size
        
        num_stages = int(math.log2(patch_size)) 
        
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16][:num_stages + 1]
        
        if len(decoder_channels) < num_stages + 1:
            last_ch = decoder_channels[-1]
            while len(decoder_channels) < num_stages + 1:
                decoder_channels = decoder_channels + [last_ch]
        elif len(decoder_channels) > num_stages + 1:
            decoder_channels = decoder_channels[:num_stages + 1]
        
        self.initial_proj = nn.Sequential(
            nn.Conv2d(embed_size, decoder_channels[0], kernel_size=1, bias=True),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_stages):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(DepthDecoderBlock(in_ch, out_ch))
        
        final_channels = decoder_channels[num_stages]
        self.depth_head = nn.Sequential(
            ConvBlock(final_channels, 32, kernel_size=3),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        self.out_act = out_act
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embed_size = x.shape
        
        if num_patches == self.feat_h * self.feat_w + 1:
            x = x[:, 1:, :]
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, embed_size, self.feat_h, self.feat_w)
        
        x = self.initial_proj(x)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        depth_map = self.depth_head(x)
        depth_map = self.out_act(depth_map)
        
        return depth_map
