from typing import List, Optional, Tuple

import torch
from torch import nn


class DepthEstimationMLPHead(nn.Module):
    def __init__(
        self,
        embed_size: int,
        patch_size: int,
        input_res: Tuple[int, int],
        hidden_features: Optional[List[int]] = None,
        use_bilinear: bool = True,
        act: type[nn.Module] = nn.ReLU,
        out_act: nn.Module = nn.Identity(),
    ):
        super().__init__()
        
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.use_bilinear = use_bilinear
        self.act = act
        self.out_act = out_act
        
        h, w = input_res
        self.feat_h = h // patch_size
        self.feat_w = w // patch_size
        
        if hidden_features is None:
           hidden_features = []
        
        layers = []
        in_dim = embed_size
        
        for hidden_dim in hidden_features:
            conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            layers.extend([conv, self.act()])
            in_dim = hidden_dim
        
        final_conv = nn.Conv2d(in_dim, 1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(final_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(final_conv.bias, -1.0) 
        layers.append(final_conv)
        
        self.mlp = nn.Sequential(*layers)
        
        if use_bilinear:
            self.upsample = nn.Upsample(
                size=input_res, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            scale_factor = patch_size
            self.upsample = nn.ConvTranspose2d(
                1, 1, 
                kernel_size=patch_size, 
                stride=patch_size, 
                padding=0,
                output_padding=0,
                bias=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, embed_size = x.shape
        
        if num_patches == self.feat_h * self.feat_w + 1:
            x = x[:, 1:, :]  # Remove class token
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, embed_size, self.feat_h, self.feat_w)
        
        depth_low_res = self.mlp(x)
        depth_map = self.upsample(depth_low_res)
        
        return self.out_act(depth_map)
