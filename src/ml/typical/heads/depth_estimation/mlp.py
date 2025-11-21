from typing import Callable, List, Optional, Tuple

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
    ):
        super().__init__()
        
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.use_bilinear = use_bilinear
        self.act = act
        
        h, w = input_res
        self.feat_h = h // patch_size
        self.feat_w = w // patch_size
        
        if hidden_features is None:
           hidden_features = []
        
        layers = []
        in_dim = embed_size
        
        for hidden_dim in hidden_features:
            layers.extend([
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=True),
                self.act()
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Conv2d(in_dim, 1, kernel_size=1, bias=True))
        
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
        
        x = x.transpose(1, 2)  # [batch_size, embed_size, num_patches]
        x = x.view(batch_size, embed_size, self.feat_h, self.feat_w)
        
        depth_low_res = self.mlp(x)
        
        depth_map = self.upsample(depth_low_res)
        
        return depth_map
