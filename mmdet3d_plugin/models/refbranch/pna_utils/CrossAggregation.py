import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_feedforward_network

class CrossAggregationModule(nn.Module):
    def __init__(
        self,
        embed_dims=None,
        use_residual=True,
        ffn_cfg=None,
        num_heads=8,
        bias=False,
    ):
        super(CrossAggregationModule, self).__init__()

        # Norm Later
        self.norm1 =nn.InstanceNorm3d(embed_dims)
        self.norm2 =nn.InstanceNorm3d(embed_dims)

    def forward(self, x, skip):
        x = self.norm1(x)
        skip = self.norm2(skip)

if __name__ == "__main__":
    ffn_cfg=dict(
        type='FFN',
        embed_dims=64,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type='ReLU', inplace=True),
        ffn_drop=0.1,
        add_identity=True
    )

    CAM = CrossAggregationModule(
        embed_dims=64, 
        use_residual=True,
        ffn_cfg=ffn_cfg,
        num_heads=8,
        bias=False
    )

    tensor_shape = (1, 64, 256, 256, 32)
    x = torch.randn(tensor_shape)

    y = CAM(x)
