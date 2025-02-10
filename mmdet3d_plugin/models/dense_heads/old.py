import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS

class ResidualConvBlock(nn.Module):
    """带残差连接的3D卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding),
            nn.InstanceNorm3d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.leaky_relu(x + residual, 0.1)

class EncoderBlock(nn.Module):
    """下采样编码块"""
    def __init__(self, in_channels, out_channels, z_down=True):
        super().__init__()
        self.down_conv = ResidualConvBlock(in_channels, out_channels, stride=2)
        self.conv = ResidualConvBlock(out_channels, out_channels)
        if z_down:
            self.pool = nn.MaxPool3d((2, 2, 2))
        else:
            self.pool = nn.MaxPool3d((2, 2, 1))

    def forward(self, x):
        x = self.down_conv(x)
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled  # 返回特征和池化结果

class DecoderBlock(nn.Module):
    """上采样解码块"""
    def __init__(self, in_channels, out_channels, z_up=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, 
                                   kernel_size=(2,2,2) if z_up else (2,2,1), 
                                   stride=(2,2,2) if z_up else (2,2,1))
        self.conv = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class RRM(nn.Module):
    """残差优化模块（来自题目图示）"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualConvBlock(channels, channels),
            ResidualConvBlock(channels, channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=64, base_channels=64, num_classes=1, z_down=True):
        super().__init__()
        # 编码器
        self.enc1 = EncoderBlock(in_channels, base_channels, z_down)
        self.enc2 = EncoderBlock(base_channels, base_channels*2, z_down)
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4, z_down)
        
        # 中间层
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(base_channels*4, base_channels*8),
            RRM(base_channels*8)
        )
        
        # 解码器
        self.dec3 = DecoderBlock(base_channels*8, base_channels*4, z_down)
        self.dec2 = DecoderBlock(base_channels*4, base_channels*2, z_down)
        self.dec1 = DecoderBlock(base_channels*2, base_channels, z_down)
        
        # 最终输出
        self.final_conv = nn.Conv3d(base_channels, num_classes, 1)

    def forward(self, x):
        '''
        input:
            x: torch.Size([1, 64, 256, 256, 32])
        '''
        print(x.size())
        skip1, x = self.enc1(x)  # skip1: ([1, 64, 128, 128, 16]) / x: ([1, 64, 64, 64, 8])
        skip2, x = self.enc2(x)  # skip2: ([1, 128, 32, 32, 4]) / x: ([1, 128, 16, 16, 2])
        print('=============')
        print(skip2.size())
        print(x.size())
        skip3, x = self.enc3(x)  # skip3: [B,256,64,64,8]
        
        # 中间层
        x = self.bottleneck(x)   # [B,512,32,32,4]
        
        # 解码器
        x = self.dec3(x, skip3)  # [B,256,64,64,8]
        x = self.dec2(x, skip2)  # [B,128,128,128,16]
        x = self.dec1(x, skip1)  # [B,64,256,256,32]
        
        return self.final_conv(x)  # 输出形状: [B, num_classes, 256,256,32]


@HEADS.register_module()
class RefHead(nn.Module):
    def __init__(
        self,
        num_class,
        geo_feat_channels,
        train_cfg=None,
        test_cfg=None
    ):
        super(RefHead, self).__init__()
        
        self.embedding = nn.Embedding(num_class, geo_feat_channels)  # [B, D, H, W, C]
        self.unet = UNet3D(in_channels=geo_feat_channels)

    
    def forward(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        return self.unet(x)
    
if __name__ == '__main__':
    REF = RefHead(
        num_class = 20,
        geo_feat_channels = 64
    )
    
    tensor = torch.randint(low=0, high=20, size=(1, 256, 256, 32), dtype=torch.long)
    
    x = REF(tensor)