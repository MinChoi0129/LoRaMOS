import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import backbone_moving
from networks.backbone_movable import MetaKernel, ResContextBlock, ResBlock, UpBlock


class MovingNet(nn.Module):
    """Cartesian BEV Encoder-Decoder (Feature Pyramid + DeformAttn)."""

    def __init__(self, in_channels):
        super(MovingNet, self).__init__()
        block = backbone_moving.BasicBlock

        self.enc1 = self._make_layer(block, in_channels, 64, num_blocks=3, stride=2)
        self.enc2 = self._make_layer(block, 64, 128, num_blocks=3, stride=2)
        self.enc3 = self._make_layer(block, 128, 256, num_blocks=4, stride=2, dilation=2)
        self.bottleneck_attn = backbone_moving.DeformAttnBottleneck(
            in_channels=256,
            d_model=128,
            d_ffn=512,
            n_heads=4,
            n_points=4,
            num_layers=2,
        )

        self.dec1 = backbone_moving.BasicConv2d(64 + 128 + 256, 128, kernel_size=3, padding=1)
        self.dec2 = backbone_moving.BasicConv2d(128, 32, kernel_size=3, padding=1)
        self.dec_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=2, dilation=1):
        layer = []
        layer.append(backbone_moving.DownSample2D(in_planes, out_planes, stride=stride))
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, bev_feat, movable_mask_bev):
        e1 = self.enc1(bev_feat)
        m1 = F.max_pool2d(movable_mask_bev, kernel_size=2, stride=2)
        e1 = e1 * (1 + m1)

        e2 = self.enc2(e1)
        m2 = F.max_pool2d(m1, kernel_size=2, stride=2)
        e2 = e2 * (1 + m2)

        e3 = self.enc3(e2)
        e3 = self.bottleneck_attn(e3)

        target_size = e1.shape[2:]
        e2_up = F.interpolate(e2, size=target_size, mode="bilinear", align_corners=True)
        e3_up = F.interpolate(e3, size=target_size, mode="bilinear", align_corners=True)

        dec = torch.cat([e1, e2_up, e3_up], dim=1)
        dec = self.dec1(dec)
        dec = self.dec2(dec)
        dec = self.dec_up(dec)

        return dec


class MovableNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=3, num_batch=6, height=64, width=2048):
        super(MovableNet, self).__init__()

        self.downCntx = ResContextBlock(in_ch, 32)
        self.metaConv = MetaKernel(num_batch=num_batch, feat_height=height, feat_width=width, coord_channels=in_ch)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4))
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True, kernel_size=(2, 4))

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock3 = UpBlock(2 * 32, 32, 0.2, drop_out=False)
        self.logits = nn.Conv2d(32, out_ch, kernel_size=(1, 1))

    def forward(self, rv_input):
        B = rv_input.size(0)
        self.metaConv.update_num_batch(B)

        x = self.downCntx(rv_input)
        x = self.metaConv(data=x, coord_data=rv_input, data_channels=x.size(1), coord_channels=rv_input.size(1), kernel_size=3)
        x = self.downCntx2(x)
        x = self.downCntx3(x)

        down0c, down0b = self.resBlock1(x)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)

        up3 = self.upBlock1(down3b, down2b)
        up2 = self.upBlock2(up3, down1b)
        up1 = self.upBlock3(up2, down0b)

        logits = self.logits(up1)
        return logits
